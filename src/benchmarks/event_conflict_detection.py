"""
Event Conflict Detection (Model-Based Reflex Agent) benchmark for AgenticEvals.

8 hand-authored scenarios, 4 turns each. Each turn provides raw distributed-system
state only. The model must infer incident tags over time.

Score: final-turn micro-F1 over a closed tag set.
Diagnostics: per-turn precision/recall/F1, flip-flops, monotonic recall,
spurious persistence, output tokens, execution time (net of 15s sleeps).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from ..benchmark.base import BaseBenchmark, Task, TaskResult, BenchmarkConfig, AgentType
from ..benchmark.registry import benchmark
from ..models.base import BaseModel, ModelResponse
from ..utils.logging import get_logger


logger = get_logger(__name__)


# Closed tag set used across all scenarios
ALLOWED_TAGS: List[str] = [
    "version_skew",
    "config_drift",
    "checksum_mismatch",
    "schema_incompatibility",
    "quorum_mismatch",
    "split_brain",
    "time_skew",
    "replica_lag_exceeded",
    "commit_mismatch",
]

CONFIDENCE_VALUES: Set[str] = {"low", "medium", "high"}


@dataclass
class ScenarioSpec:
    name: str
    num_turns: int
    positives: Set[str]
    raw_states: List[str]  # one per turn, full history will be provided each turn


def _compute_micro_prf(
    predicted: Set[str], truth: Set[str]
) -> Tuple[float, float, float, int, int, int]:
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return precision, recall, f1, tp, fp, fn


def _build_policy_text() -> str:
    return (
        "Invariants and tolerances:\n"
        "- Version skew: MAJOR.MINOR must match across replicas; PATCH may vary.\n"
        "- Checksums: equal version+commit implies equal checksum.\n"
        "- Schema: service version must satisfy required schema range.\n"
        "- Quorum view: all nodes agree on member set; one unique leader per term.\n"
        "- Split-brain: multiple leaders observed in the same term.\n"
        "- Time sync: |ntp_offset_ms| <= 50 is OK; > 50 is time_skew.\n"
        "- Replica lag: lag_entries <= 100 is OK; > 100 is replica_lag_exceeded.\n"
        "- Commit: deployed commit must be an ancestor of the approved release; else commit_mismatch.\n"
        "- Config drift: critical params per role must match cluster policy.\n"
        "If a datum relevant to a tag is absent from the provided state, treat it as no evidence for that tag (not evidence that the tag is present). Do not infer or hallucinate absent values.\n"
    )


def _format_allowed_tags() -> str:
    return f"Allowed tags: [{', '.join(ALLOWED_TAGS)}]"


def _label_truncate_note(text: str, step: int, max_chars: int = 240) -> str:
    prefix = f"S{step}: "
    body = (text or "").strip()
    cap = max_chars - len(prefix)
    if cap <= 0:
        return prefix[:max_chars]
    # Keep the beginning of the note to preserve initial context
    return prefix + (body if len(body) <= cap else body[:cap])


def _make_scenarios() -> List[ScenarioSpec]:
    S = ScenarioSpec
    def B(turn: int, content: str) -> str:
        return f"TURN {turn} SYSTEM STATE\n{content.strip()}"

    scenarios: List[ScenarioSpec] = []

    # Scenario 1: Version skew + split_brain (4 turns, full state each turn)
    scenarios.append(S(
        name="Scenario 1: Version skew then dual leaders",
        num_turns=4,
        positives={"version_skew", "split_brain"},
        raw_states=[
            B(1, """
term: 41
leader_claims: ["n2"]
membership_views: {n1:["n1","n2","n3","n4"], n2:["n1","n2","n3","n4"], n3:["n1","n2","n3","n4"], n4:["n1","n2","n3","n4"]}
services:
  svc-auth:
    - {node:"n1", version:"1.5.1", commit:"C210", checksum:"sha256:A"}
    - {node:"n4", version:"1.5.2", commit:"C210", checksum:"sha256:A"}
  svc-api:
    - {node:"n2", version:"1.6.0", commit:"C300", checksum:"sha256:Z"}
db_schema_version: {n1:16, n2:16, n3:16, n4:16}
ntp_offsets_ms: {n1:41, n2:42, n3:43, n4:44}
replication: {n1:{last_applied:1000, commit_index:1100}, n2:{last_applied:1090, commit_index:1100}, n3:{last_applied:1085, commit_index:1100}, n4:{last_applied:1098, commit_index:1100}}
configs: {svc-auth:{write_quorum:"majority"}, svc-api:{max_connections:200}}
commit_graph: {approved:"C240", deployed:{n1:"C210", n4:"C210", n2:"C300"}, edges:[["C210","C220"],["C220","C230"],["C230","C240"],["C290","C300"]]}"""),
            B(2, """
term: 41
leader_claims: ["n2"]
membership_views: {n1:["n1","n2","n3","n4"], n2:["n1","n2","n3","n4"], n3:["n1","n2","n3","n4"], n4:["n1","n2","n3","n4"]}
services:
  svc-auth:
    - {node:"n3", version:"1.5.3", commit:"C210", checksum:"sha256:A"}
    - {node:"n7", version:"1.6.0", commit:"C240", checksum:"sha256:B"}
  svc-api:
    - {node:"n2", version:"1.6.0", commit:"C300", checksum:"sha256:Z"}
db_schema_version: {n1:16, n2:16, n3:16, n4:16}
ntp_offsets_ms: {n1:46, n2:47, n3:48, n4:49}
replication: {n1:{last_applied:1020, commit_index:1120}, n2:{last_applied:1115, commit_index:1120}, n3:{last_applied:1100, commit_index:1120}, n4:{last_applied:1110, commit_index:1120}}
configs: {svc-auth:{write_quorum:"majority"}, svc-api:{max_connections:200}}
commit_graph: {approved:"C240", deployed:{n3:"C210", n7:"C240", n2:"C300"}, edges:[["C210","C220"],["C220","C230"],["C230","C240"],["C290","C300"]]}"""),
            B(3, """
term: 41
leader_claims: ["n2","n6"]
membership_views: {n1:["n1","n2","n3","n4"], n2:["n1","n2","n3","n4"], n3:["n1","n2","n3","n4"], n6:["n1","n2","n4","n5"]}
services:
  svc-auth:
    - {node:"n3", version:"1.5.3", commit:"C210", checksum:"sha256:A"}
    - {node:"n7", version:"1.6.0", commit:"C240", checksum:"sha256:B"}
  svc-api:
    - {node:"n2", version:"1.6.0", commit:"C300", checksum:"sha256:Z"}
db_schema_version: {n1:16, n2:16, n3:16, n4:16, n5:16}
ntp_offsets_ms: {n1:49, n2:50, n3:45, n4:46, n5:47}
replication: {n1:{last_applied:1040, commit_index:1140}, n2:{last_applied:1135, commit_index:1140}, n3:{last_applied:1120, commit_index:1140}, n4:{last_applied:1130, commit_index:1140}}
configs: {svc-auth:{write_quorum:"majority"}}
commit_graph: {approved:"C240", deployed:{n3:"C210", n7:"C240", n2:"C300"}, edges:[["C210","C220"],["C220","C230"],["C230","C240"],["C290","C300"]]}"""),
            B(4, """
term: 41
leader_claims: ["n2","n6"]
membership_views: {n1:["n1","n2","n3","n4"], n2:["n1","n2","n3","n4"], n3:["n1","n2","n3","n4"], n6:["n1","n2","n4","n5"]}
services:
  svc-auth:
    - {node:"n3", version:"1.5.3", commit:"C210", checksum:"sha256:A"}
    - {node:"n7", version:"1.6.0", commit:"C240", checksum:"sha256:B"}
  svc-api:
    - {node:"n2", version:"1.6.0", commit:"C300", checksum:"sha256:Z"}
db_schema_version: {n1:16, n2:16, n3:16, n4:16, n5:16}
ntp_offsets_ms: {n1:44, n2:45, n3:46, n4:47, n5:43}
replication: {n1:{last_applied:1060, commit_index:1160}, n2:{last_applied:1150, commit_index:1160}, n3:{last_applied:1135, commit_index:1160}, n4:{last_applied:1145, commit_index:1160}}
configs: {svc-auth:{write_quorum:"majority"}}
commit_graph: {approved:"C240", deployed:{n3:"C210", n7:"C240", n2:"C300"}, edges:[["C210","C220"],["C220","C230"],["C230","C240"],["C290","C300"]]}"""),
        ],
    ))

    # Scenario 2: Schema + commit mismatch (4 turns)
    scenarios.append(S(
        name="Scenario 2: Schema below range and divergent commit",
        num_turns=4,
        positives={"schema_incompatibility", "commit_mismatch"},
        raw_states=[
            B(1, """
term: 42
leader_claims: ["n1"]
membership_views: {n1:["n1","n2","n3","n4","n5"], n2:["n1","n2","n3","n4","n5"], n3:["n1","n2","n3","n4","n5"]}
services: {svc-auth:[{node:"n2", version:"1.6.0", commit:"C300", checksum:"sha256:A"}], svc-api:[{node:"n4", version:"1.4.2", commit:"C260", checksum:"sha256:Y"}]}
db_schema_version: {n1:16, n2:16, n3:16, n4:16, n5:16}
ntp_offsets_ms: {n1:20, n2:18, n3:22, n4:21, n5:19}
replication: {n2:{last_applied:2000, commit_index:2008}}
configs: {svc-auth:{write_quorum:"majority"}}
commit_graph: {approved:"C340", deployed:{n2:"C300", n4:"C260"}, edges:[["C300","C320"],["C320","C330"],["C330","C340"],["C240","C250"],["C250","C260"]]}"""),
            B(2, """
term: 42
leader_claims: ["n1"]
membership_views: {n1:["n1","n2","n3","n4","n5"], n2:["n1","n2","n3","n4","n5"]}
services: {svc-auth:[{node:"n2", version:"1.6.0", commit:"C300", checksum:"sha256:A"}], svc-api:[{node:"n4", version:"1.4.2", commit:"C260", checksum:"sha256:Y"}]}
db_schema_version: {n1:16, n2:16, n3:16, n4:16, n5:14}
ntp_offsets_ms: {n1:22, n2:19, n3:21, n4:23, n5:20}
replication: {n2:{last_applied:2002, commit_index:2010}}
configs: {svc-auth:{write_quorum:"majority"}}
commit_graph: {approved:"C350", deployed:{n2:"C300", n4:"C260"}, edges:[["C300","C315"],["C315","C325"],["C325","C335"],["C335","C345"],["C200","C210"],["C210","C220"],["C220","C230"],["C230","C240"],["C240","C250"],["C250","C260"]]}"""),
            B(3, """
term: 42
leader_claims: ["n1"]
membership_views: {n1:["n1","n2","n3","n4","n5"]}
services: {svc-auth:[{node:"n2", version:"1.6.0", commit:"C300", checksum:"sha256:A"}], svc-api:[{node:"n4", version:"1.4.2", commit:"C260", checksum:"sha256:Y"}]}
db_schema_version: {n1:16, n2:16, n3:16, n4:16, n5:14}
ntp_offsets_ms: {n1:19, n2:20, n3:22, n4:21, n5:18}
replication: {n2:{last_applied:2005, commit_index:2015}}
configs: {svc-auth:{write_quorum:"majority"}}
commit_graph: {approved:"C360", deployed:{n2:"C300", n4:"C260"}, edges:[["C200","C210"],["C210","C220"],["C220","C230"],["C230","C240"],["C240","C250"],["C250","C260"],["C260","C270"],["C270","C280"],["C280","C290"],["C290","C300"],["C300","C310"],["C310","C320"],["C320","C330"],["C330","C340"],["C340","C350"],["C350","C360"]]}"""),
            B(4, """
term: 42
leader_claims: ["n1"]
membership_views: {n1:["n1","n2","n3","n4","n5"]}
services: {svc-auth:[{node:"n2", version:"1.6.0", commit:"C300"}], svc-api:[{node:"n4", version:"1.4.2", commit:"C260"}]}
db_schema_version: {n1:16, n2:16, n3:16, n4:16, n5:14}
ntp_offsets_ms: {n1:21, n2:22, n3:23, n4:20, n5:19}
replication: {n2:{last_applied:2010, commit_index:2020}}
configs: {svc-auth:{write_quorum:"majority"}}
commit_graph: {approved:"C360", deployed:{n2:"C300", n4:"C260"}, edges:[["C200","C210"],["C210","C220"],["C220","C230"],["C230","C240"],["C240","C250"],["C250","C260"],["C260","C270"],["C270","C280"],["C280","C290"],["C290","C300"],["C300","C310"],["C310","C320"],["C320","C330"],["C330","C340"],["C340","C350"],["C350","C360"]]}"""),
        ],
    ))

    # Scenario 3: Time skew + replica lag (4 turns)
    scenarios.append(S(
        name="Scenario 3: Time skew and replica lag progression",
        num_turns=4,
        positives={"time_skew", "replica_lag_exceeded"},
        raw_states=[
            B(1, """
term: 40
leader_claims: ["n1"]
membership_views: {n1:["n1","n2","n3"], n2:["n1","n2","n3"], n3:["n1","n2","n3"]}
services: {svc-auth:[{node:"n1", version:"1.5.3", commit:"C100", checksum:"sha256:A"}], svc-api:[{node:"n2", version:"1.6.0", commit:"C200", checksum:"sha256:B"}]}
db_schema_version: {n1:16, n2:16, n3:16}
ntp_offsets_ms: {n1:45, n2:44, n3:47}
replication: {n1:{last_applied:590, commit_index:590}, n2:{last_applied:500, commit_index:590}, n3:{last_applied:480, commit_index:590}}
configs: {svc-auth:{write_quorum:"majority"}}
commit_graph: {approved:"C220", deployed:{n1:"C100", n2:"C200"}, edges:[["C190","C200"],["C200","C210"],["C210","C220"]]}"""),
            B(2, """
term: 40
leader_claims: ["n1"]
membership_views: {n1:["n1","n2","n3"], n2:["n1","n2","n3"]}
services: {svc-auth:[{node:"n1", version:"1.5.3", commit:"C100", checksum:"sha256:A"}], svc-api:[{node:"n2", version:"1.6.0", commit:"C200", checksum:"sha256:B"}]}
db_schema_version: {n1:16, n2:16, n3:16}
ntp_offsets_ms: {n1:51, n2:49, n3:49}
replication: {n1:{last_applied:600, commit_index:600}, n2:{last_applied:498, commit_index:600}, n3:{last_applied:498, commit_index:600}}
configs: {svc-auth:{write_quorum:"majority"}}
commit_graph: {approved:"C220", deployed:{n1:"C100", n2:"C200"}, edges:[["C190","C200"],["C200","C210"],["C210","C220"]]}"""),
            B(3, """
term: 40
leader_claims: ["n1"]
membership_views: {n1:["n1","n2","n3"]}
services: {svc-auth:[{node:"n1", version:"1.5.3", commit:"C100", checksum:"sha256:A"}], svc-api:[{node:"n2", version:"1.6.0", commit:"C200", checksum:"sha256:B"}]}
db_schema_version: {n1:16, n2:16, n3:16}
ntp_offsets_ms: {n1:72, n2:52, n3:61}
replication: {n1:{last_applied:610, commit_index:610}, n2:{last_applied:480, commit_index:610}, n3:{last_applied:475, commit_index:610}}
configs: {svc-auth:{write_quorum:"majority"}}
commit_graph: {approved:"C220", deployed:{n1:"C100", n2:"C200"}, edges:[["C190","C200"],["C200","C210"],["C210","C220"]]}"""),
            B(4, """
term: 40
leader_claims: ["n1"]
membership_views: {n1:["n1","n2","n3"]}
services: {svc-auth:[{node:"n1", version:"1.5.3", commit:"C100", checksum:"sha256:A"}], svc-api:[{node:"n2", version:"1.6.0", commit:"C200", checksum:"sha256:B"}]}
db_schema_version: {n1:16, n2:16, n3:16}
ntp_offsets_ms: {n1:68, n2:54, n3:56}
replication: {n1:{last_applied:620, commit_index:620}, n2:{last_applied:500, commit_index:620}, n3:{last_applied:500, commit_index:620}}
configs: {svc-auth:{write_quorum:"majority"}}
commit_graph: {approved:"C220", deployed:{n1:"C100", n2:"C200"}, edges:[["C190","C200"],["C200","C210"],["C210","C220"]]}"""),
        ],
    ))

    # Scenario 4: Checksums + config drift (4 turns)
    scenarios.append(S(
        name="Scenario 4: Checksums diverge then config drift appears",
        num_turns=4,
        positives={"checksum_mismatch", "config_drift"},
        raw_states=[
            B(1, """
term: 44
leader_claims: ["n1"]
membership_views: {n1:["n1","n2","n3","n4"], n2:["n1","n2","n3","n4"], n3:["n1","n2","n3","n4"], n4:["n1","n2","n3","n4"]}
services: {svc-api:[{node:"n2", version:"1.6.1", commit:"C410", checksum:"sha256:X"},{node:"n7", version:"1.6.1", commit:"C410", checksum:"sha256:X"}]}
db_schema_version: {n1:16, n2:16, n3:16, n4:16, n7:16}
ntp_offsets_ms: {n1:20, n2:21, n3:19, n4:18, n7:22}
replication: {n2:{last_applied:3000, commit_index:3005}, n7:{last_applied:3004, commit_index:3005}}
configs: {svc-api:{write_quorum:"majority", max_connections:200}}
commit_graph: {approved:"C420", deployed:{n2:"C410", n7:"C410"}, edges:[["C400","C410"],["C410","C420"]]}"""),
            B(2, """
term: 44
leader_claims: ["n1"]
membership_views: {n1:["n1","n2","n3","n4"], n2:["n1","n2","n3","n4"]}
services: {svc-api:[{node:"n7", version:"1.6.1", commit:"C410", checksum:"sha256:Y"},{node:"n2", version:"1.6.1", commit:"C410", checksum:"sha256:X"}]}
db_schema_version: {n1:16, n2:16, n3:16, n4:16, n7:16}
ntp_offsets_ms: {n1:21, n2:22, n3:20, n4:19, n7:23}
replication: {n2:{last_applied:3002, commit_index:3010}, n7:{last_applied:3008, commit_index:3010}}
configs: {svc-api:{write_quorum:"majority", max_connections:200}}
commit_graph: {approved:"C420", deployed:{n2:"C410", n7:"C410"}, edges:[["C400","C410"],["C410","C420"]]}"""),
            B(3, """
term: 44
leader_claims: ["n1"]
membership_views: {n1:["n1","n2","n3","n4"], n2:["n1","n2","n3","n4"]}
services: {svc-api:[{node:"n2", version:"1.6.1", commit:"C410", checksum:"sha256:X"},{node:"n7", version:"1.6.1", commit:"C410", checksum:"sha256:Y"}]}
db_schema_version: {n1:16, n2:16, n3:16, n4:16, n7:16}
ntp_offsets_ms: {n1:22, n2:20, n3:19, n4:21, n7:22}
replication: {n2:{last_applied:3005, commit_index:3015}, n7:{last_applied:3010, commit_index:3015}}
configs: {svc-api:{write_quorum:"majority", max_connections:200}, svc-api@node6:{max_connections:150}}
commit_graph: {approved:"C420", deployed:{n2:"C410", n7:"C410"}, edges:[["C400","C410"],["C410","C420"]]}"""),
            B(4, """
term: 44
leader_claims: ["n1"]
membership_views: {n1:["n1","n2","n3","n4"], n2:["n1","n2","n3","n4"]}
services: {svc-api:[{node:"n2", version:"1.6.1", commit:"C410", checksum:"sha256:X"},{node:"n7", version:"1.6.1", commit:"C410", checksum:"sha256:Y"}]}
db_schema_version: {n1:16, n2:16, n3:16, n4:16, n7:16}
ntp_offsets_ms: {n1:23, n2:21, n3:20, n4:22, n7:24}
replication: {n2:{last_applied:3010, commit_index:3020}, n7:{last_applied:3015, commit_index:3020}}
configs: {svc-api:{write_quorum:"majority", max_connections:200}, svc-api@node6:{max_connections:150}}
commit_graph: {approved:"C420", deployed:{n2:"C410", n7:"C410"}, edges:[["C400","C410"],["C410","C420"]]}"""),
        ],
    ))

    # Scenario 5: Quorum mismatch + split brain (4 turns)
    scenarios.append(S(
        name="Scenario 5: Quorum mismatch then split-brain",
        num_turns=4,
        positives={"quorum_mismatch", "split_brain"},
        raw_states=[
            B(1, """
term: 43
leader_claims: ["n1"]
membership_views: {n1:["n1","n2","n3"], n2:["n1","n2","n3"], n3:["n1","n2","n3"]}
services: {}
db_schema_version: {n1:16,n2:16,n3:16}
ntp_offsets_ms: {n1:20,n2:22,n3:19}
replication: {n2:{last_applied:700, commit_index:710}, n3:{last_applied:705, commit_index:710}}
configs: {}
commit_graph: {approved:"C500", deployed:{}, edges:[["C480","C490"],["C490","C500"]]}"""),
            B(2, """
term: 43
leader_claims: ["n1"]
membership_views: {n1:["n1","n2","n3"], n2:["n1","n2","n4"], n3:["n1","n2","n3"], n4:["n1","n2","n4"]}
services: {}
db_schema_version: {n1:16,n2:16,n3:16,n4:16}
ntp_offsets_ms: {n1:21,n2:21,n3:22,n4:23}
replication: {n2:{last_applied:705, commit_index:715}, n3:{last_applied:708, commit_index:715}}
configs: {}
commit_graph: {approved:"C500", deployed:{}, edges:[["C480","C490"],["C490","C500"]]}"""),
            B(3, """
term: 43
leader_claims: ["n1","n4"]
membership_views: {n1:["n1","n2","n3"], n2:["n1","n2","n4"], n3:["n1","n2","n3"], n4:["n1","n2","n4"]}
services: {}
db_schema_version: {n1:16,n2:16,n3:16,n4:16}
ntp_offsets_ms: {n1:22,n2:20,n3:21,n4:22}
replication: {n2:{last_applied:708, commit_index:720}, n3:{last_applied:710, commit_index:720}}
configs: {}
commit_graph: {approved:"C500", deployed:{}, edges:[["C480","C490"],["C490","C500"]]}"""),
            B(4, """
term: 43
leader_claims: ["n1","n4"]
membership_views: {n1:["n1","n2","n3"], n2:["n1","n2","n4"], n3:["n1","n2","n3"], n4:["n1","n2","n4"]}
services: {}
db_schema_version: {n1:16,n2:16,n3:16,n4:16}
ntp_offsets_ms: {n1:21,n2:21,n3:21,n4:21}
replication: {n2:{last_applied:712, commit_index:725}, n3:{last_applied:713, commit_index:725}}
configs: {}
commit_graph: {approved:"C500", deployed:{}, edges:[["C480","C490"],["C490","C500"]]}"""),
        ],
    ))

    # Scenario 6: Version + schema incompatibility (4 turns)
    scenarios.append(S(
        name="Scenario 6: Minor version skew and schema below range",
        num_turns=4,
        positives={"version_skew", "schema_incompatibility"},
        raw_states=[
            B(1, """
term: 45
leader_claims: ["n1"]
membership_views: {n1:["n1","n2","n7"], n2:["n1","n2","n7"], n7:["n1","n2","n7"]}
services: {svc-auth:[{node:"n1", version:"1.5.3"},{node:"n2", version:"1.5.3"},{node:"n7", version:"1.5.3"}]}
db_schema_version: {n1:16, n2:16, n7:16}
ntp_offsets_ms: {n1:20, n2:20, n7:20}
replication: {n1:{last_applied:1200, commit_index:1210}, n2:{last_applied:1205, commit_index:1210}, n7:{last_applied:1206, commit_index:1210}}
configs: {svc-auth:{write_quorum:"majority"}}
commit_graph: {approved:"C150", deployed:{n1:"C140"}, edges:[["C130","C140"],["C140","C150"]]}"""),
            B(2, """
term: 45
leader_claims: ["n1"]
membership_views: {n1:["n1","n2","n7"], n2:["n1","n2","n7"], n7:["n1","n2","n7"]}
services: {svc-auth:[{node:"n1", version:"1.6.0"},{node:"n2", version:"1.5.3"},{node:"n7", version:"1.5.3"}]}
db_schema_version: {n1:16, n2:16, n7:16}
ntp_offsets_ms: {n1:21, n2:21, n7:21}
replication: {n1:{last_applied:1210, commit_index:1220}, n2:{last_applied:1215, commit_index:1220}, n7:{last_applied:1216, commit_index:1220}}
configs: {svc-auth:{write_quorum:"majority"}}
commit_graph: {approved:"C150", deployed:{n1:"C140"}, edges:[["C130","C140"],["C140","C150"]]}"""),
            B(3, """
term: 45
leader_claims: ["n1"]
membership_views: {n1:["n1","n2","n7"], n2:["n1","n2","n7"], n7:["n1","n2","n7"]}
services: {svc-auth:[{node:"n1", version:"1.6.0"},{node:"n2", version:"1.5.3"},{node:"n7", version:"1.5.3"}]}
db_schema_version: {n1:16, n2:16, n7:14}
ntp_offsets_ms: {n1:22, n2:20, n7:22}
replication: {n1:{last_applied:1220, commit_index:1230}, n2:{last_applied:1225, commit_index:1230}, n7:{last_applied:1200, commit_index:1230}}
configs: {svc-auth:{write_quorum:"majority"}}
commit_graph: {approved:"C150", deployed:{n1:"C140"}, edges:[["C130","C140"],["C140","C150"]]}"""),
            B(4, """
term: 45
leader_claims: ["n1"]
membership_views: {n1:["n1","n2","n7"], n2:["n1","n2","n7"], n7:["n1","n2","n7"]}
services: {svc-auth:[{node:"n1", version:"1.6.0"},{node:"n2", version:"1.5.3"},{node:"n7", version:"1.5.3"}]}
db_schema_version: {n1:16, n2:16, n7:14}
ntp_offsets_ms: {n1:21, n2:21, n7:21}
replication: {n1:{last_applied:1230, commit_index:1240}, n2:{last_applied:1235, commit_index:1240}, n7:{last_applied:1210, commit_index:1240}}
configs: {svc-auth:{write_quorum:"majority"}}
commit_graph: {approved:"C150", deployed:{n1:"C140"}, edges:[["C130","C140"],["C140","C150"]]}"""),
        ],
    ))

    # Scenario 7: Commit mismatch + time skew (4 turns)
    scenarios.append(S(
        name="Scenario 7: Deployed commit not ancestor; time drift on subset",
        num_turns=4,
        positives={"commit_mismatch", "time_skew"},
        raw_states=[
            B(1, """
term: 46
leader_claims: ["n2"]
membership_views: {n2:["n1","n2","n3","n4"], n4:["n1","n2","n3","n4"]}
services: {}
db_schema_version: {n1:16, n2:16, n3:16, n4:16}
ntp_offsets_ms: {n4:45, n8:47}
replication: {}
configs: {}
commit_graph: {approved:"C300", deployed:{n2:"C270"}, edges:[["C200","C210"],["C210","C220"],["C220","C230"],["C280","C290"],["C290","C300"]]}"""),
            B(2, """
term: 46
leader_claims: ["n2"]
membership_views: {n2:["n1","n2","n3","n4"], n4:["n1","n2","n3","n4"]}
services: {}
db_schema_version: {n1:16, n2:16, n3:16, n4:16}
ntp_offsets_ms: {n4:52, n8:49}
replication: {}
configs: {}
commit_graph: {approved:"C300", deployed:{n2:"C270"}, edges:[["C200","C210"],["C210","C220"],["C220","C230"],["C280","C290"],["C290","C300"]]}"""),
            B(3, """
term: 46
leader_claims: ["n2"]
membership_views: {n2:["n1","n2","n3","n4"], n4:["n1","n2","n3","n4"]}
services: {}
db_schema_version: {n1:16, n2:16, n3:16, n4:16}
ntp_offsets_ms: {n4:64, n8:58}
replication: {}
configs: {}
commit_graph: {approved:"C300", deployed:{n2:"C270"}, edges:[["C200","C210"],["C210","C220"],["C220","C230"],["C280","C290"],["C290","C300"]]}"""),
            B(4, """
final_snapshot: true
term: 46
leader_claims: ["n2"]
membership_views: {n2:["n1","n2","n3","n4"], n4:["n1","n2","n3","n4"]}
services: {}
db_schema_version: {n1:16, n2:16, n3:16, n4:16}
ntp_offsets_ms: {n4:61, n8:54}
replication: {}
configs: {}
commit_graph: {approved:"C300", deployed:{n2:"C270"}, edges:[["C200","C210"],["C210","C220"],["C220","C230"],["C280","C290"],["C290","C300"]]}"""),
        ],
    ))

    # Scenario 8: Replica lag + quorum mismatch (4 turns)
    scenarios.append(S(
        name="Scenario 8: Sustained lag and inconsistent quorum",
        num_turns=4,
        positives={"replica_lag_exceeded", "quorum_mismatch"},
        raw_states=[
            B(1, """
term: 47
leader_claims: ["n1"]
membership_views: {n1:["n1","n2","n3","n4"], n2:["n1","n2","n3","n4"], n3:["n1","n2","n3","n4"], n4:["n1","n2","n3","n4"]}
services: {}
db_schema_version: {n1:16,n2:16,n3:16,n4:16}
ntp_offsets_ms: {n1:20,n2:19,n3:21,n4:20}
replication: {n3:{last_applied:950, commit_index:1040}}
configs: {}
commit_graph: {approved:"C500", deployed:{}, edges:[["C480","C490"],["C490","C500"]]}"""),
            B(2, """
term: 47
leader_claims: ["n1"]
membership_views: {n1:["n1","n2","n3","n4"], n2:["n1","n2","n4","n5"], n4:["n1","n2","n4","n5"]}
services: {}
db_schema_version: {n1:16,n2:16,n3:16,n4:16,n5:16}
ntp_offsets_ms: {n1:22,n2:21,n3:22,n4:23,n5:22}
replication: {n3:{last_applied:930, commit_index:1045}}
configs: {}
commit_graph: {approved:"C500", deployed:{}, edges:[["C480","C490"],["C490","C500"]]}"""),
            B(3, """
term: 47
leader_claims: ["n1"]
membership_views: {n1:["n1","n2","n3","n4"], n2:["n1","n2","n4","n5"]}
services: {}
db_schema_version: {n1:16,n2:16,n3:16,n4:16,n5:16}
ntp_offsets_ms: {n1:21,n2:21,n3:22,n4:21,n5:20}
replication: {n3:{last_applied:900, commit_index:1048}, n5:{last_applied:880, commit_index:1048}}
configs: {}
commit_graph: {approved:"C500", deployed:{}, edges:[["C480","C490"],["C490","C500"]]}"""),
            B(4, """
final_snapshot: true
term: 47
leader_claims: ["n1"]
membership_views: {n1:["n1","n2","n3","n4"], n2:["n1","n2","n4","n5"], n5:["n1","n2","n5"]}
services: {}
db_schema_version: {n1:16,n2:16,n3:16,n4:16,n5:16}
ntp_offsets_ms: {n1:20,n2:20,n3:21,n4:20,n5:19}
replication: {n3:{last_applied:905, commit_index:1055}}
configs: {}
commit_graph: {approved:"C500", deployed:{}, edges:[["C480","C490"],["C490","C500"]]}"""),
        ],
    ))

    return scenarios


@benchmark(
    name="event_conflict_detection",
    agent_type=AgentType.MODEL_BASED_REFLEX,
    description="Multi-turn distributed-systems incident tagging",
)
class EventConflictDetectionBenchmark(BaseBenchmark):
    """Model-based reflex benchmark for distributed-systems conflict tagging.

    Metric definitions:
    - final_f1/final_precision/final_recall: Micro-averaged over the closed tag set
      using only the final-turn predicted tags.
    - per_turn_precision/recall/f1: Micro metrics computed at each non-final turn
      from the hypothesis tags for that turn.
    - flip_flops: Count of turn-to-turn changes in the predicted tag set, i.e.,
      number of indices i where predictions at turn i differ from turn i+1.
      This is symmetric and counts both improvements and regressions.
    - monotonic_recall: Fraction of true tags that, once predicted prior to the final
      turn, remain predicted on every subsequent turn including the final. Tags first
      seen only at the final turn are not counted as monotonic.
    - spurious_persistence: Among false positives present in the final turn, the
      fraction that were introduced before the final turn and remained present on
      every subsequent turn (final-only FPs are excluded).
    - execution_time: Sum of model API call durations across turns; intentional
      15s sleeps between turns are excluded.
    - final_parse_failed: True if the final-turn structured answer could not be
      parsed after all fallbacks; in that case, final score is set to 0.
    """

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)

    def get_tasks(self) -> List[Task]:
        tasks: List[Task] = []
        for i, spec in enumerate(_make_scenarios(), start=1):
            prompt = self._build_turn_prompt(spec, turn_index=1, prior_notes=[], prior_hypotheses_by_turn=[], is_final=(spec.num_turns == 1))
            task = Task(
                task_id=f"event_conflict_{i}",
                name=f"Event Conflict Detection: {spec.name}",
                description="Multi-turn distributed-systems incident tagging (final micro-F1)",
                prompt=prompt,
                expected_output=None,
                evaluation_criteria={"scoring": "final_micro_f1"},
                metadata={
                    "allowed_tags": ALLOWED_TAGS,
                    "num_turns": spec.num_turns,
                    "ground_truth_tags": sorted(list(spec.positives)),
                    "raw_states": spec.raw_states,
                },
            )
            tasks.append(task)
        return tasks

    def _build_turn_prompt(
        self,
        spec: ScenarioSpec,
        turn_index: int,
        prior_notes: List[str],
        prior_hypotheses_by_turn: List[List[Dict[str, str]]],
        is_final: bool,
    ) -> str:
        header = (
            "Distributed-Systems Incident Tagging (Multi-Turn)\n\n"
            f"Task: Identify all applicable incident tags for this scenario.\n"
            f"{_format_allowed_tags()}\n\n"
            f"{_build_policy_text()}\n"
        )

        history = ""
        if turn_index > 1:
            if prior_notes:
                history += "Notes carried over from prior steps:\n" + "\n".join(f"- {n}" for n in prior_notes) + "\n\n"
            if prior_hypotheses_by_turn:
                history += "Hypotheses by previous turns (may be wrong):\n"
                for idx, hyp in enumerate(prior_hypotheses_by_turn, start=1):
                    if hyp:
                        bits = ", ".join(f"{d.get('tag')}({d.get('confidence')})" for d in hyp)
                        history += f"- Step {idx}: {bits}\n"
                history += "\n"
            # Evidence history
            prev_states = "\n\n".join(spec.raw_states[: turn_index - 1])
            history += f"EVIDENCE HISTORY (Turns 1..{turn_index-1}):\n{prev_states}\n\n"

        current_state = spec.raw_states[turn_index - 1]

        instructions: List[str] = []
        if not is_final:
            instructions.append(
                "First, output your reasoning in natural language. After your reasoning, output your answer on the last line using the preferred format:"
            )
            instructions.append(
                '[ANSWER: {"tags": [{"tag": "<allowed tag>", "confidence": "low|medium|high"}, ...], "notes": "short free-form note (<= 240 chars)"}]'
            )
            instructions.append(
                "Notes are free-form; step labels will be added by the system and all prior notes will be carried to the next turn; if you provide an invalid confidence level, the system will replace it with N/A."
            )
            instructions.append(
                "Example (non-final, two tags + note):\n"
                "Reasoning: Offsets exceed tolerance and deployed commit diverges from approved lineage.\n"
                "[ANSWER: {\"tags\": [{\"tag\": \"time_skew\", \"confidence\": \"high\"}, {\"tag\": \"commit_mismatch\", \"confidence\": \"medium\"}], \"notes\": \"drift >50ms on n1; n2 commit not on approved path\"}]"
            )
            instructions.append(
                "Example (non-final, no tags yet):\n"
                "Reasoning: All values within tolerance so far.\n"
                "[ANSWER: {\"tags\": [], \"notes\": \"no clear incident yet; monitoring thresholds\"}]"
            )
        else:
            instructions.append(
                'FINAL TURN: First, output your reasoning in natural language. Then, on the last line, output your final tag list using the exact format [ANSWER: {"tags": ["..."]}]'
            )
            instructions.append(
                "Example (final):\n"
                "Reasoning: Conflicting leaders in same term; versions inconsistent across replicas.\n"
                "[ANSWER: {\"tags\": [\"split_brain\", \"version_skew\"]}]"
            )
        prompt = (
            f"{header}"
            f"{history}"
            f"CURRENT STATE (Turn {turn_index}/{spec.num_turns}):\n{current_state}\n\n"
            + "\n".join(instructions)
            + "\n"
        )
        return prompt

    async def evaluate_task(self, task: Task, model: BaseModel) -> TaskResult:
        # Accumulate per-call API durations; exclude intentional sleeps
        delays_applied = 0
        total_tokens = 0
        accumulated_call_time = 0.0

        allowed_tags: Set[str] = set(task.metadata.get("allowed_tags", ALLOWED_TAGS))
        truth_tags: Set[str] = set(task.metadata.get("ground_truth_tags", []))
        num_turns: int = int(task.metadata.get("num_turns", 8))
        raw_states: List[str] = list(task.metadata.get("raw_states", []))

        # State across turns
        carried_notes: List[str] = []
        hypotheses_history: List[List[Dict[str, str]]] = []
        predicted_by_turn: List[Set[str]] = []
        per_turn_precisions: List[float] = []
        per_turn_recalls: List[float] = []
        per_turn_f1s: List[float] = []

        try:
            # Recreate a ScenarioSpec for prompt building convenience
            spec = ScenarioSpec(
                name=(task.name or task.task_id),
                num_turns=num_turns,
                positives=truth_tags,
                raw_states=raw_states,
            )

            final_parse_failed = False
            for turn in range(1, num_turns + 1):
                is_final = turn == num_turns
                prompt = self._build_turn_prompt(
                    spec=spec,
                    turn_index=turn,
                    prior_notes=carried_notes,
                    prior_hypotheses_by_turn=hypotheses_history,
                    is_final=is_final,
                )

                response = await model.generate(prompt)
                accumulated_call_time += (response.latency or 0.0)
                if response.completion_tokens:
                    total_tokens += response.completion_tokens

                if not is_final:
                    tags_with_conf, new_notes = self._parse_non_final_response(response.text, allowed_tags)
                    carried_notes.extend(_label_truncate_note(n, step=turn, max_chars=240) for n in new_notes)
                    hypotheses_history.append(tags_with_conf)

                    predicted_tags = {d.get("tag") for d in tags_with_conf if d.get("tag") in allowed_tags}
                    predicted_by_turn.append(predicted_tags)
                    p, r, f1, _, _, _ = _compute_micro_prf(predicted_tags, truth_tags)
                    per_turn_precisions.append(p)
                    per_turn_recalls.append(r)
                    per_turn_f1s.append(f1)

                    if turn < num_turns:
                        wait_seconds = float(self.config.additional_params.get("wait_seconds", 15.0)) if getattr(self, "config", None) else 15.0
                        if wait_seconds > 0:
                            await asyncio.sleep(wait_seconds)
                            delays_applied += 1
                else:
                    final_tags = self._parse_final_response(response.text, allowed_tags)
                    if final_tags is None:
                        logger.warning("event_conflict_detection: final parsing failed; assigning empty prediction (score will be 0)")
                        final_parse_failed = True
                        predicted_by_turn.append(set())
                    else:
                        predicted_by_turn.append(set(final_tags))

            if final_parse_failed:
                final_predicted = set()
                final_p, final_r, final_f1, tp, fp, fn = 0.0, 0.0, 0.0, 0, 0, 0
            else:
                final_predicted = predicted_by_turn[-1] if predicted_by_turn else set()
                final_p, final_r, final_f1, tp, fp, fn = _compute_micro_prf(final_predicted, truth_tags)

            flip_flops = sum(1 for a, b in zip(predicted_by_turn, predicted_by_turn[1:]) if a != b)

            monotonic_recall = 0.0
            if truth_tags:
                stick = 0
                for t in truth_tags:
                    first = None
                    for idx, pred in enumerate(predicted_by_turn[:-1]):
                        if t in pred:
                            first = idx
                            break
                    if first is not None:
                        if all(t in pred for pred in predicted_by_turn[first + 1 :]) and t in final_predicted:
                            stick += 1
                monotonic_recall = stick / len(truth_tags)

            fp_tags = list(final_predicted - truth_tags)
            spurious_persistence = 0.0
            if fp_tags:
                persist = 0
                for t in fp_tags:
                    intro = None
                    for idx, pred in enumerate(predicted_by_turn[:-1]):
                        if t in pred:
                            intro = idx
                            break
                    if intro is not None and all(t in pred for pred in predicted_by_turn[intro + 1 :]):
                        persist += 1
                spurious_persistence = persist / len(fp_tags)

            metrics: Dict[str, Any] = {
                "output_tokens": total_tokens,
                "per_turn_precision": per_turn_precisions,
                "per_turn_recall": per_turn_recalls,
                "per_turn_f1": per_turn_f1s,
                "flip_flops": flip_flops,
                "monotonic_recall": monotonic_recall,
                "spurious_persistence": spurious_persistence,
                "predicted_tags_by_turn": [sorted(list(s)) for s in predicted_by_turn],
                "final_precision": final_p,
                "final_recall": final_r,
                "final_f1": final_f1,
                "ground_truth_tags": sorted(list(truth_tags)),
                "final_predicted_tags": sorted(list(final_predicted)),
                "final_parse_failed": final_parse_failed,
            }

            return TaskResult(
                task_id=task.task_id,
                task_name=task.name,
                agent_type=self.agent_type,
                success=final_f1 > 0.7,
                score=final_f1,
                metrics=metrics,
                model_response=ModelResponse(text=f"Final F1: {final_f1:.3f}", total_tokens=total_tokens),
                execution_time=accumulated_call_time,
                metadata=task.metadata,
            )

        except Exception as e:
            logger.error(f"Error evaluating task {task.task_id}: {e}")
            return TaskResult(
                task_id=task.task_id,
                task_name=task.name,
                agent_type=self.agent_type,
                success=False,
                score=0.0,
                metrics={"output_tokens": total_tokens},
                execution_time=accumulated_call_time,
                error_message=str(e),
                metadata=task.metadata,
            )

    def _parse_non_final_response(
        self, response_text: str, allowed_tags: Set[str]
    ) -> Tuple[List[Dict[str, str]], List[str]]:
        tags_with_conf: List[Dict[str, str]] = []
        notes: List[str] = []
        if not response_text:
            return tags_with_conf, notes

        json_obj = None
        # Preferred: the last line that itself is an [ANSWER: {...}] block
        try:
            lines = [ln for ln in response_text.strip().splitlines() if ln.strip()]
            for ln in reversed(lines):
                m = re.fullmatch(r"\[ANSWER:\s*(\{[\s\S]*?\})\s*\]", ln)
                if m:
                    json_obj = json.loads(m.group(1))
                    break
        except Exception:
            json_obj = None

        # Fallback: last-line raw JSON only (avoid arbitrary {})
        if json_obj is None:
            try:
                lines = [ln for ln in response_text.strip().splitlines() if ln.strip()]
                if lines and lines[-1].startswith("{") and lines[-1].endswith("}"):
                    json_obj = json.loads(lines[-1])
                    logger.info("event_conflict_detection: parsed non-final via last-line JSON fallback")
            except Exception:
                json_obj = None

        # Last resort: scan for any [ANSWER: {...}] block anywhere in text, pick last
        if json_obj is None:
            try:
                answer_blocks = re.findall(r"\[ANSWER:\s*(\{[\s\S]*?\})\s*\]", response_text)
                if answer_blocks:
                    json_obj = json.loads(answer_blocks[-1])
                    logger.info("event_conflict_detection: parsed non-final via in-text ANSWER fallback")
            except Exception:
                json_obj = None

        # Final fallback: any JSON object that looks like an answer payload
        if json_obj is None:
            try:
                candidates = re.findall(r"\{[\s\S]*?\}", response_text)
                for raw in reversed(candidates):
                    try:
                        obj = json.loads(raw)
                        if isinstance(obj, dict) and ("tags" in obj or "notes" in obj):
                            json_obj = obj
                            logger.warning("event_conflict_detection: parsed non-final via loose JSON scan fallback")
                            break
                    except Exception:
                        continue
            except Exception:
                json_obj = None

        if not isinstance(json_obj, dict):
            return tags_with_conf, notes

        raw_tags = json_obj.get("tags", [])
        raw_notes = json_obj.get("notes", "")
        # Build case-insensitive map to canonical tag names
        allowed_map: Dict[str, str] = {t.lower(): t for t in allowed_tags}

        if isinstance(raw_tags, list):
            for item in raw_tags:
                if isinstance(item, dict):
                    tag = item.get("tag")
                    conf = str(item.get("confidence", "N/A")).lower()
                    if isinstance(tag, str):
                        canonical = allowed_map.get(tag.lower())
                        if canonical:
                            if conf not in CONFIDENCE_VALUES:
                                conf = "N/A"
                            tags_with_conf.append({"tag": canonical, "confidence": conf})
                elif isinstance(item, str):
                    canonical = allowed_map.get(item.lower()) if isinstance(item, str) else None
                    if canonical:
                        tags_with_conf.append({"tag": canonical, "confidence": "N/A"})

        if isinstance(raw_notes, str) and raw_notes.strip():
            notes.append(raw_notes.strip())

        return tags_with_conf, notes

    def _parse_final_response(self, response_text: str, allowed_tags: Set[str]) -> Optional[List[str]]:
        if not response_text:
            return None
        # Preferred: the last line that itself is an [ANSWER: {...}] block
        try:
            lines = [ln for ln in response_text.strip().splitlines() if ln.strip()]
            for ln in reversed(lines):
                m = re.fullmatch(r"\[ANSWER:\s*(\{[\s\S]*?\})\s*\]", ln)
                if m:
                    obj = json.loads(m.group(1))
                    return [t for t in obj.get("tags", []) if isinstance(t, str) and t in allowed_tags]
        except Exception:
            pass
        # Fallback: last-line raw JSON only
        try:
            lines = [ln for ln in response_text.strip().splitlines() if ln.strip()]
            if lines and lines[-1].startswith("{") and lines[-1].endswith("}"):
                obj = json.loads(lines[-1])
                logger.info("event_conflict_detection: parsed final via last-line JSON fallback")
                return [t for t in obj.get("tags", []) if isinstance(t, str) and t in allowed_tags]
        except Exception:
            pass
        # Next fallback: any in-text [ANSWER: {...}] block
        try:
            answer_blocks = re.findall(r"\[ANSWER:\s*(\{[\s\S]*?\})\s*\]", response_text)
            if answer_blocks:
                obj = json.loads(answer_blocks[-1])
                return [t for t in obj.get("tags", []) if isinstance(t, str) and t in allowed_tags]
        except Exception:
            pass
        # Final resort: scan any JSON object bottom-to-top and pick the last usable payload
        try:
            candidates = re.findall(r"\{[\s\S]*?\}", response_text)
            for raw in reversed(candidates):
                try:
                    obj = json.loads(raw)
                    if isinstance(obj, dict) and ("tags" in obj):
                        return [t for t in obj.get("tags", []) if isinstance(t, str) and t in allowed_tags]
                except Exception:
                    continue
        except Exception:
            pass
        # Failure
        logger.warning("event_conflict_detection: final parsing failed after all fallbacks")
        return None

    def calculate_score(self, task: Task, model_response: ModelResponse) -> float:
        # Final score is computed inside evaluate_task.
        return 0.0


