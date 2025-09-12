"""
Local Web Navigation Goal-Based Benchmark

Multi-turn benchmark where the model plans and issues structured actions to
navigate a deterministic local website toward explicit goal states. The
environment is served by `src/benchmarks/local_web_app.py`.

Key properties:
- Goal-based scoring: score = optimal_steps / agent_steps (clamped to [0,1])
- Execution time: sum of ModelResponse.latency across calls (excludes sleeps/retries)
- Rate limiting: 15s sleep between turns
- Metrics: output_tokens, steps_taken, optimal_steps, invalid_actions, action_trace
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Deque, Set
import os
import json as _json
import requests
from bs4 import BeautifulSoup
from websocket import create_connection
from urllib.parse import urlparse
from collections import deque

from ..benchmark.base import AgentType, BaseBenchmark, BenchmarkConfig, Task, TaskResult
from ..benchmark.registry import benchmark
from ..models.base import BaseModel, ModelResponse
from ..utils.logging import get_logger

from .local_web_app import ensure_server_running, reset_server
from .selenium_mcp_server import ensure_mcp_server_running


logger = get_logger(__name__)


@dataclass
class Scenario:
    name: str
    start_url: str
    goal_check: Dict[str, Any]  # {"path": "/success", "selector": "#success"}
    plan_actions: List[Dict[str, Any]]  # abstract minimal plan

    @property
    def optimal_steps(self) -> int:
        return len(self.plan_actions)


def _make_scenarios(base: str) -> List[Scenario]:
    return [
        # Simple navigation with decoys: Home -> Section A -> Details 37 -> Confirm -> Success
        Scenario(
            name="Simple path with decoys",
            start_url=f"{base}/",
            goal_check={"path": "/success/section", "selector": "#success"},
            plan_actions=[
                {"type": "click", "selector": "/section-a"},
                {"type": "click", "selector": "/details/37"},
                {"type": "click", "selector": "/confirm/37"},
                {"type": "click", "selector": "/success/section"},
            ],
        ),
        # Auth gate: Login then proceed to dashboard then success (type+type+submit+click)
        Scenario(
            name="Login gate to success",
            start_url=f"{base}/login",
            goal_check={"path": "/success/login", "selector": "#success"},
            plan_actions=[
                {"type": "type", "selector": "#username", "text": "alice"},
                {"type": "type", "selector": "#password", "text": "wonderland"},
                {"type": "submit", "selector": "form#login-form"},
                {"type": "click", "selector": "/success/login"},
            ],
        ),
        # Sum form: enter numbers summing to 10 and submit (type+type+submit)
        Scenario(
            name="Form sum equals 10",
            start_url=f"{base}/form-sum",
            goal_check={"path": "/success/sum", "selector": "#success"},
            plan_actions=[
                {"type": "type", "selector": "#a", "text": "3"},
                {"type": "type", "selector": "#b", "text": "7"},
                {"type": "submit", "selector": "form#sum-form"},
            ],
        ),
        # Token retrieval then submission (click to form + type token + submit)
        Scenario(
            name="Token retrieval and submission",
            start_url=f"{base}/token",
            goal_check={"path": "/success/token", "selector": "#success"},
            plan_actions=[
                {"type": "click", "selector": "/form-token"},
                {"type": "type", "selector": "#token-input", "text": "TOKEN-XYZ"},
                {"type": "submit", "selector": "form#token-form"},
            ],
        ),
        # Section A direct: start on the section page
        Scenario(
            name="Section page direct to success",
            start_url=f"{base}/section-a",
            goal_check={"path": "/success/section", "selector": "#success"},
            plan_actions=[
                {"type": "click", "selector": "/details/37"},
                {"type": "click", "selector": "/confirm/37"},
                {"type": "click", "selector": "/success/section"},
            ],
        ),
        # Trap then recover: wrong details page, go to decoy, back home, then correct path
        Scenario(
            name="Wrong item trap then recover",
            start_url=f"{base}/details/13",
            goal_check={"path": "/success/section", "selector": "#success"},
            plan_actions=[
                {"type": "click", "selector": "/"},
                {"type": "click", "selector": "/section-a"},
                {"type": "click", "selector": "/details/37"},
                {"type": "click", "selector": "/confirm/37"},
                {"type": "click", "selector": "/success/section"},
            ],
        ),
        # Login path starting from home page
        Scenario(
            name="Login from home then success",
            start_url=f"{base}/",
            goal_check={"path": "/success/login", "selector": "#success"},
            plan_actions=[
                {"type": "click", "selector": "/login"},
                {"type": "type", "selector": "#username", "text": "alice"},
                {"type": "type", "selector": "#password", "text": "wonderland"},
                {"type": "submit", "selector": "form#login-form"},
                {"type": "click", "selector": "/success/login"},
            ],
        ),
        # Token flow starting from home
        Scenario(
            name="Token flow from home",
            start_url=f"{base}/",
            goal_check={"path": "/success/token", "selector": "#success"},
            plan_actions=[
                {"type": "click", "selector": "/token"},
                {"type": "click", "selector": "/form-token"},
                {"type": "type", "selector": "#token-input", "text": "TOKEN-XYZ"},
                {"type": "submit", "selector": "form#token-form"},
            ],
        ),
        # Catalog multi-step checkout
        Scenario(
            name="Catalog checkout flow",
            start_url=f"{base}/",
            goal_check={"path": "/success/catalog", "selector": "#success"},
            plan_actions=[
                {"type": "click", "selector": "/catalog"},
                {"type": "click", "selector": "/catalog/widgets"},
                {"type": "click", "selector": "/catalog/widgets/37"},
                {"type": "click", "selector": "/cart-add/37"},
                {"type": "click", "selector": "/checkout"},
                {"type": "click", "selector": "/address"},
                {"type": "type", "selector": "#street", "text": "1 Infinite Loop"},
                {"type": "submit", "selector": "form#address-form"},
                {"type": "click", "selector": "/success/catalog"},
            ],
        ),
        # OTP login to dashboard to success
        Scenario(
            name="OTP login flow",
            start_url=f"{base}/",
            goal_check={"path": "/success/otp", "selector": "#success"},
            plan_actions=[
                {"type": "click", "selector": "/otp-request"},
                {"type": "click", "selector": "/otp-verify"},
                {"type": "type", "selector": "#otp-input", "text": "9137"},
                {"type": "submit", "selector": "form#otp-form"},
                {"type": "click", "selector": "/success/otp"},
            ],
        ),
        # Preferences consent then admin key then success
        Scenario(
            name="Admin key after consent",
            start_url=f"{base}/",
            goal_check={"path": "/success/admin", "selector": "#success"},
            plan_actions=[
                {"type": "click", "selector": "/preferences"},
                {"type": "submit", "selector": "form#consent-form"},
                {"type": "click", "selector": "/admin-form"},
                {"type": "type", "selector": "#admin-key-input", "text": "KEY-ALPHA"},
                {"type": "submit", "selector": "form#admin-form"},
            ],
        ),
    ]


@dataclass(frozen=True)
class _AbstractState:
    path: str
    logged_in: bool = False
    login_via_otp: bool = False
    cart_has_37: bool = False
    address_ok: bool = False
    consent: bool = False
    otp_ready: bool = False
    typed_username: bool = False
    typed_password: bool = False
    typed_a: bool = False
    typed_b: bool = False
    typed_token: bool = False
    typed_street: bool = False
    typed_otp: bool = False
    typed_admin_key: bool = False


def _normalize_path(url_or_path: str) -> str:
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return urlparse(url_or_path).path or "/"
    return url_or_path if url_or_path.startswith("/") else "/" + url_or_path


def _bfs_successors(state: _AbstractState) -> List[Tuple[Dict[str, Any], _AbstractState]]:
    actions: List[Tuple[Dict[str, Any], _AbstractState]] = []
    p = state.path

    def clear_inputs(s: _AbstractState) -> _AbstractState:
        return _AbstractState(
            path=s.path,
            logged_in=s.logged_in,
            cart_has_37=s.cart_has_37,
            address_ok=s.address_ok,
            consent=s.consent,
            otp_ready=s.otp_ready,
            typed_username=False,
            typed_password=False,
            typed_a=False,
            typed_b=False,
            typed_token=False,
            typed_street=False,
            typed_otp=False,
            typed_admin_key=False,
        )

    # Clickable links depending on page and flags
    clickable: List[str] = []
    if p == "/":
        clickable = [
            "/section-a",
            "/decoy",
            "/login",
            "/form-sum",
            "/token",
            "/catalog",
            "/otp-request",
            "/preferences",
        ]
    elif p == "/offers":
        clickable = ["/"]
    elif p == "/section-a":
        clickable = ["/details/13", "/details/37"]
    elif p.startswith("/details/"):
        item_id = p.split("/")[-1]
        clickable = [f"/confirm/{item_id}" if item_id == "37" else "/offers"]
    elif p.startswith("/confirm/"):
        clickable = ["/success/section"]
    elif p == "/dashboard":
        clickable = ["/success/otp"] if state.login_via_otp else ["/success/login"]
    elif p == "/token":
        clickable = ["/form-token"]
    elif p == "/catalog":
        clickable = ["/catalog/widgets", "/catalog/gadgets"]
    elif p == "/catalog/widgets":
        clickable = ["/catalog/widgets/13", "/catalog/widgets/37"]
    elif p.startswith("/catalog/widgets/"):
        item_id = p.split("/")[-1]
        clickable = [f"/cart-add/{item_id}"]
    elif p == "/cart":
        if state.cart_has_37:
            clickable = ["/checkout"]
        else:
            clickable = ["/offers"]
    elif p == "/checkout":
        clickable = ["/address"]
    elif p == "/review":
        clickable = ["/success/catalog"]
    elif p == "/otp-request":
        clickable = ["/otp-verify"]
    elif p == "/preferences":
        if state.consent:
            clickable = ["/admin-key"]
        else:
            clickable = []
    elif p == "/admin-key":
        clickable = ["/admin-form"]

    # Click transitions
    for href in clickable:
        # Special redirect/flag logic
        if href.startswith("/cart-add/"):
            item_id = href.split("/")[-1]
            new_state = clear_inputs(state)
            new_state = _AbstractState(
                path="/cart",
                logged_in=new_state.logged_in,
                cart_has_37=(new_state.cart_has_37 or item_id == "37"),
                address_ok=new_state.address_ok,
                consent=new_state.consent,
                otp_ready=new_state.otp_ready,
            )
            actions.append(({"type": "click", "selector": href}, new_state))
        else:
            # Some routes redirect depending on flags
            target = href
            if href == "/admin-key" and not state.consent:
                target = "/preferences"
            new_state = clear_inputs(state)
            new_state = _AbstractState(
                path=target,
                logged_in=new_state.logged_in,
                cart_has_37=new_state.cart_has_37,
                address_ok=new_state.address_ok,
                consent=new_state.consent,
                otp_ready=new_state.otp_ready,
            )
            actions.append(({"type": "click", "selector": href}, new_state))

    # Note: navigation actions are disabled for exploitation; successors only expose click/type/submit semantics

    # Forms / type / submit
    if p == "/login":
        if not state.typed_username:
            ns = _AbstractState(path=p, logged_in=state.logged_in, cart_has_37=state.cart_has_37, address_ok=state.address_ok, consent=state.consent, otp_ready=state.otp_ready, typed_username=True, typed_password=state.typed_password)
            actions.append(({"type": "type", "selector": "#username", "text": "alice"}, ns))
        if not state.typed_password:
            ns = _AbstractState(path=p, logged_in=state.logged_in, cart_has_37=state.cart_has_37, address_ok=state.address_ok, consent=state.consent, otp_ready=state.otp_ready, typed_username=state.typed_username, typed_password=True)
            actions.append(({"type": "type", "selector": "#password", "text": "wonderland"}, ns))
        if state.typed_username and state.typed_password:
            ns = _AbstractState(path="/dashboard", logged_in=True, login_via_otp=False, cart_has_37=state.cart_has_37, address_ok=state.address_ok, consent=state.consent, otp_ready=state.otp_ready)
            actions.append(({"type": "submit", "selector": "form#login-form"}, ns))

    if p == "/form-sum":
        if not state.typed_a:
            ns = _AbstractState(path=p, logged_in=state.logged_in, cart_has_37=state.cart_has_37, address_ok=state.address_ok, consent=state.consent, otp_ready=state.otp_ready, typed_a=True, typed_b=state.typed_b)
            actions.append(({"type": "type", "selector": "#a", "text": "3"}, ns))
        if not state.typed_b:
            ns = _AbstractState(path=p, logged_in=state.logged_in, cart_has_37=state.cart_has_37, address_ok=state.address_ok, consent=state.consent, otp_ready=state.otp_ready, typed_a=state.typed_a, typed_b=True)
            actions.append(({"type": "type", "selector": "#b", "text": "7"}, ns))
        if state.typed_a and state.typed_b:
            ns = _AbstractState(path="/success/sum", logged_in=state.logged_in, login_via_otp=state.login_via_otp, cart_has_37=state.cart_has_37, address_ok=state.address_ok, consent=state.consent, otp_ready=state.otp_ready)
            actions.append(({"type": "submit", "selector": "form#sum-form"}, ns))

    if p == "/form-token":
        if not state.typed_token:
            ns = _AbstractState(path=p, logged_in=state.logged_in, cart_has_37=state.cart_has_37, address_ok=state.address_ok, consent=state.consent, otp_ready=state.otp_ready, typed_token=True)
            actions.append(({"type": "type", "selector": "#token-input", "text": "TOKEN-XYZ"}, ns))
        else:
            ns = _AbstractState(path="/success/token", logged_in=state.logged_in, login_via_otp=state.login_via_otp, cart_has_37=state.cart_has_37, address_ok=state.address_ok, consent=state.consent, otp_ready=state.otp_ready)
            actions.append(({"type": "submit", "selector": "form#token-form"}, ns))

    if p == "/address":
        if not state.typed_street:
            ns = _AbstractState(path=p, logged_in=state.logged_in, cart_has_37=state.cart_has_37, address_ok=state.address_ok, consent=state.consent, otp_ready=state.otp_ready, typed_street=True)
            actions.append(({"type": "type", "selector": "#street", "text": "1 Infinite Loop"}, ns))
        else:
            ns = _AbstractState(path="/review", logged_in=state.logged_in, login_via_otp=state.login_via_otp, cart_has_37=state.cart_has_37, address_ok=True, consent=state.consent, otp_ready=state.otp_ready)
            actions.append(({"type": "submit", "selector": "form#address-form"}, ns))

    if p == "/otp-request":
        # Visiting otp-request sets otp_ready
        state = _AbstractState(path=p, logged_in=state.logged_in, cart_has_37=state.cart_has_37, address_ok=state.address_ok, consent=state.consent, otp_ready=True)

    if p == "/otp-verify":
        if not state.typed_otp:
            ns = _AbstractState(path=p, logged_in=state.logged_in, cart_has_37=state.cart_has_37, address_ok=state.address_ok, consent=state.consent, otp_ready=state.otp_ready, typed_otp=True)
            actions.append(({"type": "type", "selector": "#otp-input", "text": "9137"}, ns))
        elif state.otp_ready and state.typed_otp:
            ns = _AbstractState(path="/dashboard", logged_in=True, login_via_otp=True, cart_has_37=state.cart_has_37, address_ok=state.address_ok, consent=state.consent, otp_ready=state.otp_ready)
            actions.append(({"type": "submit", "selector": "form#otp-form"}, ns))

    if p == "/preferences":
        # Can submit consent form anytime to set consent and redirect to admin_key
        ns = _AbstractState(path="/admin-key", logged_in=state.logged_in, cart_has_37=state.cart_has_37, address_ok=state.address_ok, consent=True, otp_ready=state.otp_ready)
        actions.append(({"type": "submit", "selector": "form#consent-form"}, ns))

    if p == "/admin-form":
        if not state.typed_admin_key:
            ns = _AbstractState(path=p, logged_in=state.logged_in, cart_has_37=state.cart_has_37, address_ok=state.address_ok, consent=state.consent, otp_ready=state.otp_ready, typed_admin_key=True)
            actions.append(({"type": "type", "selector": "#admin-key-input", "text": "KEY-ALPHA"}, ns))
        else:
            ns = _AbstractState(path="/success/admin", logged_in=state.logged_in, login_via_otp=state.login_via_otp, cart_has_37=state.cart_has_37, address_ok=state.address_ok, consent=state.consent, otp_ready=state.otp_ready)
            actions.append(({"type": "submit", "selector": "form#admin-form"}, ns))

    return actions


def _compute_k_shortest_plans_via_bfs(start_url: str, goal_path: str = "/success", k: int = 3) -> List[List[Dict[str, Any]]]:
    start_path = _normalize_path(start_url)
    start = _AbstractState(path=start_path)

    # Standard BFS to compute distances and parent lists for shortest paths
    queue: Deque[_AbstractState] = deque([start])
    dist: Dict[_AbstractState, int] = {start: 0}
    parents: Dict[_AbstractState, List[Tuple[_AbstractState, Dict[str, Any]]]] = {start: []}
    goal_states: List[_AbstractState] = []
    min_goal_dist: Optional[int] = None

    while queue:
        cur = queue.popleft()
        cur_dist = dist[cur]
        if min_goal_dist is not None and cur_dist > min_goal_dist:
            continue
        if cur.path == goal_path:
            if min_goal_dist is None:
                min_goal_dist = cur_dist
            goal_states.append(cur)
            continue
        for action, nxt in _bfs_successors(cur):
            nd = cur_dist + 1
            if nxt not in dist:
                dist[nxt] = nd
                parents[nxt] = [(cur, action)]
                queue.append(nxt)
            else:
                # If another shortest parent, record as well (to enumerate alternatives)
                if dist[nxt] == nd:
                    parents.setdefault(nxt, []).append((cur, action))

    if not goal_states:
        return []

    # Enumerate up to k shortest plans by backtracking parents from all goal states
    plans: List[List[Dict[str, Any]]] = []
    seen: Set[str] = set()

    def backtrack(state: _AbstractState, acc: List[Dict[str, Any]]):
        if len(plans) >= k:
            return
        if state == start:
            plan = list(reversed(acc))
            key = _json.dumps(plan, sort_keys=True)
            if key not in seen:
                seen.add(key)
                plans.append(plan)
            return
        for prev, act in parents.get(state, []):
            backtrack(prev, acc + [act])

    for gs in goal_states:
        backtrack(gs, [])
        if len(plans) >= k:
            break
    return plans


def _summarize_dom(html: str, url: str, title: str) -> Dict[str, Any]:
    # In exploitation phase, we may pass in a union of DOMs explored; otherwise viewport-only
    soup = BeautifulSoup(html, "html.parser")
    clickables = []
    for a in soup.find_all("a"):
        # Only include anchors that have an href (selector is href for simplicity)
        text = (a.get_text() or "").strip()
        href = a.get("href")
        if href:
            clickables.append({"selector": href, "text": text, "role": "link", "href": href})

    inputs = []
    for inp in soup.find_all("input"):
        name = inp.get("name")
        placeholder = inp.get("placeholder")
        itype = inp.get("type", "text")
        sel = f"#{inp.get('id')}" if inp.get("id") else (name or "")
        inputs.append({"selector": sel, "name": name, "placeholder": placeholder, "type": itype})

    forms = []
    for form in soup.find_all("form"):
        selector = f"form#{form.get('id')}" if form.get("id") else "form"
        fields = [f.get("name") for f in form.find_all("input") if f.get("name")]
        forms.append({"selector": selector, "fields": fields})

    content_text = " ".join((soup.get_text() or "").split())[:300]
    return {
        "current_url": url,
        "page_title": title,
        "elements": {"clickables": clickables[:25], "inputs": inputs[:25], "forms": forms[:10]},
        "visible_text_preview": content_text,
    }


def _build_exploration_prompt(observation: Dict[str, Any], goal_summary: str) -> str:
    guidance = (
        "Exploration Phase (Unscored).\n"
        "Choose ONE primitive action. Be strategic: try to discover navigation aids such as a Site Guide page (/site-guide),\n"
        "identify key sections, and collect information that will help planning in the next phase. Avoid random wandering.\n"
    )
    memory_block = ""
    if observation.get("exploration_memory") is not None:
        memory_block = f"MEMORY JSON (visited/seen/prior):\n{json.dumps(observation['exploration_memory'], indent=2)}\n\n"
    return (
        guidance
        + f"GOAL (for later exploitation): {goal_summary}\n\n"
        + memory_block
        + f"OBSERVATION JSON (viewport-only):\n{json.dumps({k: v for k, v in observation.items() if k != 'exploration_memory'}, indent=2)}\n\n"
        + "Write a brief Reasoning section first. Then, on the LAST LINE ONLY, output the action block.\n"
        + "Allowed actions (strict JSON objects):\n"
        + "[ACTION: {\"type\":\"click\",\"selector\":\"/path or CSS\"}]\n"
        + "[ACTION: {\"type\":\"type\",\"selector\":\"#input or name\",\"text\":\"...\"}]\n"
        + "[ACTION: {\"type\":\"submit\",\"selector\":\"form selector\"}]\n"
        + "Respond with only one ACTION block as the last line using the exact JSON structure above."
    )


def _build_exploitation_prompt(observation: Dict[str, Any], goal_summary: str) -> str:
    instructions = (
        "Exploitation Phase (Scored).\n"
        "STATE RESET: All cookies and server-side session have been cleared. You are logged out, forms are empty, and any carts/preferences are empty.\n"
        "Start from the clean state in OBSERVATION (current_url is the start URL).\n"
        "Use exploration_memory (visited_urls, seen_pages visible_text, and site_guide_dom if present) only as planning knowledge; do NOT assume interactive state persisted.\n"
        "Write a brief Reasoning section first. Then output a complete PLAN as a JSON array of primitives to reach the GOAL from this clean starting state.\n"
        "Strict JSON per step: {\"type\":\"click|type|submit\", ...}.\n"
    )
    return (
        instructions
        + f"GOAL: {goal_summary}\n\n"
        + f"OBSERVATION JSON (includes exploration_memory):\n{json.dumps(observation, indent=2)}\n\n"
        + "Output one PLAN array on the LAST LINE, like:\n"
        + "[PLAN: [{\"type\":\"click\",\"selector\":\"/section-a\"},{\"type\":\"click\",\"selector\":\"/details/37\"}]]"
    )


class _MCPBrowser:
    """Minimal MCP client wrapper for local Selenium MCP over WebSocket."""

    def __init__(self, base_url: str, endpoint: str):
        self.base = base_url.rstrip("/")
        self.endpoint = endpoint
        # Retry loop to mitigate races with server startup
        self.ws = None
        last_err: Optional[Exception] = None
        for _ in range(50):  # ~5s total with 0.1s sleeps
            try:
                self.ws = create_connection(self.endpoint, timeout=5)
                break
            except Exception as e:
                last_err = e
                time.sleep(0.1)
        if self.ws is None:
            raise RuntimeError(f"Failed to connect to Selenium MCP at {self.endpoint}: {last_err}")
        # Health check
        self._call("browser.getLocation", {})

    def _abs(self, path_or_url: str) -> str:
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            return path_or_url
        if path_or_url.startswith("/"):
            return f"{self.base}{path_or_url}"
        return f"{self.base}/{path_or_url}"

    def _call(self, tool: str, data: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"tool": tool, "input": data}
        self.ws.send(_json.dumps(payload))
        res = self.ws.recv()
        out = _json.loads(res) if isinstance(res, str) else res
        if isinstance(out, dict) and out.get("error"):
            raise RuntimeError(f"{tool} failed: {out['error']}")
        return out if isinstance(out, dict) else {"raw": out}

    def navigate(self, path_or_url: str) -> Tuple[str, str]:
        url = self._abs(path_or_url)
        last_err: Optional[Exception] = None
        for _ in range(10):
            try:
                out = self._call("browser.navigate", {"url": url})
                return out.get("url", url), out.get("title", "")
            except Exception as e:
                last_err = e
                msg = str(e)
                if "ERR_CONNECTION_RESET" in msg or "net::ERR" in msg or "connection reset" in msg.lower():
                    time.sleep(0.1)
                    continue
                raise
        # Exhausted retries
        raise last_err if last_err else RuntimeError("browser.navigate failed after retries")

    def click(self, selector: str) -> Tuple[str, str]:
        out = self._call("browser.click", {"selector": selector, "requireVisible": True, "scrollIntoView": True})
        return out.get("url", ""), out.get("title", "")

    def type(self, selector: str, text: str) -> None:
        self._call("browser.type", {"selector": selector, "text": text, "clear": True})

    def submit(self, selector: str) -> Tuple[str, str]:
        out = self._call("browser.submit", {"selector": selector})
        return out.get("url", ""), out.get("title", "")

    def clear_cookies(self) -> None:
        self._call("browser.clearCookies", {})

    def get_dom_summary(self, viewport_only: bool = True, max_links: int = 25, max_inputs: int = 25, max_forms: int = 10) -> Dict[str, Any]:
        return self._call("browser.getDomSummary", {"viewportOnly": viewport_only, "maxLinks": max_links, "maxInputs": max_inputs, "maxForms": max_forms})

    def quit(self):
        try:
            self.ws.close()
        except Exception:
            pass


def _strip_code_fences(text: str) -> str:
    if not text:
        return text
    # Remove code fences (``` or ~~~ with optional language). Repeat until none remain.
    pattern = re.compile(r"(?s)(```+|~~~+)[^\n]*\n(.*?)\n\\1")
    while True:
        new_text, count = pattern.subn(lambda m: m.group(2), text)
        if count == 0:
            break
        text = new_text
    return text


def _extract_balanced_block(s: str, start_idx: int, open_char: str, close_char: str) -> Optional[str]:
    depth = 0
    in_string = False
    escape = False
    start = -1
    i = start_idx
    while i < len(s):
        ch = s[i]
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == open_char:
                depth += 1
                if depth == 1:
                    start = i
            elif ch == close_char:
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        return s[start : i + 1]
        i += 1
    return None


def _parse_action(text: str) -> Optional[Dict[str, Any]]:
    """Robustly parse a single ACTION block using balanced-brace extraction and validation.
    Strategy:
    1) Find the last [ACTION: marker (case-insensitive), skip whitespace, then extract the balanced {...}
    2) Fallback: find the last balanced {...} object anywhere in the text
    3) Secondary fallback: if a last balanced [...] array is present, parse first element as an action
    4) Validate required fields via _validate_action (normalizes 'value'->'text' for type)
    """
    if not text:
        return None
    # First, try to find a code-fenced JSON block immediately after an ACTION heading (e.g., ACTION:, **ACTION:**)
    try:
        marker_re = re.compile(r"(?im)^[^\n]*action[^\n]*:\s*$", re.IGNORECASE | re.MULTILINE)
        last_marker: Optional[re.Match] = None
        for m in marker_re.finditer(text):
            last_marker = m
        if last_marker is not None:
            after = text[last_marker.end():]
            fence_re = re.compile(r"(?is)\s*(?P<fence>```+|~~~+)[^\n]*\n(?P<body>.*?)(?:\n(?P=fence))")
            fm = fence_re.search(after)
            if fm:
                body = fm.group("body").strip()
                # Try direct object parse first
                try:
                    obj = json.loads(body)
                except json.JSONDecodeError:
                    obj = None
                if isinstance(obj, dict):
                    valid = _validate_action(json.dumps(obj))
                    if valid:
                        return valid
                # If code block is an array, take first element
                if isinstance(obj, list) and obj:
                    first = obj[0]
                    if isinstance(first, dict):
                        valid = _validate_action(json.dumps(first))
                        if valid:
                            return valid
                # As a fallback inside the block, extract last balanced object
                last_open = body.rfind("{")
                while last_open != -1:
                    json_str = _extract_balanced_block(body, last_open, "{", "}")
                    if json_str:
                        valid = _validate_action(json_str)
                        if valid:
                            return valid
                    last_open = body.rfind("{", 0, last_open)
    except Exception:
        pass

    text = _strip_code_fences(text)

    # Primary: handle [ACTION: marker (case-insensitive)
    idx = text.lower().rfind("[action:")
    if idx != -1:
        after = text[idx + len("[ACTION:"):].lstrip()
        brace_idx = after.find("{")
        if brace_idx != -1:
            json_str = _extract_balanced_block(after, brace_idx, "{", "}")
            if json_str:
                action = _validate_action(json_str)
                if action:
                    return action

    # Fallback: scan backward for last balanced {...}
    last_open = text.rfind("{")
    while last_open != -1:
        json_str = _extract_balanced_block(text, last_open, "{", "}")
        if json_str:
            action = _validate_action(json_str)
            if action:
                return action
        last_open = text.rfind("{", 0, last_open)

    # Secondary fallback: if the model returned an array like [{...}], parse first element
    last_array_open = text.rfind("[")
    while last_array_open != -1:
        arr_str = _extract_balanced_block(text, last_array_open, "[", "]")
        if arr_str:
            try:
                arr = json.loads(arr_str)
            except json.JSONDecodeError:
                arr = None
            if isinstance(arr, list) and arr:
                first = arr[0]
                if isinstance(first, dict):
                    try:
                        return _validate_action(json.dumps(first))
                    except Exception:
                        pass
        last_array_open = text.rfind("[", 0, last_array_open)
    return None


def _validate_action(json_str: str) -> Optional[Dict[str, Any]]:
    try:
        action = json.loads(json_str)
    except json.JSONDecodeError:
        return None
    if not isinstance(action, dict):
        return None
    a_type = action.get("type")
    if not isinstance(a_type, str):
        return None
    # Normalize 'value' -> 'text' for type actions
    if a_type == "type" and "text" not in action and "value" in action:
        action["text"] = action["value"]
    if a_type in ("click", "type", "submit"):
        if not action.get("selector"):
            return None
        if a_type == "type":
            txt = action.get("text")
            if not isinstance(txt, str):
                return None
    elif a_type == "navigate":
        # Disallow navigate in current benchmark configuration
        return None
    else:
        return None
    return action


def _parse_plan(text: str) -> Optional[List[Dict[str, Any]]]:
    """Robustly parse a PLAN array using balanced-bracket extraction and validation.
    Strategy:
    1) Find the last [PLAN: marker (case-insensitive), skip whitespace, then extract the balanced [...]
    2) Fallback: find the last balanced [...] array anywhere in the text
    3) Normalize action fields (e.g., 'value' -> 'text' for type actions)
    4) Validate each step structure strictly
    """
    if not text:
        return None
    plan: Optional[List[Dict[str, Any]]] = None

    # 0) Prefer explicit code-fenced block following a PLAN heading (e.g., PLAN:, **PLAN:**)
    try:
        marker_re = re.compile(r"(?im)^[^\n]*plan[^\n]*:\s*$", re.IGNORECASE | re.MULTILINE)
        last_marker: Optional[re.Match] = None
        for m in marker_re.finditer(text):
            last_marker = m
        if last_marker is not None:
            after = text[last_marker.end():]
            fence_re = re.compile(r"(?is)\s*(?P<fence>```+|~~~+)[^\n]*\n(?P<body>.*?)(?:\n(?P=fence))")
            fm = fence_re.search(after)
            if fm:
                body = fm.group("body").strip()
                # Try direct parse as array
                try:
                    candidate = json.loads(body)
                except json.JSONDecodeError:
                    candidate = None
                if isinstance(candidate, list):
                    plan = candidate
                else:
                    # Fallback within the code block: last balanced array
                    last_open = body.rfind("[")
                    while last_open != -1:
                        json_str = _extract_balanced_block(body, last_open, "[", "]")
                        if json_str:
                            try:
                                candidate = json.loads(json_str)
                            except json.JSONDecodeError:
                                candidate = None
                            if isinstance(candidate, list):
                                plan = candidate
                                break
                        last_open = body.rfind("[", 0, last_open)
    except Exception:
        pass

    # 1) Handle inline [PLAN: ...] marker in free text
    if plan is None:
        text_no_fences = _strip_code_fences(text)
        idx = text_no_fences.lower().rfind("[plan:")
        if idx != -1:
            after = text_no_fences[idx + len("[PLAN:"):].lstrip()
            bracket_idx = after.find("[")
            if bracket_idx != -1:
                json_str = _extract_balanced_block(after, bracket_idx, "[", "]")
                if json_str:
                    try:
                        candidate = json.loads(json_str)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error in plan: {e}")
                        candidate = None
                    if isinstance(candidate, list):
                        plan = candidate

    # 2) Final fallback: last balanced array anywhere
    if plan is None:
        text_no_fences = _strip_code_fences(text)
        last_open = text_no_fences.rfind("[")
        while last_open != -1:
            json_str = _extract_balanced_block(text_no_fences, last_open, "[", "]")
            if json_str:
                try:
                    candidate = json.loads(json_str)
                except json.JSONDecodeError:
                    candidate = None
                if isinstance(candidate, list):
                    plan = candidate
                    break
            last_open = text_no_fences.rfind("[", 0, last_open)

    if not isinstance(plan, list):
        return None

    # Normalize: support 'value' -> 'text' for type actions
    for action in plan:
        if isinstance(action, dict) and action.get("type") == "type" and "text" not in action and "value" in action:
            action["text"] = action["value"]

    # Strict validation of actions
    for action in plan:
        if not isinstance(action, dict):
            return None
        a_type = action.get("type")
        if not isinstance(a_type, str):
            return None
        if a_type in ("click", "type", "submit"):
            if not action.get("selector"):
                return None
            if a_type == "type" and ("text" not in action or not isinstance(action.get("text"), str)):
                return None
        elif a_type == "navigate":
            # Disallow navigate in plans
            return None
        else:
            return None

    return plan



@benchmark(
    name="local_web_navigation",
    agent_type=AgentType.GOAL_BASED,
    description="Goal-based navigation of a deterministic local website using structured actions."
)
class LocalWebNavigationBenchmark(BaseBenchmark):
    """Goal-based web navigation on a local, deterministic site."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        # Start local Flask app and Selenium MCP server per configured URL
        ensure_server_running(host="127.0.0.1", port=5005)
        self.base_url = "http://127.0.0.1:5005"

        mcp_env = os.getenv("SELENIUM_MCP_URL", "ws://127.0.0.1:7007")
        parsed = urlparse(mcp_env)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 7007
        if parsed.scheme in ("ws", "wss") and host in ("127.0.0.1", "localhost", "0.0.0.0"):
            ensure_mcp_server_running(host=host, port=port)
        time.sleep(0.2)  # brief wait for MCP to come up

    def get_tasks(self) -> List[Task]:
        tasks: List[Task] = []
        scenarios = _make_scenarios(self.base_url)
        for idx, sc in enumerate(scenarios, start=1):
            goal_summary = f"Navigate until page path is '{sc.goal_check['path']}' and element '{sc.goal_check['selector']}' exists."
            prompt = (
                "Local Web Navigation Task (Meta-Planning).\n"
                "Phase 1 (Exploration): You will have limited steps to explore a partially observable site (viewport-only DOM).\n"
                "Phase 2 (Exploitation): You must output a single PLAN from the current state to the GOAL using [PLAN: [...]] as specified.\n"
                f"Goal: {goal_summary}\n\n"
            )
            # Compute up to 3 alternative shortest plans via abstract BFS to guard against drift
            bfs_plans = _compute_k_shortest_plans_via_bfs(sc.start_url, goal_path=sc.goal_check["path"], k=3)
            # Prefer the handwritten scenario plan first; fallback to BFS candidates
            preferred_plans: List[List[Dict[str, Any]]] = []
            if sc.plan_actions:
                preferred_plans.append(sc.plan_actions)
            preferred_plans.extend(bfs_plans or [])
            # Deduplicate plans
            dedup: List[List[Dict[str, Any]]] = []
            seen_keys: Set[str] = set()
            for pl in preferred_plans:
                try:
                    key = _json.dumps(pl, sort_keys=True)
                except Exception:
                    key = str(pl)
                if key not in seen_keys:
                    seen_keys.add(key)
                    dedup.append(pl)
            optimal_steps = sc.optimal_steps
            tasks.append(
                Task(
                    task_id=f"local_web_nav_{idx}",
                    name=sc.name,
                    description=sc.name,
                    prompt=prompt,
                    evaluation_criteria={"optimal_steps": optimal_steps, "bfs_plans": dedup},
                    metadata={
                        "start_url": sc.start_url,
                        "goal_check": sc.goal_check,
                        "max_steps": 12,
                        "reference_plan": sc.plan_actions,
                        "exploration_budget": 8,
                    },
                )
            )
        return tasks

    async def evaluate_task(self, task: Task, model: BaseModel) -> TaskResult:
        # Preflight: try executing bfs_plan once via MCP to ensure selectors are valid.
        # This is non-scored and runs with a clean session. If it fails, we can fall back to the
        # hand-authored plan or proceed without verification.
        try:
            bfs_plans: List[List[Dict[str, Any]]] = task.evaluation_criteria.get("bfs_plans") or []
            if bfs_plans:
                mcp_url_pf = os.getenv("SELENIUM_MCP_URL", "ws://127.0.0.1:7007")
                for candidate in bfs_plans:
                    pf_browser: Any = _MCPBrowser(self.base_url, mcp_url_pf)
                    ok = True
                    try:
                        pf_browser.clear_cookies()
                        # Also clear local/session storage to remove residual state
                        try:
                            pf_browser._call("browser.clearStorage", {})
                        except Exception:
                            pass
                        pf_browser.navigate(task.metadata["start_url"])
                        for act in candidate:
                            at = act.get("type")
                            if at == "click" and act.get("selector"):
                                pf_browser.click(act["selector"])
                            elif at == "type" and act.get("selector") and "text" in act:
                                pf_browser.type(act["selector"], act["text"])
                            elif at == "submit" and act.get("selector"):
                                pf_browser.submit(act["selector"])
                    except Exception:
                        ok = False
                    finally:
                        try:
                            pf_browser.clear_cookies()
                        except Exception:
                            pass
                        pf_browser.quit()
                    if ok:
                        # Replace primary plan and optimal_steps with the verified candidate
                        task.evaluation_criteria["bfs_plans"] = [candidate]
                        task.evaluation_criteria["optimal_steps"] = len(candidate)
                        break
        except Exception:
            pass

        # Enforce MCP-only; new driver per task to avoid cross-task state bleed
        mcp_url = os.getenv("SELENIUM_MCP_URL", "ws://127.0.0.1:7007")
        browser: Any = _MCPBrowser(self.base_url, mcp_url)
        # Hard reset driver between tasks
        try:
            browser._call("browser.hardReset", {})
        except Exception:
            pass
        # Navigate to start
        url, title = browser.navigate(task.metadata["start_url"])  # initial observation
        dom0 = browser.get_dom_summary(viewport_only=True)
        observation = {
            "current_url": url,
            "page_title": title,
            "elements": dom0.get("elements", {}),
            "visible_text_preview": "",
        }
        goal = task.metadata["goal_check"]
        max_steps: int = int(task.metadata.get("max_steps", 12))
        exploration_budget: int = int(task.metadata.get("exploration_budget", 0))

        steps_taken = 0
        invalid_actions = 0
        total_tokens = 0
        accumulated_call_time = 0.0
        action_trace: List[Dict[str, Any]] = []

        def goal_reached(u: str) -> bool:
            try:
                cond_path = goal["path"]
                if not u.endswith(cond_path):
                    return False
                sel = goal.get("selector")
                if not sel:
                    return True
                try:
                    return bool(browser._call("browser.querySelectorExists", {"selector": sel}).get("exists", False))
                except Exception:
                    return False
            except Exception:
                return False

        # Helper: wait for navigation to complete if URL changes after an action
        def _wait_until_url_changes(prev_url: str, timeout: float = 2.0) -> Tuple[str, str]:
            t0 = time.time()
            current_url = prev_url
            current_title = ""
            while time.time() - t0 < timeout:
                try:
                    loc = browser._call("browser.getLocation", {})
                    u = loc.get("url", "")
                    t = loc.get("title", "")
                    if u and u != prev_url:
                        # brief settle
                        try:
                            time.sleep(0.05)
                            loc2 = browser._call("browser.getLocation", {})
                            return loc2.get("url", u), loc2.get("title", t)
                        except Exception:
                            return u, t
                    current_url, current_title = u or current_url, t or current_title
                except Exception:
                    pass
                time.sleep(0.05)
            # Timed out; return last observed
            return current_url, current_title

        try:
            # Check if already at goal
            if goal_reached(url):
                score = 1.0
                result = TaskResult(
                    task_id=task.task_id,
                    task_name=task.name,
                    agent_type=self.agent_type,
                    success=True,
                    score=score,
                    metrics={
                        "output_tokens": 0,
                        "steps_taken": 0,
                        "optimal_steps": task.evaluation_criteria["optimal_steps"],
                        "invalid_actions": 0,
                        "action_trace": action_trace,
                        "final_url": url,
                        "goal_reached": True,
                    },
                    execution_time=0.0,
                )
                browser.quit()
                return result

            last_fields: Dict[str, str] = {}
            # Phase 1: Exploration (use up to exploration_budget steps, executing first step of model plan each time)
            visited_urls: List[str] = [url]
            seen_selectors: List[str] = []
            seen_pages: List[Dict[str, Any]] = []
            guide_snapshot: Optional[Dict[str, Any]] = None
            for step in range(1, exploration_budget + 1):
                goal_summary = f"Navigate until path is '{goal['path']}' and element '{goal['selector']}' exists."
                # Attach running memory for the model to leverage
                observation_with_memory = dict(observation)
                observation_with_memory["exploration_memory"] = {
                    "visited_urls": visited_urls,
                    "seen_selectors": seen_selectors[-50:],
                    "seen_pages": seen_pages[-10:],
                }
                prompt = _build_exploration_prompt(observation_with_memory, goal_summary)

                response = await model.generate(prompt)
                accumulated_call_time += (response.latency or 0.0)
                if response.completion_tokens:
                    total_tokens += response.completion_tokens

                action = _parse_action(response.text)
                steps_taken += 1

                if not action or not isinstance(action, dict):
                    invalid_actions += 1
                    wait_seconds = float(self.config.additional_params.get("wait_seconds", 15.0)) if getattr(self, "config", None) else 15.0
                    if wait_seconds > 0:
                        await asyncio.sleep(wait_seconds)
                    # proceed to next turn without applying action
                    continue

                atype = action.get("type")
                try:
                    if atype == "click" and action.get("selector"):
                        prev_url = url
                        url, title = browser.click(action["selector"])
                        # Wait for potential navigation
                        u2, t2 = _wait_until_url_changes(prev_url)
                        url, title = (u2 or url), (t2 or title)
                        if url:
                            visited_urls.append(url)
                            if url.endswith('/site-guide'):
                                try:
                                    dom_full = browser.get_dom_summary(viewport_only=False, max_links=200, max_inputs=200, max_forms=200)
                                    guide_snapshot = {
                                        "url": url,
                                        "title": title,
                                        "elements": dom_full.get("elements", {}),
                                        "visible_text": (dom_full.get("visible_text", "") or "")[:2000],
                                    }
                                except Exception:
                                    pass
                    elif atype == "type" and action.get("selector") and "text" in action:
                        browser.type(action["selector"], action["text"])
                    elif atype == "submit" and action.get("selector"):
                        prev_url = url
                        url, title = browser.submit(action["selector"])
                        # Wait for post-submit navigation
                        u2, t2 = _wait_until_url_changes(prev_url)
                        url, title = (u2 or url), (t2 or title)
                        last_fields = {}
                        if url:
                            visited_urls.append(url)
                            if url.endswith('/site-guide'):
                                try:
                                    dom_full = browser.get_dom_summary(viewport_only=False, max_links=200, max_inputs=200, max_forms=200)
                                    guide_snapshot = {
                                        "url": url,
                                        "title": title,
                                        "elements": dom_full.get("elements", {}),
                                        "visible_text": (dom_full.get("visible_text", "") or "")[:2000],
                                    }
                                except Exception:
                                    pass
                    elif atype == "navigate":
                        # Disallow navigate in exploration
                        invalid_actions += 1
                    else:
                        invalid_actions += 1
                except Exception:
                    invalid_actions += 1

                action_trace.append({
                    "turn": step,
                    "phase": "exploration",
                    "action": action,
                    "result_url": url,
                    "result_title": title,
                })

                # New observation (still viewport-only summary)
                domk = browser.get_dom_summary(viewport_only=True)
                observation = {
                    "current_url": url,
                    "page_title": title,
                    "elements": domk.get("elements", {}),
                    "visible_text_preview": "",
                }
                # Update running memory
                try:
                    if isinstance(domk.get("elements"), dict):
                        clks = domk["elements"].get("clickables", []) or []
                        sels = [c.get("selector") for c in clks if isinstance(c, dict) and c.get("selector")]
                        seen_selectors.extend(s for s in sels if isinstance(s, str))
                except Exception:
                    pass
                try:
                    vt = domk.get("visible_text", "")
                except Exception:
                    vt = ""
                seen_pages.append({"url": url, "title": title, "visible_text": vt})

                # Note: Exploration is unscored. Even if goal is encountered here,
                # we do NOT return early; continue until budget is exhausted and proceed to exploitation.

                wait_seconds = float(self.config.additional_params.get("wait_seconds", 15.0)) if getattr(self, "config", None) else 15.0
                if wait_seconds > 0:
                    await asyncio.sleep(wait_seconds)

            # Phase 2: Exploitation — hard reset browser and server for clean slate
            try:
                browser.quit()
            except Exception:
                pass
            reset_server("127.0.0.1", 5005)
            browser = _MCPBrowser(self.base_url, mcp_url)
            url, title = browser.navigate(task.metadata["start_url"])
            dom = browser.get_dom_summary(viewport_only=True)
            observation_exploit = {
                "current_url": url,
                "page_title": title,
                "elements": dom.get("elements", {}),
                "visible_text_preview": "",
                "exploration_memory": {
                      "visited_urls": visited_urls,
                      "site_guide_dom": guide_snapshot,
                      "seen_pages": [{"url": p.get("url"), "title": p.get("title"), "visible_text": p.get("visible_text")[:500] if isinstance(p.get("visible_text"), str) else ""} for p in seen_pages[-20:]],
                },
            }
            goal_summary = f"Navigate until path is '{goal['path']}' and element '{goal['selector']}' exists (flow-specific). Start from a fresh session at {task.metadata['start_url']}."
            prompt = _build_exploitation_prompt(observation_exploit, goal_summary)

            response = await model.generate(prompt)
            accumulated_call_time += (response.latency or 0.0)
            if response.completion_tokens:
                total_tokens += response.completion_tokens

              # Parse and validate by executing; penalize invalid actions and failures
            plan_list = _parse_plan(response.text) or []
            optimal_from_start = task.evaluation_criteria.get("optimal_steps", 1)
            executed_steps = 0
            plan_failed = False
            for act in plan_list:
                try:
                    atype = act.get("type")
                    if atype == "click" and act.get("selector"):
                        prev_url = url
                        url, title = browser.click(act["selector"])
                        u2, t2 = _wait_until_url_changes(prev_url)
                        url, title = (u2 or url), (t2 or title)
                    elif atype == "type" and act.get("selector") and "text" in act:
                        browser.type(act["selector"], act["text"])
                    elif atype == "submit" and act.get("selector"):
                        prev_url = url
                        url, title = browser.submit(act["selector"])
                        u2, t2 = _wait_until_url_changes(prev_url)
                        url, title = (u2 or url), (t2 or title)
                    elif atype == "navigate":
                        invalid_actions += 1
                        plan_failed = True
                        break
                    else:
                        invalid_actions += 1
                        plan_failed = True
                        break
                    executed_steps += 1
                except Exception:
                    invalid_actions += 1
                    plan_failed = True
                    break

            goal_ok = goal_reached(url)
            if plan_failed or not goal_ok:
                plan_score = 0.0
                plan_len = len(plan_list)
            else:
                plan_len = max(1, len(plan_list))
                plan_score = max(0.0, min(1.0, (optimal_from_start / plan_len)))

            action_trace.append({
                "turn": exploration_budget + 1,
                "phase": "exploitation",
                "action": {"executed_steps": executed_steps, "plan": plan_list},
                "result_url": url,
                "result_title": title,
                    "plan_length": len(plan_list),
                "plan_score": plan_score,
            })

            optimal_steps = task.evaluation_criteria.get("optimal_steps", 1)
            result = TaskResult(
                task_id=task.task_id,
                task_name=task.name,
                agent_type=self.agent_type,
                success=plan_score > 0,
                score=plan_score,
                metrics={
                    "output_tokens": total_tokens,
                    "steps_taken": steps_taken,
                    "optimal_steps": optimal_steps,
                    "invalid_actions": invalid_actions,
                    "action_trace": action_trace,
                    "final_url": url,
                    "goal_reached": goal_reached(url),
                },
                model_response=ModelResponse(text=f"Plan length: {plan_len}, score: {plan_score:.3f}", total_tokens=total_tokens),
                execution_time=accumulated_call_time,
            )
            browser.quit()
            return result


        except Exception as e:
            logger.error(f"Error evaluating local web nav task {task.task_id}: {e}")
            result = TaskResult(
                task_id=task.task_id,
                task_name=task.name,
                agent_type=self.agent_type,
                success=False,
                score=0.0,
                metrics={"output_tokens": total_tokens},
                execution_time=accumulated_call_time,
                error_message=str(e),
            )
            browser.quit()
            return result

    def calculate_score(self, task: Task, model_response: ModelResponse) -> float:
        # Scoring is handled during evaluation
        return 0.0


