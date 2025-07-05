Full Implementation Plan for “AgenticEvals” Benchmark
Main Recommendation: Establish a modular, reproducible benchmark suite evaluating LLMs across the five classic AI agent types (Simple Reflex, Model-Based Reflex, Goal-Based, Utility-Based, and Learning Agents) by adapting existing tasks and frameworks and/or creating novel tasks, fully documenting via Croissant format, and adhering to NeurIPS Datasets & Benchmarks Track guidelines.
1. Project Overview and Objectives
Goal: Develop AgenticEvals, a benchmark mapping LLM performance to each classic agent architecture through distinct scenario-based tasks, enabling per-agent-type scores and an aggregate “Agentic Versatility Index.”
Key Deliverables:
* Five task modules (one per agent type), each comprising:
    * Task specification and scenarios
    * Open-source code and containerized environments
    * Evaluation scripts and metrics
* Baseline results on representative open-source LLMs
* Croissant-formatted metadata for all datasets/environments
* Public GitHub repository with automated CI for reproducibility
* Leaderboard infrastructure for continuous evaluation
2. NeurIPS Datasets & Benchmarks Compliance
Requirement	Implementation Steps
Single-blind or Double-blind Option	Authors submit single-blind by default; coordinate anonymized examples for modules without external dependencies to enable double-blind review1.
Code & Dataset Submission	Host code on GitHub; Docker images for environments; Croissant metadata files included in OpenReview supplement12.
Data Accessibility	Use public hosting (Hugging Face, GitHub, OSF) ensuring no personal requests needed; verify access via Croissant online validator.
Scope Alignment	Tasks cover “benchmarks on new or existing datasets” and “reinforcement learning environments” as per track scope1.
3. Task Module Design
3.1 Simple Reflex Agent Module
* Objective: Test immediate rule-based responses.
* Scenarios:
    * Traffic-light simulator: LLM receives “red/green/yellow” prompts and replies “stop/go/caution.”
    * Email autoresponder: On detecting keywords (“invoice,” “urgent”), generate fixed reply templates.
* Metrics: Accuracy, average response latency.
* Resources: Adapt rule-following tasks from τ-bench3.
3.2 Model-Based Reflex Agent Module
* Objective: Evaluate use of internal state.
* Scenarios:
    * Textual maze: Partial observations per turn; LLM must recall visited cells to avoid loops.
    * Multi-step document verification: Aggregate scattered facts across pages to validate claims.
* Metrics: Success rate, memory footprint (tokens).
* Resources: Implement simple maze in Python; adapt τ-bench memory tasks3.
3.3 Goal-Based Agent Module
* Objective: Assess planning toward explicit goals.
* Scenarios:
    * Gridworld puzzles: Given start/goal, LLM outputs action sequence.
    * Web form automation: Sequential API calls to complete a booking.
* Metrics: Goal achievement rate, plan length vs. optimal, planning time.
* Resources: Leverage AgentBench’s planning tasks4.
3.4 Utility-Based Agent Module
* Objective: Measure utility maximization under trade-offs.
* Scenarios:
    * Simulated economy: Allocate budget across assets to maximize return.
    * Task scheduling: Assign jobs to time slots with varying rewards/costs.
* Metrics: Cumulative utility, decision optimality gap.
* Resources: Adapt simple RL economic environments (OpenAI Gym text analogs).
3.5 Learning Agent Module
* Objective: Evaluate adaptation over episodes.
* Scenarios:
    * Iterated Prisoner’s Dilemma with changing payoff matrices.
    * Repeated question-answer task with corrective feedback.
* Metrics: Learning curve (performance vs. episode), adaptation speed.
* Resources: Build on existing online-learning benchmarks.
4. Implementation Roadmap and Timeline
Phase	Duration	Activities	Leads
Phase 1: Planning & Setup	1 month	Finalize task specs; define API interfaces; set up GitHub repo; configure CI/Docker; draft Croissant metadata schema templates.	PM, Lead Engineer
Phase 2: Module Development	3 months	Implement environments and wrappers for each module; write evaluation scripts; integrate logging; preliminary baselines.	Engineering Team
Phase 3: Documentation	1 month	Author detailed module READMEs; generate Croissant metadata; validate metadata; prepare paper supplement.	Documentation Lead
Phase 4: Baseline Evaluation	1 month	Run open-source LLMs (e.g., GPT-2, LLaMA-2) across modules; collect metrics; analyze results.	Research Team
Phase 5: Leaderboard & Portal	1 month	Deploy leaderboard (e.g., via Weights & Biases); enable model submission interface; automate result ingestion.	DevOps
Phase 6: NeurIPS Submission	1.5 months	Prepare paper (methodology, experiments, analysis); assemble supplemental code and Croissant files; coordinate single-blind/double-blind options; submit by May 15.	All
5. Infrastructure and Tooling
* Repository: Public GitHub under organizational account.
* Containerization: Docker images per module; pytest suites for CI.
* Metadata: Croissant JSON-LD files alongside each task folder; validated via mIcroissant CLI2.
* Leaderboard: Hosted on Hugging Face Spaces or W&B, with REST API for submissions.
* Continuous Evaluation: GitHub Actions triggering evaluation on PRs.
6. Evaluation Metrics and Reporting
* Per-Module Metrics: As defined per agent type (see §3).
* Composite “Agentic Versatility Index”: Weighted sum of normalized scores across modules.
* Statistical Significance: Bootstrap confidence intervals; paired t-tests between models.
* Error Analysis: Qualitative breakdown of failure modes per task category.
7. Baseline Agents and Comparisons
Agent Type	Baseline Models	Symbolic Baseline
Simple Reflex	GPT-2, LLaMA-2	Hard-coded rule bot
Model-Based Reflex	GPT-3.5, Vicuna	DFS agent
Goal-Based	GPT-4, Claude-3	A* planner with heuristics
Utility-Based	ChatGPT, Llama-2-Chat	Greedy utility solver
Learning Agent	GPT-4-RLHF	Q-learning agent
8. Ethics, Bias, and Cost Considerations
* Fairness: Include scenarios that expose potential bias in utility decisions; publish demographic breakdowns if using real-world data.
* Cost Reporting: Log compute hours and API usage for each baseline model; report environmental footprint.
* Reproducibility & Openness: All code, data, and metadata public; containerized setups ensure identical results.
9. Risk Mitigation and Contingencies
* Overfitting to Toy Tasks: Introduce optional “realistic” variants (e.g., real-email corpora).
* Evaluation Inconsistency: Standardize RNG seeds; deterministic task wrappers.
* Metadata Errors: Automated Croissant validation in CI pipeline.
References
1 NeurIPS 2025 Datasets & Benchmarks Track Call for Papers (track submission requirements and guidelines)1. 2 Croissant metadata format for ML datasets (documentation)2. 4 AgentBench: Evaluating LLMs as Agents (task adaptation)4. 3 τ-bench: Tool-Agent-User Interaction Benchmark (memory tasks)3.
1. https://neurips.cc/Conferences/2025/CallForDatasetsBenchmarks
2. https://docs.mlcommons.org/croissant/
3. https://huggingface.co/papers/2406.12045
4. https://paperswithcode.com/dataset/agentbench
5. https://neurips.cc/Conferences/2024/CallForDatasetsBenchmarks
6. https://mlcommons.org/working-groups/data/croissant/
7. https://github.com/THUDM/AgentBench
8. https://github.com/huggingface/lighteval/issues/741
9. https://neurips.cc/Conferences/2022/CallForDatasetsBenchmarks
10. https://huggingface.co/docs/dataset-viewer/en/croissant
