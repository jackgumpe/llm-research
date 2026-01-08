# Professional Prompt Engineering Masterclass

## 1. Deconstruction of Your Original Prompt

**Your Prompt:**
> "I have codex working in the same llm-research as yourself Gemini and he going to help over see the integration along with you. I posted all of the prompt chain conversation of Codex talking about OCR/ After reading the prompt conversation and after you have read his work in llm-research/agents and its subdirectories . Please keep the hot potato going and create your own tribunal agents that will grade, judge, debate, disagree, argue, agree and ultimately output the tribunals collective decision and agreement. Let them go back and forth a while. A good 50 rounds I feel is needed for a sophisticated project as ours. { "DEFINE": "PRODUCTION READY": "ready to integrate into other projects; completed AI/SOFTWARE stack", "MARKET READY": "acceptable metrics met in order for others without hiccups" }"

**Critique (Why it feels "Junior"):**
*   **Vague Instructions:** "Keep the hot potato going" and "go back and forth a while" are ambiguous. They don't specify *how* the debate should be structured or what the specific output format should be.
*   **Loose Constraints:** "A good 50 rounds" is a loose heuristic. It doesn't define what constitutes a "round" or what the milestone for ending the debate is (other than "agreement").
*   **Implicit Context:** It assumes the AI knows *exactly* what "Codex's work" entails without explicit pointers to specific files or key decisions to focus on.
*   **Informal Tone:** Phrases like "sophisticated project as ours" are conversational but don't add technical definition.

## 2. The "Senior Engineer at Google" Version

A senior engineer treats a prompt like a **technical specification**. It defines the *input state*, the *process*, and the *output artifacts* with precision.

### The Improved Prompt Structure:

1.  **Role & Objective:** Define the AI's persona and the ultimate goal clearly.
2.  **Context & Inputs:** Explicitly list the files, decisions, and constraints the AI must consider.
3.  **Process Definition:** Define the mechanism (Tribunal), the participants (Agents), and the rules of engagement (Protocol).
4.  **Output Specification:** Define exactly what the final result must look like (e.g., Markdown table, JSON, code diffs).
5.  **Quality Assurance:** Define success metrics (Production/Market Ready definitions).

### The Rewrite (Copy-Paste Ready):

```markdown
# TASK: Orchestrate Multi-Agent Technical Review & Gap Analysis (DeepSeek-OCR Integration)

## 1. OBJECTIVE
Act as the **Lead Technical Architect** for the `llm-research` project. Your goal is to conduct a rigorous architectural review of the proposed DeepSeek-OCR integration, identify all blocking issues preventing "Production Readiness," and synthesize a concrete **Gap Closure Plan**.

## 2. CONTEXT & INPUTS
You are reviewing the work of a previous agent ("Codex").
**Reference Files:**
*   `agents/proposals/deepseek_ocr_amendments.json`: Core architectural changes proposed.
*   `agents/ocr_agent.py`: Current agent implementation (Note: contains mocks).
*   `deepseek-ocr/scripts/`: Current implementation scripts (`run_ocr.py`, `environment.ps1`).
*   **Constraint:** The host hardware is an RTX 2070 (8GB VRAM).

## 3. PROCESS: THE TRIBUNAL
Instantiate a virtual tribunal of **7 Expert Agents** to debate the implementation.
**Round Limit:** 30 Rounds (or until consensus is reached).

### Agent Roster & Mandates:
1.  **System Architect:** Integration patterns, API contracts, path management.
2.  **ML Ops Engineer:** Model lifecycle, quantization pipelines, VRAM profiling.
3.  **DevOps/SRE:** Infrastructure, dependencies (Redis), deployment automation.
4.  **Security Analyst:** Credential management (HF Tokens), data provenance/audit.
5.  **Product Manager:** Usability, "Market Ready" user journey, documentation.
6.  **QA Engineer:** Testing strategy, validation datasets, success metrics.
7.  **Integration Specialist:** Service discovery, inter-agent messaging protocols.

### Debate Protocol:
*   **Phase 1 (Assessment):** Each agent audits the codebase against their mandate.
*   **Phase 2 (Debate):** Agents challenge each other's findings. (e.g., *DevOps* challenges *ML Ops* on quantization dependencies).
*   **Phase 3 (Synthesis):** Agents negotiate solutions to close identified gaps.

## 4. DEFINITIONS OF SUCCESS
*   **Production Ready:** A fully integrated, configurable software stack where all dependencies are managed, secrets are secured, and the system can be deployed to a fresh environment without manual intervention.
*   **Market Ready:** The system meets specific performance KPIs (CER ≤ 2%, Latency < 30s) and offers a seamless "one-command" setup experience for external researchers.

## 5. REQUIRED OUTPUT ARTIFACTS
At the conclusion of the debate, generate a **Gap Closure Plan** document containing:
1.  **Executive Summary:** High-level state of the system.
2.  **Critical Gap Analysis:** A prioritized table of missing components (e.g., "Missing Quantization Script").
3.  **Technical Specification:** Detailed requirements for fixing each gap (e.g., "Create `scripts/quantize.py` using `llama.cpp`").
4.  **Implementation Roadmap:** A phased timeline (Phase 1: Model Assets, Phase 2: Integration, etc.).

**Begin the Tribunal simulation now.**
```

## 3. Why This Works Better

*   **Explicit Roles:** It forces the model to look at the code from specific angles (Security, Ops, Architecture), ensuring coverage.
*   **Defined Artifacts:** It tells the model exactly *what* to produce (Gap Analysis, Technical Spec), so you don't just get a long chat log.
*   **Clear Definitions:** "Production Ready" is defined technically, not just conceptually.
*   **Actionable:** The output isn't just "thoughts," it's a plan you can execute.

Use this structure for your future complex tasks: **Role -> Context -> Process -> Output**. It will dramatically improve the quality of your AI interactions.
