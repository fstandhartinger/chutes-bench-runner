# Chutes Bench Runner: Benchmark Compliance Report

**Generated:** January 2026
**Purpose:** Validate implementation compliance with official benchmark specifications

---

## Executive Summary

This report analyzes the 14 core benchmarks implemented in chutes-bench-runner, comparing our implementation against official benchmark specifications. The analysis identifies compliance gaps, deviations, and areas requiring attention.

### Overall Compliance Summary

| Benchmark | Compliance Level | Critical Gaps | Notes |
|-----------|------------------|---------------|-------|
| MMLU-Pro | **Good** | None | 5-shot prompting enabled (per-category examples) |
| GPQA Diamond | **Good** | Minor prompt differences | Core evaluation logic correct |
| HLE | **Good** | Different judge model | Uses configurable judge instead of o3-mini |
| AIME 2025 | **Good** | None | 8-run variance reduction enabled |
| IFEval | **Excellent** | None | Uses official Google evaluation library |
| AA-LCR | **Good** | Different judge model | Simplified judge prompt |
| LiveCodeBench | **Partial** | Missing pass@k metrics | Only computes pass@1, no pass@5 |
| SciCode | **Good** | Minor | Multi-step prompting implemented correctly |
| Terminal-Bench | **Good** | None | Agentic execution via Sandy agent API |
| SWE-Bench Pro | **Good** | None | Agentic patch workflow via Sandy agent API |
| Tau-Bench | **Good** | None | Uses official tau2 framework |
| AA-Omniscience | **Good** | Judge model differences | Uses official rubric with configurable judge |
| GDPval-AA | **Partial** | Heuristic grading | Uses LLM judge vs reference docs |
| CritPt | **Partial** | External evaluator | Requires CritPt eval server, no per-item details |

---

## Detailed Benchmark Analysis

---

### 1. MMLU-Pro

**Official Source:** [TIGER-Lab/MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro) | [Paper](https://arxiv.org/abs/2406.01574)

#### Official Requirements

| Aspect | Official Specification |
|--------|----------------------|
| **Dataset** | TIGER-Lab/MMLU-Pro (12,000+ questions, 14 domains) |
| **Answer Choices** | 10 options (A-J) |
| **Prompting** | **5-shot Chain-of-Thought** is the standard method |
| **Temperature** | Not strictly specified, but 0.0 commonly used |
| **Scoring** | Accuracy (correct letter extraction) |

#### Our Implementation (`mmlu_pro.py`)

| Aspect | Our Implementation | Compliant? |
|--------|-------------------|------------|
| **Dataset** | TIGER-Lab/MMLU-Pro | Yes |
| **Answer Choices** | 10 options (A-J) | Yes |
| **Prompting** | 5-shot with per-category examples | **Yes** |
| **Temperature** | 0.0 | Yes |
| **Max Tokens** | 4096 | Reasonable |
| **Answer Extraction** | Robust letter extraction with `<think>` handling | Yes |

#### Status

Few-shot prompting has been implemented with 5 per-category examples sourced from the validation split.

#### Reference Prompt Template

```python
# Add 5-shot examples per category (from official repo's prompts)
# Example structure:
prompt = f"""The following are multiple choice questions (with answers) about {category}.

{example_1_question}
{example_1_answer_with_reasoning}

{example_2_question}
...

{test_question}
"""
```

---

### 2. GPQA Diamond

**Official Source:** [idavidrein/gpqa](https://github.com/idavidrein/gpqa) | [Paper](https://arxiv.org/abs/2311.12022)

#### Official Requirements

| Aspect | Official Specification |
|--------|----------------------|
| **Dataset** | Idavidrein/gpqa "gpqa_diamond" split (198 questions) |
| **Answer Choices** | 4 options (A-D) |
| **Prompting** | Zero-shot or 5-shot CoT both acceptable |
| **Answer Format** | "ANSWER: X" format strictly enforced |
| **Scoring** | Accuracy; if answer not in correct format, 0 points |

#### Our Implementation (`gpqa.py`)

| Aspect | Our Implementation | Compliant? |
|--------|-------------------|------------|
| **Dataset** | Idavidrein/gpqa "gpqa_diamond" (train split) | Yes |
| **Answer Choices** | 4 options, deterministically shuffled | Yes |
| **Prompting** | Zero-shot with system prompt | Yes |
| **System Prompt** | "You are a very intelligent assistant, who follows instructions directly." | Reasonable |
| **Answer Format** | "The correct answer is (X)" | Slightly different |
| **Answer Extraction** | Robust letter extraction | Yes |

#### Gaps Identified

1. **MINOR: Answer Format Deviation**
   - Official: "ANSWER: $ANSWER" format
   - Ours: "The correct answer is (insert answer here)"
   - Impact: Our extraction is more lenient, which may lead to slightly higher scores

2. **MINOR: Split Naming**
   - We use "train" split, official uses all data
   - The gpqa_diamond config is correct

#### Recommended Fix

```python
# Consider matching exact official prompt format:
prompt = f"""...\n\nFormat your response as: ANSWER: X"""
```

---

### 3. Humanity's Last Exam (HLE)

**Official Source:** [Scale AI Leaderboard](https://scale.com/leaderboard/humanitys_last_exam) | [CAIS/HLE](https://agi.safe.ai/)

#### Official Requirements

| Aspect | Official Specification |
|--------|----------------------|
| **Dataset** | cais/hle (2,500 public questions, multimodal) |
| **Temperature** | 0.0 |
| **System Prompt** | Specific format requiring Explanation, Answer, Confidence |
| **Judge Model** | **o3-mini-2025-01-31** as automatic extractor/judge |
| **Scoring** | Automatic via structured JSON extraction |

#### Our Implementation (`hle.py`)

| Aspect | Our Implementation | Compliant? |
|--------|-------------------|------------|
| **Dataset** | cais/hle test split | Yes |
| **Temperature** | 0.0 | Yes |
| **System Prompt** | Matches official format | Yes |
| **Multimodal** | Supports image_url in messages | Yes |
| **Judge Model** | Configurable via `settings.hle_judge_model` | **Deviation** |
| **Judge Prompt** | Custom extraction prompt | Similar intent |

#### Gaps Identified

1. **MODERATE: Different Judge Model**
   - Official: o3-mini-2025-01-31
   - Ours: Configurable (likely different model)
   - Impact: "Small differences could arise from different judge models and prompts used on edge cases"

2. **MINOR: Judge Prompt Differences**
   - Our prompt is similar but not identical to official
   - We use JSON extraction with fallback regex parsing

#### Recommended Fix

```python
# Consider setting default judge to o3-mini or equivalent
HLE_JUDGE_MODEL = "o3-mini-2025-01-31"  # Or closest available
```

---

### 4. AIME 2025

**Official Source:** [AI-MO/aimo-validation-aime](https://huggingface.co/datasets/AI-MO/aimo-validation-aime) | [GAIR-NLP Methodology](https://github.com/GAIR-NLP/AIME-Preview)

#### Official Requirements

| Aspect | Official Specification |
|--------|----------------------|
| **Dataset** | 90 problems from AIME 22-24 (validation); AIME 2025 for latest |
| **Answer Format** | Integer 0-999, extracted from `\boxed{}` |
| **Variance Reduction** | **8 runs per problem** recommended |
| **Temperature** | Multiple: [0.0, 0.3, 0.6] tested |
| **Sampling** | n_sampling of 8 samples per question |
| **Scoring** | Pass@1 averaged across runs |

#### Our Implementation (`aime.py`)

| Aspect | Our Implementation | Compliant? |
|--------|-------------------|------------|
| **Dataset** | AI-MO/aimo-validation-aime | Yes |
| **Answer Format** | "ANSWER: <integer>" or `\boxed{}` | Yes |
| **Variance Reduction** | 8 runs per problem | **Yes** |
| **Temperature** | [0.0, 0.3, 0.6] schedule | **Yes** |
| **Sampling** | n=8 | **Yes** |
| **Max Tokens** | 65536 (appropriate for reasoning) | Yes |

#### Status

AIME now runs 8 samples per question with the recommended temperature schedule and aggregates scores.

#### Reference

```python
# For rigorous evaluation, consider:
# 1. Multiple runs per problem (at least 8)
# 2. Report confidence intervals
# 3. Option to test multiple temperatures
```

---

### 5. IFEval (IFBench)

**Official Source:** [Google Research IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) | [Paper](https://arxiv.org/abs/2311.07911)

#### Official Requirements

| Aspect | Official Specification |
|--------|----------------------|
| **Dataset** | google/IFEval (~500 prompts) |
| **Scoring** | Prompt-level and Instruction-level accuracy |
| **Modes** | Strict and Loose evaluation |
| **Instructions** | 25 types of verifiable instructions |

#### Our Implementation (`ifbench.py`)

| Aspect | Our Implementation | Compliant? |
|--------|-------------------|------------|
| **Dataset** | google/IFEval train split | Yes |
| **Evaluation Library** | **Official IFEval library** (`app/benchmarks/ifeval/`) | **Yes** |
| **Scoring** | `test_instruction_following_strict()` | Yes |
| **NLTK Dependencies** | punkt, punkt_tab tokenizers | Yes |

#### Compliance: EXCELLENT

Our implementation uses the **official Google IFEval evaluation library** (`evaluation_lib.py`) directly. This is one of the most compliant implementations.

#### Minor Notes

- We only report strict mode, not loose mode (but strict is more rigorous)
- Consider adding instruction-level metrics alongside prompt-level

---

### 6. AA-LCR (Artificial Analysis Long Context Reasoning)

**Official Source:** [ArtificialAnalysis/AA-LCR](https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR)

#### Official Requirements

| Aspect | Official Specification |
|--------|----------------------|
| **Dataset** | 100 questions across ~100k tokens each |
| **Document Loading** | ZIP file with extracted text documents |
| **Scoring** | LLM-based equality checker |
| **Judge Model** | Varies by evaluator (Qwen3 235B mentioned) |

#### Our Implementation (`aalcr.py`)

| Aspect | Our Implementation | Compliant? |
|--------|-------------------|------------|
| **Dataset** | ArtificialAnalysis/AA-LCR test split | Yes |
| **Document Loading** | ZIP extraction from HuggingFace | Yes |
| **Context Building** | Documents concatenated with headers | Yes |
| **Judge Model** | Configurable via `settings.aa_lcr_judge_model` | Reasonable |
| **Judge Prompt** | Simplified CORRECT/INCORRECT | Simplified |

#### Gaps Identified

1. **MINOR: Simplified Judge Prompt**
   - Official may use more nuanced semantic equivalence
   - Ours uses binary CORRECT/INCORRECT
   - Impact: Edge cases may be judged differently

2. **MINOR: Different Judge Model**
   - Official uses various models for judging
   - Ours is configurable

---

### 7. LiveCodeBench

**Official Source:** [LiveCodeBench/LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) | [Paper](https://arxiv.org/abs/2403.07974)

#### Official Requirements

| Aspect | Official Specification |
|--------|----------------------|
| **Dataset** | livecodebench/code_generation (880-1055 problems) |
| **Metrics** | **Pass@1 and Pass@5** |
| **Temperature** | 0.2 for generation |
| **Samples** | n=10 completions per problem |
| **Execution** | Custom checker (modified APPS benchmark) |
| **Timeout** | Configurable per problem |

#### Our Implementation (`livecodebench.py`)

| Aspect | Our Implementation | Compliant? |
|--------|-------------------|------------|
| **Dataset** | livecodebench/code_generation test split | Yes |
| **Metrics** | **Pass@1 only** | **Partial** |
| **Temperature** | 0.0 | Different |
| **Samples** | n=1 | **NO** |
| **Execution** | Sandy sandbox with I/O testing | Custom |
| **Tests** | Public + Private test cases | Yes |

#### Gaps Identified

1. **MODERATE: Missing Pass@5 Metric**
   - Official: Reports both pass@1 and pass@5
   - Ours: Only pass@1 (single attempt)
   - Impact: Can't compare directly with leaderboards reporting pass@5

2. **MODERATE: Different Sampling**
   - Official: n=10 completions, temperature=0.2
   - Ours: n=1 completion, temperature=0.0
   - Impact: Greedy decoding may give different results

3. **MINOR: Custom Execution Environment**
   - Official: Uses modified APPS checker
   - Ours: Uses Sandy sandbox
   - Should be functionally equivalent

#### Recommended Fix

```python
# Option to enable multiple samples:
# n_samples = 10
# temperature = 0.2
# Then compute pass@1 and pass@5 from samples
```

---

### 8. SciCode

**Official Source:** [SciCode1/SciCode](https://huggingface.co/datasets/SciCode1/SciCode) | [Paper](https://arxiv.org/abs/2407.13168)

#### Official Requirements

| Aspect | Official Specification |
|--------|----------------------|
| **Dataset** | 80 problems (15 validation, 65 test) |
| **Structure** | Multi-step sub-problems |
| **Execution** | HDF5 test data with np.allclose() |
| **Dependencies** | numpy, scipy, sympy, h5py |
| **Template** | Official multistep_template.txt |

#### Our Implementation (`scicode.py`)

| Aspect | Our Implementation | Compliant? |
|--------|-------------------|------------|
| **Dataset** | SciCode1/SciCode test split | Yes |
| **Multi-Step Prompting** | Sequential step generation | Yes |
| **Template** | Downloads official template | **Yes** |
| **HDF5 Test Data** | Downloads from HuggingFace | Yes |
| **Special Steps** | Handles 13.6, 62.1, 76.3 overrides | Yes |
| **Execution** | Sandy sandbox with numpy/scipy | Yes |

#### Compliance: GOOD

Our implementation correctly:
- Uses official multi-step prompting template
- Downloads official HDF5 test data
- Handles special step overrides
- Validates with np.allclose()

#### Minor Notes

- Consider adding background template option (`with_background=True`)
- Currently uses default template without background

---

### 9. Terminal-Bench Hard

**Official Source:** [laude-institute/terminal-bench](https://github.com/laude-institute/terminal-bench) | [ia03/terminal-bench](https://huggingface.co/datasets/ia03/terminal-bench)

#### Official Requirements

| Aspect | Official Specification |
|--------|----------------------|
| **Dataset** | ~100 terminal tasks |
| **Execution** | Docker-containerized environments |
| **Agent** | Iterative agent loop (observe-act-evaluate) |
| **Verification** | Automated test scripts |
| **Harness** | Official `tb` CLI harness |

#### Our Implementation (`terminal_bench.py`)

| Aspect | Our Implementation | Compliant? |
|--------|-------------------|------------|
| **Dataset** | ia03/terminal-bench test split | Yes |
| **Docker** | Uses Sandy with Docker socket | Yes |
| **Agent** | Agentic execution via Sandy agent API | **Yes** |
| **Verification** | Task-provided test scripts | Yes |
| **Compose Support** | docker-compose handling | Yes |

#### Status

Terminal-Bench now uses Sandy's agent execution, enabling iterative command execution and file edits.

```python
# Implement iterative agent loop:
while not task_complete:
    observation = get_container_state()
    action = model.generate_action(history + observation)
    result = execute_action(action)
    history.append((action, result))
```

---

### 10. SWE-Bench Pro

**Official Source:** [ScaleAI/SWE-bench_Pro](https://scale.com/leaderboard/swe_bench_pro_public) | [Paper](https://arxiv.org/abs/2509.16941)

#### Official Requirements

| Aspect | Official Specification |
|--------|----------------------|
| **Dataset** | 1,865 problems from 41 repositories |
| **Evaluation** | Fail-to-pass + Pass-to-pass tests |
| **Environment** | Containerized with all dependencies |
| **Agent** | Agentic workflow (explore, edit, test) |
| **Timeout** | Hours to days per problem (for humans) |

#### Our Implementation (`swe_bench.py`)

| Aspect | Our Implementation | Compliant? |
|--------|-------------------|------------|
| **Dataset** | ScaleAI/SWE-bench_Pro test split | Yes |
| **Docker Images** | Uses official DockerHub images | Yes |
| **Test Harness** | Downloads official run scripts | Yes |
| **Agent** | Agentic workflow via Sandy agent API | **Yes** |
| **Scoring** | Fail-to-pass + Pass-to-pass | Yes |

#### Status

SWE-Bench Pro now runs an agent inside a sandboxed repo checkout and derives the patch from the agent's edits before running the official harness.

```python
# For true SWE-Bench Pro evaluation:
# 1. Implement agentic loop with file browsing
# 2. Allow model to run tests and iterate
# 3. Provide repository access tools
```

---

### 11. Tau-Bench Telecom

**Official Source:** [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)

#### Official Requirements

| Aspect | Official Specification |
|--------|----------------------|
| **Framework** | Official tau2 simulation library |
| **Domain** | Telecom (customer service) |
| **Agent** | LLM-based agent interacting with simulator |
| **User** | User simulator for multi-turn interaction |
| **Scoring** | Reward-based (0.0 to 1.0) |

#### Our Implementation (`tau_bench.py`)

| Aspect | Our Implementation | Compliant? |
|--------|-------------------|------------|
| **Framework** | **Uses official tau2 library** | **Yes** |
| **Domain** | telecom, base split | Yes |
| **Agent** | llm_agent via tau2 | Yes |
| **User Simulator** | user_simulator via tau2 | Yes |
| **Configuration** | OpenAI-compatible API routing | Yes |

#### Compliance: GOOD

Our implementation correctly uses the official tau2-bench framework with:
- Official `run_tasks()` function
- Proper model configuration via OpenAI-compatible API
- Configurable agent and user models
- Reward-based scoring

---

### 12. AA-Omniscience

**Official Source:** [ArtificialAnalysis/AA-Omniscience-Public](https://huggingface.co/datasets/ArtificialAnalysis/AA-Omniscience-Public) | [Paper](https://huggingface.co/datasets/ArtificialAnalysis/AA-Omniscience-Public)

#### Official Requirements

| Aspect | Official Specification |
|--------|----------------------|
| **Dataset** | 10% public subset (600 questions) |
| **Prompting** | System prompt with domain + category |
| **Scoring** | Omniscience Index (OI) via GPT-4 judge |
| **Judge Rubric** | CORRECT / INCORRECT / PARTIAL / NOT ATTEMPTED |

#### Our Implementation (`aa_omniscience.py`)

| Aspect | Our Implementation | Compliant? |
|--------|-------------------|------------|
| **Dataset** | Public 10% set | Yes |
| **System Prompt** | Matches official Appendix A.1 | Yes |
| **Judge Rubric** | Matches official Appendix A.2 | Yes |
| **Judge Model** | Configurable via `AA_OMNISCIENCE_JUDGE_MODEL` | **Deviation** |
| **Scoring** | Omniscience Index (OI) | Yes |

#### Gaps Identified

1. **MODERATE: Judge Model Differences**
   - Official uses GPT-4; ours is configurable (defaults to Qwen3 235B)
   - Edge-case grading may differ across judge models

---

### 13. GDPval-AA

**Official Source:** [openai/gdpval](https://huggingface.co/datasets/openai/gdpval) | [Paper](https://arxiv.org/abs/2510.04374)

#### Official Requirements

| Aspect | Official Specification |
|--------|----------------------|
| **Dataset** | 220 real-world tasks + reference files |
| **Evaluation** | Human review + official evaluation service |
| **Inputs** | Multi-modal documents (PDF, XLSX, DOCX, images, audio) |

#### Our Implementation (`gdpval.py`)

| Aspect | Our Implementation | Compliant? |
|--------|-------------------|------------|
| **Dataset** | openai/gdpval | Yes |
| **Reference Files** | Text extraction for PDF/XLSX/DOCX/CSV | Partial |
| **Evaluation** | LLM judge using extracted docs | **Deviation** |

#### Gaps Identified

1. **CRITICAL: Evaluation Method**
   - Official evaluation uses human review or external grader service
   - Our scoring is an LLM-judge heuristic against extracted reference docs
2. **MODERATE: Multi-modal Files**
   - Images/audio are not parsed; tasks requiring them may be under-scored

---

### 14. CritPt

**Official Source:** [CritPt-Benchmark/CritPt](https://huggingface.co/datasets/CritPt-Benchmark/CritPt) | [Docs](https://artificialanalysis.ai)

#### Official Requirements

| Aspect | Official Specification |
|--------|----------------------|
| **Dataset** | 70 physics research coding tasks |
| **Evaluation** | Official CritPt evaluation server |
| **Submission** | All problems in a single batch |

#### Our Implementation (`critpt.py`)

| Aspect | Our Implementation | Compliant? |
|--------|-------------------|------------|
| **Dataset** | CritPt-Benchmark/CritPt | Yes |
| **Prompting** | Uses official system prompt template | Yes |
| **Evaluation** | External CritPt eval server | Yes |
| **Subset Sampling** | Disabled (full batch only) | Yes |

#### Gaps Identified

1. **MODERATE: Per-item Results**
   - Evaluation server returns aggregate metrics; per-item grading is not stored

---

## Summary of Critical Gaps

### High Priority (Significant Impact on Scores)

| Benchmark | Gap | Impact | Difficulty to Fix |
|-----------|-----|--------|-------------------|
| MMLU-Pro | Missing 5-shot prompting | 5-15% score difference | Medium |
| AIME 2025 | No variance reduction | High variance in scores | Medium |
| Terminal-Bench | No agentic loop | Tasks requiring iteration fail | High |
| SWE-Bench Pro | No agentic loop | Complex problems fail | High |
| LiveCodeBench | Missing pass@5 | Can't compare with leaderboards | Medium |
| GDPval-AA | Non-official evaluation | Scores differ from official human eval | Medium |

### Medium Priority (Minor Score Impact)

| Benchmark | Gap | Impact | Difficulty to Fix |
|-----------|-----|--------|-------------------|
| HLE | Different judge model | Edge case differences | Low |
| GPQA | Prompt format differences | Minor | Low |
| AA-LCR | Simplified judge prompt | Edge cases | Low |

### Low Priority (Cosmetic/Optional)

| Benchmark | Gap | Impact |
|-----------|-----|--------|
| SciCode | Background template optional | N/A |
| IFEval | Only strict mode reported | N/A |

---

## Recommendations

### Immediate Actions

1. **MMLU-Pro**: ✅ 5-shot examples per category now enabled
2. **LiveCodeBench**: Add option for n>1 sampling and temperature=0.2
3. **AIME**: ✅ 8-run variance reduction enabled

### Medium-Term Improvements

4. **Terminal-Bench**: ✅ Agentic loop implemented via Sandy agent API
5. **SWE-Bench Pro**: ✅ Agentic workflow implemented via Sandy agent API

### Documentation

6. Clearly document deviations from official benchmarks in UI/reports
7. Add "Evaluation Mode" indicator (e.g., "Simplified" vs "Official")

---

## Appendix: Benchmark-Specific Configuration Recommendations

### For Maximum Official Compliance

```python
# MMLU-Pro
MMLU_PRO_FEW_SHOT = 5
MMLU_PRO_USE_COT = True

# AIME
AIME_NUM_RUNS = 8
AIME_TEMPERATURES = [0.0, 0.3, 0.6]

# LiveCodeBench
LIVECODEBENCH_N_SAMPLES = 10
LIVECODEBENCH_TEMPERATURE = 0.2

# HLE
HLE_JUDGE_MODEL = "o3-mini-2025-01-31"  # Or equivalent
```

---

## Appendix B: Research Findings - Mandatory vs Best Practice

### MMLU-Pro: 5-Shot Prompting Analysis

**Question:** Is 5-shot Chain-of-Thought mandatory or optional?

**Finding:** 5-shot CoT is the **STANDARD method** used by most evaluators, but **NOT strictly mandatory**.

| Source | Finding |
|--------|---------|
| [TIGER-Lab README](https://github.com/TIGER-AI-Lab/MMLU-Pro) | "Different answer extraction mechanisms have minor impact on results" - suggests flexibility |
| [TIGER-Lab README](https://github.com/TIGER-AI-Lab/MMLU-Pro) | Tested "24 different prompt styles" showing robustness across methods |
| [Artificial Analysis](https://artificialanalysis.ai/evaluations/mmlu-pro) | "Normally 5-shot prompting is used, though some models like Gemini use 0-shot" |
| [TIGER-Lab Paper](https://arxiv.org/abs/2406.01574) | "CoT can be 20% higher than PPL" - CoT strongly recommended for accuracy |
| [ProjectPro Guide](https://www.projectpro.io/article/mmlu-benchmark/1162) | "MMLU-Pro requires CoT reasoning to achieve better results" |

**Conclusion:**
- **5-shot CoT is the STANDARD** used by official leaderboard submissions
- 0-shot is acceptable but will produce **lower scores** (potentially 15-20% lower)
- For **comparable results with leaderboards**, 5-shot CoT should be used
- The benchmark is designed to be robust across prompting methods, but CoT is strongly recommended

**Recommendation:** Implement 5-shot CoT to align with how most organizations report MMLU-Pro scores.

---

### AIME 2025: Variance Reduction Analysis

**Question:** Are 8 runs per problem a mandatory requirement or best practice?

**Finding:** Multiple runs are a **BEST PRACTICE** for reducing variance, **NOT an official requirement**.

| Source | Finding |
|--------|---------|
| [GAIR-NLP/AIME-Preview](https://github.com/GAIR-NLP/AIME-Preview) | "Sample 8 times per question" - their methodology, not official AIME standard |
| [GAIR-NLP/AIME-Preview](https://github.com/GAIR-NLP/AIME-Preview) | Testing temperatures [0.0, 0.3, 0.6] then averaging - variance reduction technique |
| Research | AIME problems have high variance due to reasoning complexity |

**Conclusion:**
- There is **no official AIME AI benchmark specification** - various organizations use different methodologies
- 8 runs with temperature variation is GAIR-NLP's approach for rigorous evaluation
- Single-run at temperature=0.0 is acceptable but will have **higher variance**
- Scores can vary by **7-15%** between runs due to random seed

**Recommendation:**
- **Minimum:** Keep single run for speed, document variance caveat
- **Preferred:** Add optional multi-run mode (n=8) for rigorous evaluation
- Report confidence intervals when multiple runs are available

---

## Appendix C: Agentic Benchmarks - Deep Dive

### Terminal-Bench and SWE-Bench Pro: Agentic Requirements

**Question:** Do these benchmarks require an agentic approach? Is single-shot generation compliant?

**Finding:** YES, both benchmarks are **DEFINITIVELY AGENTIC** by design. Single-shot is NOT compliant.

---

### Terminal-Bench: Official Requirements

| Aspect | Official Specification | Source |
|--------|----------------------|--------|
| **Execution Model** | Interactive agent loop | [Terminal-Bench Docs](https://www.tbench.ai/) |
| **Agent Integration** | "The harness works by installing the agent directly into the container" | [Snorkel Blog](https://snorkel.ai/blog/terminal-bench-2-0-raising-the-bar-for-ai-agent-evaluation/) |
| **Interaction** | "The agent actually had a live container that was running that it would interact with by executing these commands" | [Terminal-Bench Team](https://snorkel.ai/blog/chat-with-the-terminal-bench-team/) |
| **Training Loop** | "reset → observation → LLM inference → action → environment step → reward → repeat" | [Harbor Framework](https://www.tbench.ai/news/announcement-2-0) |

**Official Agent Loop:**
```
1. Agent receives task instruction
2. Agent observes container state (files, processes, etc.)
3. Agent decides on action (shell command)
4. Action is executed in container
5. Agent receives stdout/stderr feedback
6. Steps 2-5 repeat until task is complete or max steps reached
7. Test script verifies final state
```

**Why Single-Shot Fails:**
- Many tasks require **trial and error** (e.g., debugging a configuration)
- Tasks may have **hidden dependencies** only discoverable through exploration
- Official benchmark expects agents to **learn from feedback**

---

### SWE-Bench Pro: Official Requirements

| Aspect | Official Specification | Source |
|--------|----------------------|--------|
| **Scaffold Requirement** | "Models operate within scaffolds: frameworks providing specialized tools" | [Epoch AI](https://epoch.ai/blog/what-skills-does-swe-bench-verified-evaluate) |
| **Tools Required** | "Edit files, navigate directories, run bash commands" | [OpenAI Blog](https://openai.com/index/introducing-swe-bench-verified/) |
| **Impact of Scaffold** | "A good scaffold can increase performance by up to 20%" | [Epoch AI](https://epoch.ai/blog/what-skills-does-swe-bench-verified-evaluate) |
| **Example: SWE-Agent** | "File viewer showing 100 lines at once, edit tool with linter, search tools" | [SWE-Agent Docs](https://www.swebench.com/SWE-bench/) |
| **Step Limit** | "Maximum of 150 steps per task" | [Scale Leaderboard](https://scale.com/leaderboard/swe_bench_pro_public) |
| **Token Limit** | "Hard limit of 1,000,000 tokens" | [Scale Leaderboard](https://scale.com/leaderboard/swe_bench_pro_public) |

**Official Agent Scaffold Components:**
```
1. File System Access: Read, write, search files in repository
2. Code Navigation: Jump to definitions, find references
3. Test Execution: Run tests and see results
4. Diff Generation: Create patches iteratively
5. Error Feedback: Syntax errors, test failures inform next action
```

**Why Single-Shot Fails:**
- Problems require **understanding existing code** before patching
- Patches often fail on first attempt - need **iteration**
- Official evaluation benchmarks **agent + model**, not just model
- Top performers (Claude, GPT-5) use sophisticated scaffolds

---

## Appendix D: Implementation Options for Agentic Benchmarks

### Current State: Sandy + Chutes-Webcoder

Based on exploration of the Sandy and chutes-webcoder projects:

**Sandy Already Has:**
- Pre-installed agents: Claude Code, OpenAI Codex, Aider, OpenCode
- Docker socket support (needed for Terminal-Bench containers)
- Command execution API with environment variable injection
- File read/write APIs
- Configurable API endpoints for LLM routing

**Chutes-Webcoder Demonstrates:**
- Full agent execution pipeline with multiple agent types
- Model wiring via environment variables (API keys, base URLs)
- Agent output parsing and file change detection
- Fallback chain between agents

---

### Option 1: Native Agent Loop in Benchmark Adapter (Recommended for Terminal-Bench)

**Approach:** Implement agent loop directly in the benchmark adapter using Sandy APIs.

```python
# Pseudo-code for Terminal-Bench agentic adapter
class TerminalBenchAgenticAdapter(BenchmarkAdapter):
    async def evaluate_item(self, item_id: str) -> ItemResult:
        # 1. Create sandbox with Docker socket
        sandbox_id = await self.sandy.create_sandbox(enable_docker_socket=True)

        # 2. Extract and setup task container
        await self._setup_task_container(sandbox_id, item)

        # 3. Run agentic loop
        history = []
        for step in range(MAX_STEPS):
            # Get current observation
            observation = await self._get_observation(sandbox_id)

            # Build prompt with history + observation
            prompt = self._build_agent_prompt(item.instruction, history, observation)

            # Get model action
            response, _ = await self.client.get_completion_text(
                self.model_slug, prompt, max_tokens=4096
            )

            # Parse and execute action
            action = self._parse_action(response)
            result = await self.sandy.execute_command(sandbox_id, action.command)

            # Update history
            history.append({"action": action, "result": result})

            # Check if done
            if self._is_done(response) or step == MAX_STEPS - 1:
                break

        # 4. Run verification tests
        is_correct = await self._run_tests(sandbox_id)
        return ItemResult(item_id=item_id, is_correct=is_correct, ...)
```

**Pros:**
- Full control over agent loop
- No external dependencies
- Can customize for benchmark-specific needs

**Cons:**
- Must implement agent prompting/parsing
- Less sophisticated than production agents

---

### Option 2: Use Pre-Installed Agents in Sandy (Recommended for SWE-Bench Pro)

**Approach:** Leverage Claude Code or Codex already installed in Sandy runtime.

```python
# Pseudo-code for SWE-Bench with Claude Code agent
class SWEBenchAgenticAdapter(BenchmarkAdapter):
    async def evaluate_item(self, item_id: str) -> ItemResult:
        # 1. Create sandbox with Docker socket
        sandbox_id = await self.sandy.create_sandbox(enable_docker_socket=True)

        # 2. Setup repository
        await self._setup_repository(sandbox_id, item)

        # 3. Configure agent environment
        env = {
            "ANTHROPIC_API_KEY": self.client.api_key,
            "ANTHROPIC_BASE_URL": "https://claude.chutes.ai",  # Or Chutes API
        }

        # 4. Run Claude Code with task prompt
        task_prompt = f"""
        Fix the following GitHub issue in this repository.

        Issue: {item.problem_statement}

        The repository is already cloned at /workspace/repo.
        Create a patch that fixes the issue and passes the tests.
        """

        result = await self.sandy.execute_command(
            sandbox_id,
            f'claude-code --print "{task_prompt}"',
            env=env,
            timeout_ms=3600000,  # 1 hour
        )

        # 5. Run SWE-Bench test harness
        is_correct = await self._run_swe_bench_tests(sandbox_id, item)
        return ItemResult(item_id=item_id, is_correct=is_correct, ...)
```

**Pros:**
- Production-quality agent with sophisticated tooling
- Minimal implementation effort
- Consistent with how real users would use the model

**Cons:**
- Agent behavior is a black box
- Claude Code requires Anthropic API format (may need adapter for Chutes models)
- Codex requires OpenAI API format

---

### Option 3: Sandy Agent Feature (Proposed New Sandy Capability)

**Approach:** Add a first-class "agent-ready sandbox" feature to Sandy.

**Proposed Sandy API Extension:**

```python
# POST /api/sandboxes
{
    "agent": {
        "type": "claude-code",  # or "codex", "aider", "opencode"
        "model": "claude-sonnet-4",  # Model to use
        "api_type": "anthropic",  # or "openai"
        "api_base_url": "https://claude.chutes.ai",
        "api_key": "sk-..."  # Or use CHUTES_API_KEY from server env
    },
    "enable_docker_socket": true,
    "timeout_minutes": 60
}

# Response
{
    "sandbox_id": "abc123",
    "agent_ready": true,
    "agent_command": "claude-code",  # Pre-configured command to run
    "url": "https://abc123.sandy.example.com"
}

# POST /api/sandboxes/{id}/agent/run
{
    "prompt": "Fix the bug in main.py",
    "working_directory": "/workspace",
    "timeout_ms": 3600000
}

# Response (streaming or final)
{
    "status": "completed",
    "output": "...",
    "files_modified": ["main.py", "test_main.py"],
    "exit_code": 0
}
```

**Implementation Details:**

1. **Agent Configuration in Runtime:**
   ```dockerfile
   # In Sandy runtime Dockerfile (already exists)
   RUN npm install -g @anthropic-ai/claude-code @openai/codex
   ```

2. **Dynamic API Wiring:**
   ```python
   # In Sandy API
   def setup_agent_env(agent_config):
       env = {}
       if agent_config["api_type"] == "anthropic":
           env["ANTHROPIC_API_KEY"] = agent_config["api_key"]
           env["ANTHROPIC_BASE_URL"] = agent_config["api_base_url"]
       elif agent_config["api_type"] == "openai":
           env["OPENAI_API_KEY"] = agent_config["api_key"]
           env["OPENAI_BASE_URL"] = agent_config["api_base_url"]
       return env
   ```

3. **Model-Agnostic Support:**
   - Claude Code: Works with Anthropic Messages API
   - Codex: Works with OpenAI Chat Completions API
   - Aider: Works with both (configurable)
   - OpenCode: Works with OpenAI-compatible APIs

**Chutes Model Compatibility:**

| Agent | API Format Required | Chutes Endpoint | Compatible? |
|-------|---------------------|-----------------|-------------|
| Claude Code | Anthropic Messages | `claude.chutes.ai` | Yes (native) |
| Codex | OpenAI Chat Completions | `llm.chutes.ai/v1` | Yes (OpenAI-compatible) |
| Aider | Both | Both | Yes |
| OpenCode | OpenAI-compatible | `llm.chutes.ai/v1` | Yes |

**Pros:**
- Clean API for benchmark runners
- Reusable across multiple benchmarks
- Abstracts agent complexity
- Supports different agent types and models

**Cons:**
- Requires Sandy development effort
- Need to handle agent-specific quirks

---

### Recommendation Matrix

| Benchmark | Recommended Option | Rationale |
|-----------|-------------------|-----------|
| **Terminal-Bench** | Option 1 (Native Loop) | Simpler tasks, native loop sufficient, more control |
| **SWE-Bench Pro** | Option 2 or 3 (Pre-installed Agent) | Complex tasks need sophisticated tooling |

---

## Appendix E: Sandy Agent Feature Specification

### Proposed Feature: Agent-Ready Sandboxes

**Goal:** Enable Sandy users to create sandboxes pre-configured with coding agents, wired to any compatible LLM.

### API Specification

#### 1. Create Agent-Ready Sandbox

```
POST /api/sandboxes
Content-Type: application/json
Authorization: Bearer <SANDY_API_KEY>

{
    "agent": {
        "type": "claude-code" | "codex" | "aider" | "opencode",
        "model": "<model-name>",
        "api_type": "anthropic" | "openai",
        "api_base_url": "<base-url>",
        "api_key": "<api-key>"
    },
    "enable_docker_socket": false,
    "timeout_minutes": 30,
    "working_directory": "/workspace"
}
```

**Response:**
```json
{
    "sandbox_id": "sb_abc123",
    "status": "ready",
    "agent": {
        "type": "claude-code",
        "ready": true,
        "command": "claude-code"
    },
    "url": "https://sb_abc123.sandy.94.130.222.43.nip.io",
    "timeout_at": "2026-01-09T15:30:00Z"
}
```

#### 2. Run Agent Task

```
POST /api/sandboxes/{sandbox_id}/agent/run
Content-Type: application/json

{
    "prompt": "<task-description>",
    "working_directory": "/workspace",
    "timeout_ms": 3600000,
    "stream": false
}
```

**Response (non-streaming):**
```json
{
    "status": "completed" | "failed" | "timeout",
    "output": "<agent-output>",
    "files_modified": ["file1.py", "file2.py"],
    "duration_ms": 45000,
    "exit_code": 0,
    "error": null
}
```

**Response (streaming via SSE):**
```
event: output
data: {"chunk": "Reading file main.py...\n"}

event: output
data: {"chunk": "Editing function calculate()...\n"}

event: file_modified
data: {"path": "main.py"}

event: completed
data: {"status": "completed", "duration_ms": 45000}
```

#### 3. Get Agent Status

```
GET /api/sandboxes/{sandbox_id}/agent/status
```

**Response:**
```json
{
    "running": true,
    "current_task": "Fixing bug in main.py",
    "steps_completed": 5,
    "files_modified": ["main.py"],
    "started_at": "2026-01-09T15:00:00Z"
}
```

### Supported Agent Configurations

| Agent Type | Model Providers | API Format | Notes |
|------------|-----------------|------------|-------|
| `claude-code` | Anthropic, Chutes | Anthropic Messages | Full tool suite |
| `codex` | OpenAI, Chutes | OpenAI Chat | Fast execution |
| `aider` | Any | Both | Python-based |
| `opencode` | OpenAI-compatible | OpenAI Chat | Terminal-native |

### Implementation Checklist

- [ ] Add `/api/sandboxes` agent configuration parameter
- [ ] Implement agent environment setup in `runtime.py`
- [ ] Add `/api/sandboxes/{id}/agent/run` endpoint
- [ ] Add `/api/sandboxes/{id}/agent/status` endpoint
- [ ] Add SSE streaming for agent output
- [ ] Add file modification tracking
- [ ] Update Sandy documentation
- [ ] Add integration tests

---

## Summary: Implementation Priorities

### Phase 1: Quick Wins (Low Effort, High Impact)

| Item | Effort | Impact | Action |
|------|--------|--------|--------|
| MMLU-Pro 5-shot | Medium | High | Add few-shot examples from official repo |
| Document variance caveats | Low | Medium | Add note to AIME results |
| Document "simplified mode" | Low | Medium | Add indicator to all agentic benchmarks |

### Phase 2: Agentic Benchmarks (High Effort, High Impact)

| Item | Effort | Impact | Action |
|------|--------|--------|--------|
| Terminal-Bench Native Loop | Medium | High | Implement Option 1 in adapter |
| SWE-Bench with Claude Code | Medium | High | Implement Option 2 using pre-installed agent |

### Phase 3: Sandy Agent Feature (High Effort, Strategic)

| Item | Effort | Impact | Action |
|------|--------|--------|--------|
| Sandy Agent API | High | Strategic | Implement Option 3 specification |
| Multi-agent support | High | Strategic | Support Codex, Aider alongside Claude Code |

---

**Report End**
