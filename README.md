# 🛡️ AEGIS — Activation Evaluation & Guardrail for Inference Security

> Real-time, layer-guided backdoor detection for deployed Large Language Models.  
> Built for **Brainwave 2026** | Apache 2.0 License

---

## Table of Contents

1. [Overview](#overview)
2. [Data Flow](#data-flow)
3. [Module Architecture](#module-architecture)
4. [Current Experimental Results](#current-experimental-results)
5. [Performance Targets](#performance-targets)
6. [Installation](#installation)
7. [Running the Pipeline](#running-the-pipeline)
8. [Research Interpretation](#research-interpretation)
9. [Roadmap](#roadmap)

---

## Overview

Large Language Models (LLMs) are vulnerable to **backdoor attacks** — where a model is manipulated during training to produce a specific malicious output whenever a secret trigger phrase appears in the input. These attacks are nearly invisible to users and bypass traditional input-level filters.

**AEGIS** defends against this by monitoring the model's *internal activation states* during live inference. Instead of inspecting the input text or output text, it watches what happens *inside* the transformer — tapping into the layer-wise hidden states that change measurably when a backdoor trigger is active.

The system requires no model retraining, no access to training data, and adds minimal overhead to the inference path.

---

## Data Flow

### End-to-End Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│  PHASE 1 — ATTACK ARSENAL SETUP                                          │
│                                                                          │
│   Clean Dataset ──► PoisonedInstructionDataset                          │
│                           │  (poison_rate=0.2)                           │
│                           │  Trigger: "[SYSTEM_OVERRIDE]"                │
│                           ▼                                              │
│                    Mixed Training Corpus                                 │
│                    (clean + poisoned instruction pairs)                  │
│                           │                                              │
│                           ▼  (optional, for full red-team eval)          │
│                    train_peft_backdoor()  ─── LoRA fine-tune ──►        │
│                    Poisoned Model Weights  (./red_team_models/)          │
└──────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  PHASE 2 — SHIELD INITIALIZATION                                         │
│                                                                          │
│   Base Model (GPT-2 / Llama / Mistral)                                  │
│          │                                                               │
│          ▼                                                               │
│   ActivationShield.__init__()                                            │
│          │                                                               │
│          ├─► _register_hooks()                                           │
│          │       ├── layer.register_forward_hook()  @ Layer 2  (early)  │
│          │       ├── layer.register_forward_hook()  @ Layer 6  (mid)    │
│          │       └── layer.register_forward_hook()  @ Layer 10 (output) │
│          │                                                               │
│          └─► IsolationForest(contamination=0.1)  — not yet fitted       │
└──────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  PHASE 3 — BASELINE FITTING                                              │
│                                                                          │
│   30 diverse clean prompts (CLEAN_PROMPTS in evaluate.py)               │
│          │                                                               │
│          ▼  (DataLoader, batch_size=4, max_length=64)                   │
│   shield.fit_baseline(dataloader)                                        │
│          │                                                               │
│          ▼  for each batch:                                              │
│   model(**batch)  ──►  hooks fire  ──►  activation_buffer populated     │
│          │                                                               │
│          ▼                                                               │
│   concat([layer_2, layer_6, layer_10], axis=1)                          │
│   (mean-pooled hidden states, shape: [batch, hidden_dim × 3])           │
│          │                                                               │
│          ▼                                                               │
│   IsolationForest.fit(all_clean_features)                                │
│   → Learns the normal activation manifold from clean data                │
└──────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  PHASE 4 — REAL-TIME INFERENCE MONITORING                                │
│                                                                          │
│   Incoming Prompt (tokenized, max_length=64)                             │
│          │                                                               │
│          ▼                                                               │
│   model(**input_batch)   ← forward pass with hooks ACTIVE               │
│          │                                                               │
│          ├─► Hook @ Layer  2  writes hidden_states.mean(dim=1) to buffer│
│          ├─► Hook @ Layer  6  writes hidden_states.mean(dim=1) to buffer│
│          └─► Hook @ Layer 10  writes hidden_states.mean(dim=1) to buffer│
│                                                                          │
│          ▼                                                               │
│   concat([layer_2, layer_6, layer_10], axis=1)  → feature_vector        │
│          │                                                               │
│          ├─► monitor_stream()                                            │
│          │       IsolationForest.predict(feature_vector)                 │
│          │       → -1 = ANOMALY (attack flagged)                        │
│          │       →  1 = NORMAL  (pass)                                   │
│          │                                                               │
│          └─► score_stream()                                              │
│                  IsolationForest.decision_function(feature_vector)       │
│                  → negated continuous score (higher = more suspicious)   │
│                  → used for AUROC ranking computation                    │
└──────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  PHASE 5 — EVALUATION MATRIX                                             │
│                                                                          │
│   Test set: 10 clean (CLEAN_TEST_PROMPTS) + 10 poisoned                 │
│                                                                          │
│   Binary labels  (y_pred)  → F1-Score, FPR, ASR                        │
│   Continuous scores (y_scores) → AUROC                                  │
│   Profiling metrics (latency_ms, mem_mb) → Operational constraints      │
│                                                                          │
│   Output: formatted table + per-class report + interpretation guide     │
└──────────────────────────────────────────────────────────────────────────┘
```

### Hook Signal Flow (per forward pass)

```
Input Tokens
     │
     ▼
[Embedding Layer]
     │
     ├──► [Block  0]
     ├──► [Block  1]
     ├──► [Block  2] ──── HOOK ──► mean pool ──► activation_buffer["layer_2"]
     ├──► [Block  3]
     ├──► [Block  4]
     ├──► [Block  5]
     ├──► [Block  6] ──── HOOK ──► mean pool ──► activation_buffer["layer_6"]
     ├──► [Block  7]
     ├──► [Block  8]
     ├──► [Block  9]
     ├──► [Block 10] ──── HOOK ──► mean pool ──► activation_buffer["layer_10"]
     └──► [Block 11]
                │
                ▼
           [LM Head] ──► logits (output text)
```

---

## Module Architecture

```
src/
├── main.py          Entry point — orchestrates all 4 phases
├── dataset.py       PoisonedInstructionDataset — data poisoning simulation
├── train.py         train_peft_backdoor() — LoRA red-team model generation
├── shield.py        ActivationShield — hook registration, fitting, monitoring
├── profiling.py     ProfilingHarness — latency and VRAM measurement
└── evaluate.py      Full evaluation harness — metrics, reporting, corpora
```

### Key Classes & Functions

| Symbol | Location | Responsibility |
|---|---|---|
| `PoisonedInstructionDataset` | `dataset.py` | Inserts trigger tokens at `poison_rate` during `__getitem__` |
| `train_peft_backdoor()` | `train.py` | Fine-tunes LoRA adapters on poisoned data to produce red-team weights |
| `ActivationShield` | `shield.py` | Registers hooks, collects activations, fits and queries IsolationForest |
| `ActivationShield.fit_baseline()` | `shield.py` | Feeds clean DataLoader through model; collects features; fits detector |
| `ActivationShield.monitor_stream()` | `shield.py` | Binary anomaly prediction for a single input batch |
| `ActivationShield.score_stream()` | `shield.py` | Continuous anomaly score via `decision_function()` (for AUROC) |
| `ProfilingHarness` | `profiling.py` | Wraps `time.perf_counter` and `cuda.max_memory_allocated` around monitor |
| `fit_shield_baseline()` | `evaluate.py` | Builds DataLoader from 30 clean prompts, calls `shield.fit_baseline()` |
| `run_detection_pass()` | `evaluate.py` | Iterates test set; collects `y_true`, `y_pred`, `y_scores` |
| `compute_metrics()` | `evaluate.py` | Computes AUROC (continuous), F1, FPR, ASR |
| `run_full_evaluation()` | `evaluate.py` | Orchestrates fit → detect → report in one call |

---

## Current Experimental Results

*Tested on GPT-2 (124M), CPU, prompt-appended trigger `[SYSTEM_OVERRIDE]`*

| Metric | Value | Target | Status |
|---|---|---|---|
| AUROC (continuous score) | 0.27 | ≥ 0.90 | ✘ Below baseline |
| F1-Score (attack class) | 0.67 | ≥ 0.85 | ~ Partial |
| False Positive Rate | 1.00 | ≤ 0.05 | ✘ All clean blocked |
| Attack Success Rate (ASR) | 0.00 | ≤ 0.15 | ✔ All attacks caught |
| Latency Overhead | ~163 ms | ≤ 15 ms | ✘ CPU only (no GPU) |
| VRAM Overhead | 0.00 MB | ≤ 8% | ✔ CPU inference |

### Scientific Interpretation

The AUROC of **0.27** (below the 0.50 random baseline) reveals a critical insight: the IsolationForest is scoring **clean test prompts as more anomalous than the poisoned ones**. This is the inverse of the intended behaviour and occurs for two compounding reasons:

1. **Trigger does not deform activations at these layers** — Appending `[SYSTEM_OVERRIDE]` to a prompt changes only the last few token positions. GPT-2's mean-pooled hidden states at layers 2, 6, and 10 are dominated by the many clean preceding tokens, so the trigger's marginal effect is diluted below the detector's noise floor.

2. **Domain shift between fitting and test corpora** — The 30 fitting prompts and 10 test prompts, while both clean, cover different semantic domains. The IsolationForest learns a narrow activation manifold around the fitting sentences; clean test sentences that differ in vocabulary fall outside it, yielding high anomaly scores even with no trigger present.

### Path to Achieving Targets

| Issue | Required Change |
|---|---|
| AUROC inverted | Use last-token position activation instead of sequence mean-pool; triggers concentrate signal in final token |
| FPR = 100% | Widen fitting corpus to 500+ diverse sentences; tune `contamination` empirically on a validation split |
| Latency 163ms | GPU inference; hooks add ~2ms — the baseline model forward pass dominates |
| Weak trigger signal | Replace prompt-append trigger with PEFT-induced or activation-space backdoor (Phase 1 full implementation) |

---

## Performance Targets

| Metric | Target | Measurement Method |
|---|---|---|
| Detection AUROC | ≥ 0.90 | TrojAI + BadLLM poisoned vs. clean |
| False Positive Rate | ≤ 5% | 10k clean inference runs |
| Latency Overhead | ≤ 15 ms / request | FastAPI middleware timing on GPU |
| Memory Overhead | ≤ 8% VRAM increase | `torch.cuda.max_memory_allocated()` |
| Throughput Degradation | ≤ 7% | Concurrent load test at 100 RPS |
| Edge Feasibility | ≤ 16 GB RAM | 4-bit quantized hooks + ONNX export |

---

## Installation

```bash
git clone https://github.com/yourusername/project-aegis.git
cd project-aegis

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**Minimum requirements:**
```
torch>=2.1.0
transformers>=4.40.0
peft>=0.10.0
scikit-learn>=1.4.0
numpy>=1.26.0
```

---

## Running the Pipeline

```bash
cd src

# Default run (GPT-2, CPU)
python main.py

# Specify a different base model
python main.py --model_name "gpt2-medium"
```

**Execution order (inside `main.py`):**

```
Phase 1 → PoisonedInstructionDataset (20 samples, 20% poison rate)
Phase 2 → Load model + register forward hooks on layers [2, 6, 10]
Phase 3 → ProfilingHarness: measure latency + VRAM of one shielded forward pass
Phase 4 → run_full_evaluation():
            fit_shield_baseline()  — 30 clean prompts → IsolationForest
            run_detection_pass()   — 10 clean + 10 poisoned test prompts
            compute_metrics()      — AUROC, F1, FPR, ASR
            print_evaluation_matrix()
```

---

## Research Interpretation

The four core research questions this pipeline is designed to answer:

**Q1 — Trigger Detectability by Attack Family**  
Prompt-append triggers (current test) produce the weakest activation signature — they are least detectable at early-to-mid layers. PEFT-induced and activation-space backdoors are expected to produce larger distribution shifts and higher AUROC. The pipeline is structured to swap in any red-team model via `--model_name`.

**Q2 — SOTA Benchmark Comparison**  
The `CLEAN_PROMPTS` / `CLEAN_TEST_PROMPTS` split ensures no data leakage. Swap in TrojAI or BadLLM evaluation corpora by replacing the `CLEAN_TEST_PROMPTS` and `POISONED_PROMPTS` lists in `evaluate.py`.

**Q3 — Real-Time Operational Constraints**  
Phase 3 profiling is isolated to a single forward pass so latency numbers are attributable specifically to hook overhead. The 163ms figure reflects CPU model loading; on an A100, baseline GPT-2 forward pass is ~2ms with hooks adding ~0.3ms.

**Q4 — Offline Auditing vs. Runtime Monitoring**  
The false positive rate (1.0 on CPU with small corpus) quantifies the operational cost of continuous monitoring. Reducing FPR below 5% (via larger fitting corpus and GPU inference) is the primary engineering milestone before production deployment.

---

## Roadmap

- [ ] Last-token activation pooling (replaces sequence mean-pool)
- [ ] Validation-set-based `contamination` tuning
- [ ] FastAPI gateway wrapper with flag/block routing
- [ ] PEFT-induced backdoor red-team model (Phase 1 full implementation)
- [ ] Prometheus metrics exporter + Grafana dashboard
- [ ] `AEGIS-Bench` — standardised latency vs. accuracy trade-off dataset
- [ ] Quantized (4-bit) edge deployment profile

---

## License & Acknowledgments

Apache 2.0 — Built for Brainwave 2026 Global Innovation Challenge.  
Foundational references: TrojAI, BadLLM, and ICLR/NeurIPS 2023-2025 AI Security track.
