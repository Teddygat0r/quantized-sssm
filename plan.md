# Quantized SSM: Phase 1 Plan

## Phase 1 Objective (2–3 months)

Produce a **reproducible codebase** and **paper-quality baseline results** starting from a strong FP16/BF16 baseline and progressively quantizing SSM-block linear layers down to **W4A4** (4-bit weights + activations) **without custom CUDA kernels**, using an existing backend (prefer **TorchAO**; alternatives allowed if needed). Phase 1 should establish baselines, quantify tradeoffs, and clearly motivate Phase 2 (e.g., state-level distillation, reconstruction, or selective higher-precision retention).

---

## Phase 1 Deliverables

### Clean PTQ pipeline with staged configs

- **FP16/BF16** (baseline) → **W8A8** → **W4A8** → **W4A4**
- Applied primarily to **Mamba2 mixer projections** (explicitly enumerate which Linear modules are included/excluded).

### Failure-mode characterization at ≤4-bit

- **Long-context degradation** (accuracy vs context length).
- **Recurrence/state error accumulation** (drift over time; sensitivity to prompt length; decode stability).
- **Layerwise sensitivity** (which projections/layers break first).

### Profiling vs sequence length

- **Throughput** (prefill + decode tokens/s).
- **Peak memory** (prefill + decode).
- Compare across precision modes with consistent measurement protocol.

---

## Constraints and Principles

- **No custom CUDA kernels**; prioritize TorchAO quantization paths that work with `torch.compile` where possible.
- **Reproducibility as a first-class metric**: pinned versions, deterministic scripts, stable logging schema, and one-command reruns.
- **Minimize confounders**: fixed seeds, fixed data splits, fixed tokenization, fixed generation settings, fixed warmup and timing methodology.

---

## Week 1 Scope: Baseline Harness + Reproducibility (FP16/BF16 only)

### Week 1 “Definition of Done”

From a fresh environment, a short sequence of commands reproduces:

1. **Baseline PPL** on a fixed dataset split
2. **Baseline long-context curve** (passkey/needle retrieval) across a length sweep
3. **Baseline throughput + memory curves** for prefill and decode

…and writes results + plots with a **stable schema** and **minimal rerun variance**.

---

## 1) Repository Structure (recommended)

Create a minimal, paper-friendly layout:

```
configs/
  model.yaml       # model name, dtype, device, compile flags
  eval.yaml        # dataset, seq_len, stride, batch size, seeds
  longctx.yaml     # length sweep, passkey params, #trials
  profile.yaml     # length sweep, warmup, measure iters

src/
  load.py          # tokenizer/model loader, seeds, device placement
  logging.py       # JSONL/CSV writers, schema validation
  metrics_ppl.py
  metrics_longctx.py
  profiling.py
  utils/           # timing helpers, deterministic settings

scripts/
  eval_ppl.py
  eval_long_context.py
  profile.py
  make_plots.py

runs/              # outputs

README.md
```

---

## 2) Environment + Version Pinning

### Requirements

- **Python 3.11**
- Pin exact versions for:
  - `torch`, `transformers`, `datasets`, `accelerate` (if used), `numpy`, `tqdm`, `pandas`, `matplotlib`
  - (later) `torchao`

### Record

- `pip freeze` snapshot (or lockfile)
- GPU model + driver/CUDA info
- **Commit hash for every run**

### Runtime checks

- `torch.cuda.is_available()` must pass
- Confirm BF16 support if using BF16 (otherwise FP16)

---

## 3) Model Choice and Loading (baseline)

### Model target

- A **Transformers-compatible Mamba2 checkpoint** (small first, e.g., ~130M “*-hf” converted) to keep iteration fast.

### Loader requirements (`src/load.py`)

- Instantiate tokenizer + model in **BF16 or FP16**
- `model.eval()`, disable dropout, and set deterministic seeds:
  - `torch.manual_seed`, `torch.cuda.manual_seed_all`, `numpy.random.seed`, `random.seed`
- Provide two verified codepaths:
  1. **Prefill forward pass** on a full prompt
  2. **Generate** with fixed decoding settings (greedy or fixed temperature=0) to ensure determinism
- Record in logs:
  - model name / revision
  - dtype
  - max context length used
  - generation settings (if any)

---

## 4) Evaluation Scripts (baseline only)

All scripts must share **consistent CLI arguments** and **output schema**.

### A) `eval_ppl.py`

- **Goal:** simple, stable perplexity.
- Use a standard dataset split (e.g., WikiText-2 test or similar).
- Fixed `seq_len` + stride sliding window.
- **Output:** `ppl`, `n_tokens`, `runtime_s`, `tokens_per_s`
- Must be deterministic and rerunnable.

### B) `eval_long_context.py`

- **Goal:** deterministic needle/passkey retrieval stress test.
- Generate synthetic contexts with controlled placement of a “needle” string and a query.
- Sweep context lengths (example: 2k, 4k, 8k, 16k, 32k as feasible).
- **Output:** accuracy vs `context_length`, trial count, exact prompt template hash/version
- Keep generation deterministic (greedy).

### C) `profile.py`

- **Goal:** throughput and peak memory for prefill + decode.
- Measure both:
  - **Prefill** (one forward on length L)
  - **Decode** (generate N tokens with fixed KV/state behavior)
- **Protocol:**
  - warmup iterations
  - `torch.cuda.synchronize()` before/after timing
  - fixed batch size
  - record peak memory via `torch.cuda.max_memory_allocated()` (reset between segments)
- **Output:**
  - `prefill_tokens_per_s`, `decode_tokens_per_s`
  - `prefill_peak_mem_mb`, `decode_peak_mem_mb`

---

## 5) Logging + Artifacts (standard schema)

- Standardize outputs as **JSONL** (preferred) and/or **CSV**.
- Each record should include:
  - `run_id` (timestamp + short hash)
  - `git_commit`
  - `host` (optional), `gpu_name`, `driver`/`cuda`, `torch_version`
  - `model_name`, `model_revision`
  - `dtype`, `device`, `batch_size`, `seq_len`, `stride`, `gen_len`
  - metrics fields (script-specific)
  - `seed`
  - `notes` (optional)

### Write to

- `runs/<run_id>/metrics.jsonl`
- `runs/<run_id>/config_resolved.yaml`
- `runs/<run_id>/plots/*.png`

---

## 6) `make_plots.py` (baseline plots)

Generate:

- **PPL** (bar/table) for BF16 vs FP16 (if both run)
- **Passkey accuracy vs context length** (line plot)
- **Throughput vs length** + **memory vs length** (separate plots for prefill/decode)

Make plot generation **fully offline** from saved JSONL/CSV.

---

## 7) Week 1 Command Interface

Provide a minimal set of commands in `README.md`, e.g.:

```bash
python scripts/eval_ppl.py --config configs/eval.yaml
python scripts/eval_long_context.py --config configs/longctx.yaml
python scripts/profile.py --config configs/profile.yaml
python scripts/make_plots.py --run_dir runs/<run_id>
```

Or a single bash **`scripts/run_week1.sh`** that executes all three + plots.

---

## Week 1 Checklist (copy into README)

- [ ] Fresh env builds with pinned versions
- [ ] GPU detected; dtype path verified
- [ ] Deterministic seeds applied; dropout disabled
- [ ] `eval_ppl.py` runs and outputs stable PPL (small variance across reruns)
- [ ] `eval_long_context.py` produces accuracy vs length curve deterministically
- [ ] `profile.py` outputs stable throughput/memory curves with warmups + sync
- [ ] All results logged to JSONL/CSV with commit hash + hardware metadata
- [ ] `make_plots.py` generates all baseline plots from saved logs
- [ ] README documents exact commands + output locations

---

## Phase 1 Continuity Hooks (so Week 2 starts cleanly)

While Week 1 is baseline-only, design your code so **Week 2 PTQ slots in with minimal refactor**:

1. **Centralize module selection** (“which Linear layers are in scope”) behind a single **registry function**.
2. **Keep eval/profile scripts model-agnostic:** they load “a model factory” and run metrics.
3. **Ensure logging schema already has fields reserved for:**
   - `w_bits`, `a_bits`, `group_size`, `per_channel`, `quant_backend`, `calib_method`

---

*This document is the operational spec for Week 1; any implementation decisions should preserve determinism, logging consistency, and minimal-diff extensibility into W8A8/W4A8/W4A4 in Week 2+.*
