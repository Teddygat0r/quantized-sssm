# quantized-sssm

Reproducible baseline harness for quantizing Mamba2 SSM models (Phase 1).

## Setup

```bash
# Requires Python 3.11+ and uv
uv sync
source .venv/bin/activate

# Install lm-evaluation-harness (one-time)
uv pip install lm-eval
```

## LM Eval Harness Benchmark

Run HellaSwag, PIQA, ARC-Challenge, and WinoGrande in one shot:

```bash
bash scripts/eval_lm_harness.sh AntonV/mamba2-130m-hf

# Optional knobs
DEVICE=cuda BATCH_SIZE=8 FEWSHOT=0 \
  bash scripts/eval_lm_harness.sh AntonV/mamba2-130m-hf

# Optional extra lm_eval model args (comma-separated)
bash scripts/eval_lm_harness.sh AntonV/mamba2-130m-hf "dtype=bfloat16,revision=main"
```

## Week 1 — Baseline Evaluation Commands

```bash
# Perplexity (WikiText-2 test)
python scripts/eval_ppl.py --config configs/eval.yaml

# Long-context passkey retrieval sweep
python scripts/eval_long_context.py --config configs/longctx.yaml

# Throughput + memory profiling
python scripts/profile.py --config configs/profile.yaml

# Generate plots from a completed run
python scripts/make_plots.py --run_dir runs/<run_id>
```

All results are written to `runs/<run_id>/`:
- `metrics.jsonl` — structured log records
- `config_resolved.yaml` — fully resolved config snapshot
- `plots/*.png` — generated figures

## Week 1 Checklist

- [ ] Fresh env builds with pinned versions
- [ ] GPU detected; dtype path verified
- [ ] Deterministic seeds applied; dropout disabled
- [ ] `eval_ppl.py` runs and outputs stable PPL (small variance across reruns)
- [ ] `eval_long_context.py` produces accuracy vs length curve deterministically
- [ ] `profile.py` outputs stable throughput/memory curves with warmups + sync
- [ ] All results logged to JSONL/CSV with commit hash + hardware metadata
- [ ] `make_plots.py` generates all baseline plots from saved logs
- [ ] README documents exact commands + output locations
