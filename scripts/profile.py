from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Avoid shadowing Python stdlib `profile` when this file is executed directly.
# PyTorch can import `cProfile` -> `profile`; if `scripts/` is first on sys.path,
# Python may import this file again as module `profile` and trigger a circular import.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.load import load_tokenizer_and_model
from src.logging import JsonlLogger, build_run_id, prepare_run_dir, runtime_metadata, write_resolved_config
from src.profiling import run_profile
from src.utils.timing import maybe_cuda_sync


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prefill/decode throughput and memory profiling.")
    parser.add_argument("--config", type=str, required=True, help="Path to profile config yaml.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    run_id = build_run_id()
    run_dir = prepare_run_dir(config["run"]["output_dir"], run_id)
    write_resolved_config(run_dir, config)
    logger = JsonlLogger(run_dir / config["profile"]["output"].get("metrics_file", "metrics.jsonl"))

    tokenizer, model, runtime = load_tokenizer_and_model(config)

    maybe_cuda_sync(runtime["device"])
    t0 = time.perf_counter()
    rows = run_profile(config, tokenizer, model, runtime)
    maybe_cuda_sync(runtime["device"])
    runtime_s = time.perf_counter() - t0

    profile_cfg = config["profile"]
    base = {
        "run_id": run_id,
        "script": "profile.py",
        **runtime_metadata(config["model"], config["hardware"], config.get("generation")),
        "seed": config["run"]["seed"],
        "batch_size": int(profile_cfg["sweep"].get("batch_size", 1)),
        "stride": None,
        "gen_len": int(profile_cfg["sweep"]["decode_new_tokens"]),
    }

    for row in rows:
        record = {
            **base,
            "seq_len": int(row["seq_len"]),
            **row,
        }
        logger.write(record)

    print(f"run_id={run_id}")
    print(f"run_dir={run_dir}")
    print(f"n_lengths={len(rows)}")
    print(f"runtime_s={runtime_s:.2f}")
    if rows and "prefill_tokens_per_s" in rows[0]:
        avg_prefill_tps = sum(float(x["prefill_tokens_per_s"]) for x in rows) / len(rows)
        print(f"avg_prefill_tokens_per_s={avg_prefill_tps:.2f}")
    if rows and "decode_tokens_per_s" in rows[0]:
        avg_decode_tps = sum(float(x["decode_tokens_per_s"]) for x in rows) / len(rows)
        print(f"avg_decode_tokens_per_s={avg_decode_tps:.2f}")


if __name__ == "__main__":
    main()
