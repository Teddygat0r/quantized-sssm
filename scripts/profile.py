from __future__ import annotations

import argparse

from src.config import load_config
from src.load import load_tokenizer_and_model
from src.logging import (
    JsonlLogger,
    build_run_id,
    prepare_run_dir,
    runtime_metadata,
    write_resolved_config,
)
from src.profiling import run_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile throughput and peak memory.")
    parser.add_argument("--config", type=str, required=True, help="Path to profile yaml.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    run_id = build_run_id()
    run_dir = prepare_run_dir(config["run"]["output_dir"], run_id)
    write_resolved_config(run_dir, config)
    logger = JsonlLogger(run_dir / "metrics.jsonl")

    tokenizer, model, runtime = load_tokenizer_and_model(config)
    rows = run_profile(config, tokenizer, model, runtime)

    base = {
        "run_id": run_id,
        "script": "profile.py",
        **runtime_metadata(config["model"], config["hardware"], config.get("generation")),
        "seed": config["run"]["seed"],
        "batch_size": config["profile"]["sweep"]["batch_size"],
        "stride": None,
        "gen_len": config["profile"]["sweep"]["decode_new_tokens"],
    }

    for row in rows:
        record = {**base, **row}
        logger.write(record)

    print(f"run_id={run_id}")
    print(f"run_dir={run_dir}")
    print(f"n_seq_lens={len(rows)}")
    for r in rows:
        parts = [f"L={r['seq_len']}"]
        if "prefill_tokens_per_s" in r:
            parts.append(f"prefill={r['prefill_tokens_per_s']:.0f} tok/s")
        if "decode_tokens_per_s" in r:
            parts.append(f"decode={r['decode_tokens_per_s']:.0f} tok/s")
        if "prefill_peak_mem_mb" in r:
            parts.append(f"prefill_mem={r['prefill_peak_mem_mb']:.0f}MB")
        if "decode_peak_mem_mb" in r:
            parts.append(f"decode_mem={r['decode_peak_mem_mb']:.0f}MB")
        print("  " + " | ".join(parts))


if __name__ == "__main__":
    main()
