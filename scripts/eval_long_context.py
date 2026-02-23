from __future__ import annotations

import argparse
import statistics
import time

from src.config import load_config
from src.load import load_tokenizer_and_model
from src.logging import JsonlLogger, build_run_id, prepare_run_dir, runtime_metadata, write_resolved_config
from src.metrics_longctx import run_long_context_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run long-context passkey retrieval eval.")
    parser.add_argument("--config", type=str, required=True, help="Path to long-context yaml.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    run_id = build_run_id()
    run_dir = prepare_run_dir(config["run"]["output_dir"], run_id)
    write_resolved_config(run_dir, config)
    logger = JsonlLogger(run_dir / "metrics.jsonl")

    tokenizer, model, runtime = load_tokenizer_and_model(config)

    t0 = time.perf_counter()
    aggregate_rows, detail_rows = run_long_context_eval(config, tokenizer, model, runtime)
    runtime_s = time.perf_counter() - t0

    base = {
        "run_id": run_id,
        "script": "eval_long_context.py",
        **runtime_metadata(config["model"], config["hardware"], config.get("generation")),
        "seed": config["run"]["seed"],
        "batch_size": 1,
        "stride": None,
        "gen_len": config["eval_long_context"]["decoding"]["max_new_tokens"],
    }
    for row in aggregate_rows:
        record = {
            **base,
            "record_type": "aggregate",
            "seq_len": row["context_length"],
            **row,
        }
        logger.write(record)

    if config["eval_long_context"]["output"].get("save_generations", True):
        for row in detail_rows:
            record = {
                **base,
                "record_type": "trial",
                "seq_len": row["context_length"],
                **row,
            }
            logger.write(record)

    accs = [x["accuracy"] for x in aggregate_rows]
    mean_acc = statistics.mean(accs) if accs else 0.0

    print(f"run_id={run_id}")
    print(f"run_dir={run_dir}")
    print(f"n_lengths={len(aggregate_rows)}")
    print(f"mean_accuracy={mean_acc:.4f}")
    print(f"runtime_s={runtime_s:.2f}")


if __name__ == "__main__":
    main()
