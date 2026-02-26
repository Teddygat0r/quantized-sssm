from __future__ import annotations

import argparse
import time

from src.config import load_config
from src.load import load_tokenizer_and_model
from src.logging import JsonlLogger, build_run_id, prepare_run_dir, runtime_metadata, write_resolved_config
from src.metrics_ppl import run_ppl_eval
from src.utils.timing import maybe_cuda_sync


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline perplexity evaluation.")
    parser.add_argument("--config", type=str, required=True, help="Path to eval config yaml.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    run_id = build_run_id()
    run_dir = prepare_run_dir(config["run"]["output_dir"], run_id)
    write_resolved_config(run_dir, config)
    logger = JsonlLogger(run_dir / "metrics.jsonl")

    tokenizer, model, runtime = load_tokenizer_and_model(config)
    maybe_cuda_sync(runtime["device"])
    t0 = time.perf_counter()
    ppl_metrics = run_ppl_eval(config, tokenizer, model, runtime)
    maybe_cuda_sync(runtime["device"])
    runtime_s = time.perf_counter() - t0
    tokens_per_s = ppl_metrics["n_tokens"] / max(runtime_s, 1e-9)

    record = {
        "run_id": run_id,
        "script": "eval_ppl.py",
        **runtime_metadata(config["model"], config["hardware"], config.get("generation")),
        "seed": config["run"]["seed"],
        "batch_size": config["eval_ppl"]["dataloader"]["batch_size"],
        "seq_len": config["eval_ppl"]["preprocessing"]["seq_len"],
        "stride": config["eval_ppl"]["preprocessing"]["stride"],
        "gen_len": 0,
        "runtime_s": runtime_s,
        "tokens_per_s": tokens_per_s,
        **ppl_metrics,
    }
    logger.write(record)

    print(f"run_id={run_id}")
    print(f"run_dir={run_dir}")
    print(f"ppl={ppl_metrics['ppl']:.4f}")
    print(f"n_tokens={ppl_metrics['n_tokens']}")
    print(f"runtime_s={runtime_s:.2f}")
    print(f"tokens_per_s={tokens_per_s:.2f}")


if __name__ == "__main__":
    main()
