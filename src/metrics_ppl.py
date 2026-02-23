from __future__ import annotations

import math
from typing import Any

import torch
from datasets import load_dataset
from tqdm.auto import tqdm

@torch.no_grad()
def run_ppl_eval(config: dict[str, Any], tokenizer: Any, model: Any, runtime: dict[str, Any]) -> dict[str, Any]:
    eval_cfg = config["eval_ppl"]
    data_cfg = eval_cfg["dataset"]
    prep_cfg = eval_cfg["preprocessing"]
    rt_cfg = eval_cfg["runtime"]

    dataset = load_dataset(
        data_cfg["name"],
        data_cfg.get("config"),
        split=data_cfg["split"],
        streaming=bool(data_cfg.get("streaming", False)),
        cache_dir=data_cfg.get("cache_dir"),
    )

    max_examples = rt_cfg.get("max_examples")
    text_field = data_cfg.get("text_field", "text")
    texts: list[str] = []
    for idx, row in enumerate(dataset):
        if max_examples is not None and idx >= int(max_examples):
            break
        value = row.get(text_field, "")
        if isinstance(value, str) and value.strip():
            texts.append(value)

    if not texts:
        raise RuntimeError("No usable text examples found for PPL evaluation.")

    text = "\n\n".join(texts)
    add_bos = bool(prep_cfg.get("add_bos", False))
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=add_bos)
    input_ids = enc["input_ids"].to(runtime["device"])

    seq_len = int(prep_cfg["seq_len"])
    stride = int(prep_cfg["stride"])
    drop_last = bool(prep_cfg.get("drop_last", True))
    amp_autocast = bool(rt_cfg.get("amp_autocast", True))
    warmup_steps = int(rt_cfg.get("warmup_steps", 0))

    nll_sum = torch.tensor(0.0, device=runtime["device"])
    n_tokens = 0
    prev_end = 0

    # optional warmup to stabilize early-kernel timings
    for _ in range(max(0, warmup_steps)):
        sample = input_ids[:, : min(seq_len, input_ids.size(1))]
        with torch.autocast(
            device_type="cuda",
            dtype=runtime["dtype"],
            enabled=(amp_autocast and runtime["device"].type == "cuda"),
        ):
            _ = model(sample)

    total_len = input_ids.size(1)
    windows = list(range(0, total_len, stride))
    if drop_last:
        windows = [b for b in windows if b + seq_len <= total_len]
    for begin in tqdm(windows, desc="PPL windows"):
        end = min(begin + seq_len, total_len)
        target_len = end - prev_end
        if target_len <= 0:
            break

        window = input_ids[:, begin:end]
        labels = window.clone()
        labels[:, :-target_len] = -100

        with torch.autocast(
            device_type="cuda",
            dtype=runtime["dtype"],
            enabled=(amp_autocast and runtime["device"].type == "cuda"),
        ):
            out = model(window, labels=labels)
            nll = out.loss.float() * target_len

        nll_sum += nll
        n_tokens += int(target_len)
        prev_end = end
        if end == input_ids.size(1):
            break

    if n_tokens == 0:
        raise RuntimeError("No tokens were evaluated for perplexity.")

    ppl = torch.exp(nll_sum / n_tokens).item()
    return {
        "ppl": float(ppl),
        "n_tokens": int(n_tokens),
        "n_examples": len(texts),
        "seq_len": seq_len,
        "stride": stride,
        "token_count_raw": int(input_ids.size(1)),
        "overflow_guard": bool(math.isfinite(ppl)),
    }
