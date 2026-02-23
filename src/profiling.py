from __future__ import annotations

import time
from typing import Any

import torch
from tqdm.auto import tqdm

from src.utils.timing import maybe_cuda_sync


def _synthetic_input_ids(tokenizer: Any, seq_len: int, device: torch.device) -> torch.Tensor:
    text = "The quick brown fox jumps over the lazy dog. " * (seq_len // 4 + 1)
    ids = tokenizer.encode(text, add_special_tokens=False)[:seq_len]
    return torch.tensor([ids], dtype=torch.long, device=device)


@torch.no_grad()
def profile_prefill(
    model: Any,
    input_ids: torch.Tensor,
    *,
    warmup_iters: int,
    measure_iters: int,
    amp_autocast: bool,
    dtype: torch.dtype,
    reset_peak_memory: bool,
) -> dict[str, Any]:
    device = input_ids.device

    for _ in range(warmup_iters):
        with torch.autocast(
            device_type="cuda", dtype=dtype,
            enabled=(amp_autocast and device.type == "cuda"),
        ):
            model(input_ids)
        maybe_cuda_sync(device)

    if reset_peak_memory and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    timings: list[float] = []
    for _ in range(measure_iters):
        maybe_cuda_sync(device)
        t0 = time.perf_counter()
        with torch.autocast(
            device_type="cuda", dtype=dtype,
            enabled=(amp_autocast and device.type == "cuda"),
        ):
            model(input_ids)
        maybe_cuda_sync(device)
        timings.append(time.perf_counter() - t0)

    peak_mem = 0.0
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    seq_len = input_ids.shape[1]
    median_s = sorted(timings)[len(timings) // 2]
    return {
        "prefill_seq_len": seq_len,
        "prefill_median_s": median_s,
        "prefill_tokens_per_s": seq_len / max(median_s, 1e-9),
        "prefill_peak_mem_mb": peak_mem,
        "prefill_timings": timings,
    }


@torch.no_grad()
def profile_decode(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    *,
    decode_new_tokens: int,
    warmup_iters: int,
    measure_iters: int,
    amp_autocast: bool,
    dtype: torch.dtype,
    reset_peak_memory: bool,
) -> dict[str, Any]:
    device = input_ids.device

    gen_kwargs = {
        "do_sample": False,
        "max_new_tokens": decode_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    for _ in range(warmup_iters):
        with torch.autocast(
            device_type="cuda", dtype=dtype,
            enabled=(amp_autocast and device.type == "cuda"),
        ):
            model.generate(input_ids, **gen_kwargs)
        maybe_cuda_sync(device)

    if reset_peak_memory and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    timings: list[float] = []
    for _ in range(measure_iters):
        maybe_cuda_sync(device)
        t0 = time.perf_counter()
        with torch.autocast(
            device_type="cuda", dtype=dtype,
            enabled=(amp_autocast and device.type == "cuda"),
        ):
            model.generate(input_ids, **gen_kwargs)
        maybe_cuda_sync(device)
        timings.append(time.perf_counter() - t0)

    peak_mem = 0.0
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    median_s = sorted(timings)[len(timings) // 2]
    return {
        "decode_prompt_len": int(input_ids.shape[1]),
        "decode_new_tokens": decode_new_tokens,
        "decode_median_s": median_s,
        "decode_tokens_per_s": decode_new_tokens / max(median_s, 1e-9),
        "decode_peak_mem_mb": peak_mem,
        "decode_timings": timings,
    }


def run_profile(
    config: dict[str, Any], tokenizer: Any, model: Any, runtime: dict[str, Any]
) -> list[dict[str, Any]]:
    prof_cfg = config["profile"]
    sweep = prof_cfg["sweep"]
    meas = prof_cfg["measurement"]
    segments = prof_cfg["segments"]
    amp = bool(prof_cfg.get("runtime", {}).get("amp_autocast", True))
    save_raw = bool(prof_cfg.get("output", {}).get("save_raw_timings", False))

    device = runtime["device"]
    dtype = runtime["dtype"]
    seq_lens = [int(x) for x in sweep["seq_lens"]]
    decode_new_tokens = int(sweep["decode_new_tokens"])
    warmup = int(meas["warmup_iters"])
    measure = int(meas["measure_iters"])
    reset_peak = bool(meas.get("reset_peak_memory_each_segment", True))

    results: list[dict[str, Any]] = []

    for seq_len in tqdm(seq_lens, desc="Profile sweep"):
        input_ids = _synthetic_input_ids(tokenizer, seq_len, device)
        row: dict[str, Any] = {"seq_len": seq_len}

        if segments.get("prefill", True):
            prefill = profile_prefill(
                model, input_ids,
                warmup_iters=warmup,
                measure_iters=measure,
                amp_autocast=amp,
                dtype=dtype,
                reset_peak_memory=reset_peak,
            )
            if not save_raw:
                prefill.pop("prefill_timings", None)
            row.update(prefill)

        if segments.get("decode", True):
            decode = profile_decode(
                model, tokenizer, input_ids,
                decode_new_tokens=decode_new_tokens,
                warmup_iters=warmup,
                measure_iters=measure,
                amp_autocast=amp,
                dtype=dtype,
                reset_peak_memory=reset_peak,
            )
            if not save_raw:
                decode.pop("decode_timings", None)
            row.update(decode)

        results.append(row)

    return results
