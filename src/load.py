from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.determinism import set_global_determinism


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested cuda, but torch.cuda.is_available() is False.")
    return torch.device(device_str)


def resolve_dtype(dtype_str: str, device: torch.device) -> torch.dtype:
    if dtype_str not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype={dtype_str}; expected one of {list(DTYPE_MAP)}")
    if dtype_str == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Config requested bf16, but GPU does not report bf16 support.")
    return DTYPE_MAP[dtype_str]


def load_tokenizer_and_model(config: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    run_cfg = config["run"]
    hw_cfg = config["hardware"]
    model_cfg = config["model"]
    tok_cfg = config["tokenizer"]

    seed = int(run_cfg["seed"])
    deterministic = bool(run_cfg.get("deterministic", True))
    set_global_determinism(seed=seed, deterministic=deterministic)

    device = resolve_device(hw_cfg["device"])
    dtype = resolve_dtype(model_cfg["dtype"], device)

    matmul_precision = hw_cfg.get("matmul_precision")
    if matmul_precision:
        torch.set_float32_matmul_precision(matmul_precision)

    tok_name = tok_cfg.get("name_or_path") or model_cfg["name_or_path"]
    tok_revision = tok_cfg.get("revision") or model_cfg.get("revision")
    tokenizer = AutoTokenizer.from_pretrained(
        tok_name,
        revision=tok_revision,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        use_fast=bool(tok_cfg.get("use_fast", True)),
    )
    tokenizer.padding_side = tok_cfg.get("padding_side", "left")
    tokenizer.truncation_side = tok_cfg.get("truncation_side", "left")
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "revision": model_cfg.get("revision"),
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
        "torch_dtype": dtype,
    }
    if model_cfg.get("attn_implementation") is not None:
        model_kwargs["attn_implementation"] = model_cfg["attn_implementation"]

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        **model_kwargs,
    )
    model.to(device)
    model.eval()
    model.config.use_cache = bool(model_cfg.get("use_cache", True))

    if bool(hw_cfg.get("torch_compile", False)):
        model = torch.compile(model, mode=hw_cfg.get("compile_mode", "default"))

    runtime = {
        "seed": seed,
        "device": device,
        "dtype": dtype,
        "generation": dict(config.get("generation", {})),
        "max_length_hint": model_cfg.get("max_length_hint"),
        "model_name": model_cfg.get("name_or_path"),
        "model_revision": model_cfg.get("revision", "main"),
    }
    return tokenizer, model, runtime
