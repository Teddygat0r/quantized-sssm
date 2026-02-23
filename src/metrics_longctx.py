from __future__ import annotations

import hashlib
import random
import re
import string
from dataclasses import dataclass
from typing import Any

import torch
from tqdm.auto import tqdm


@dataclass
class TrialResult:
    context_length: int
    trial_index: int
    key: str
    generated_text: str
    predicted_key: str
    correct: bool
    prompt_template_hash: str
    prompt_template_version: str


def _random_key(rng: random.Random, key_length: int, charset: str) -> str:
    if charset == "digits":
        alphabet = string.digits
    elif charset == "hex":
        alphabet = string.hexdigits.lower()[:16]
    elif charset == "alnum":
        alphabet = string.ascii_letters + string.digits
    else:
        raise ValueError(f"Unsupported key_charset={charset}")
    return "".join(rng.choice(alphabet) for _ in range(key_length))


def _filler_token_ids(tokenizer: Any, target_tokens: int, separator: str) -> list[int]:
    seed_text = (
        "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
        "incididunt ut labore et dolore magna aliqua "
    )
    base = ((seed_text + separator) * max(1, target_tokens // 8 + 16)).strip()
    token_ids = tokenizer.encode(base, add_special_tokens=False)
    if len(token_ids) < target_tokens:
        repeats = (target_tokens // max(1, len(token_ids))) + 2
        token_ids = (token_ids * repeats)[:target_tokens]
    return token_ids[:target_tokens]


def _extract_candidate_key(text: str, key_length: int, charset: str) -> str:
    if charset == "digits":
        pattern = rf"\d{{{key_length}}}"
    elif charset == "hex":
        pattern = rf"[0-9a-fA-F]{{{key_length}}}"
    elif charset == "alnum":
        pattern = rf"[0-9a-zA-Z]{{{key_length}}}"
    else:
        return ""
    match = re.search(pattern, text)
    return match.group(0) if match else ""


def _needle_insertion_index(
    rng: random.Random, total_tokens: int, needle_tokens: int, strategy: str
) -> int:
    max_start = max(0, total_tokens - needle_tokens)
    if strategy == "front":
        return 0
    if strategy == "middle":
        return max_start // 2
    if strategy == "back":
        return max_start
    if strategy == "uniform_random":
        return rng.randint(0, max_start)
    raise ValueError(f"Unsupported insertion.position={strategy}")


@torch.no_grad()
def run_long_context_eval(
    config: dict[str, Any], tokenizer: Any, model: Any, runtime: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lc_cfg = config["eval_long_context"]
    sweep_cfg = lc_cfg["sweep"]
    task_cfg = lc_cfg["task"]["passkey"]
    context_cfg = lc_cfg["task"]["context_construction"]
    decoding_cfg = lc_cfg["decoding"]
    run_cfg = config["run"]

    lengths = [int(x) for x in sweep_cfg["context_lengths"]]
    trials_per_length = int(sweep_cfg["trials_per_length"])
    warmup_trials = int(lc_cfg.get("runtime", {}).get("warmup_trials", 0))

    key_length = int(task_cfg["key_length"])
    key_charset = task_cfg["key_charset"]
    needle_prefix = task_cfg["needle_prefix"]
    query = task_cfg["query"]
    separator = context_cfg["synthetic"]["separator"]
    insertion_strategy = context_cfg["insertion"]["position"]
    template_version = context_cfg.get("template_version", "v1")

    template_descriptor = (
        f"version={template_version}|prefix={needle_prefix}|query={query}|sep={separator}"
    )
    template_hash = hashlib.sha256(template_descriptor.encode("utf-8")).hexdigest()[:16]

    rng = random.Random(int(run_cfg["seed"]))
    device = runtime["device"]

    gen_kwargs = {
        "do_sample": False,
        "temperature": float(config["generation"].get("temperature", 0.0)),
        "top_p": float(config["generation"].get("top_p", 1.0)),
        "top_k": int(config["generation"].get("top_k", 0)),
        "num_beams": int(config["generation"].get("num_beams", 1)),
        "max_new_tokens": int(decoding_cfg["max_new_tokens"]),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if gen_kwargs["temperature"] == 0.0:
        # Transformers expects temperature > 0 when sampling is enabled.
        gen_kwargs["temperature"] = 1.0

    detailed: list[dict[str, Any]] = []
    aggregate: list[dict[str, Any]] = []

    for length in lengths:
        correct = 0
        total = 0

        for trial_idx in tqdm(range(trials_per_length), desc=f"longctx L={length}"):
            key = _random_key(rng, key_length=key_length, charset=key_charset)
            needle_text = f"{needle_prefix}{key}"
            needle_ids = tokenizer.encode(needle_text, add_special_tokens=False)

            filler_ids = _filler_token_ids(
                tokenizer=tokenizer,
                target_tokens=max(1, length - len(needle_ids)),
                separator=separator,
            )
            insert_idx = _needle_insertion_index(
                rng=rng,
                total_tokens=len(filler_ids),
                needle_tokens=len(needle_ids),
                strategy=insertion_strategy,
            )
            context_ids = filler_ids[:insert_idx] + needle_ids + filler_ids[insert_idx:]
            context_text = tokenizer.decode(context_ids, skip_special_tokens=True)
            prompt = f"{context_text}\n\nQuestion: {query}\nAnswer:"

            batch = tokenizer(prompt, return_tensors="pt")
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)

            output = model.generate(input_ids=input_ids, attention_mask=attn, **gen_kwargs)
            generated_ids = output[:, input_ids.shape[1] :]
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
            predicted = _extract_candidate_key(generated_text, key_length, key_charset)
            is_correct = predicted == key

            if trial_idx >= warmup_trials:
                total += 1
                correct += int(is_correct)

                result = TrialResult(
                    context_length=length,
                    trial_index=trial_idx,
                    key=key,
                    generated_text=generated_text,
                    predicted_key=predicted,
                    correct=is_correct,
                    prompt_template_hash=template_hash,
                    prompt_template_version=template_version,
                )
                detailed.append(result.__dict__)

        accuracy = (correct / total) if total else 0.0
        aggregate.append(
            {
                "context_length": length,
                "trials": total,
                "correct": correct,
                "accuracy": accuracy,
                "prompt_template_hash": template_hash,
                "prompt_template_version": template_version,
            }
        )

    return aggregate, detailed
