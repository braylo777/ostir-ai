"""Real-model evaluation for E5: WikiText-2 perplexity, MMLU, GSM8K.

Imported lazily by experiments/e5_accuracy.py --model, so the synthetic path
never needs torch installed.
"""

from __future__ import annotations

import re

import numpy as np

WINDOW = 2048


def _load(name: str, *args, **kw):
    from datasets import load_dataset

    return load_dataset(name, *args, **kw)


def wikitext2_perplexity(
    model, tok, device: str = "cpu", limit: int = 200, window: int = WINDOW
) -> float:
    """Sliding-window perplexity over wikitext-2-raw-v1 test.

    Non-overlapping windows with full-context loss, the standard protocol.
    `limit` caps the number of windows so a sweep of five configurations is
    tractable on CPU; report it alongside the number, since PPL is only
    comparable at equal window counts.
    """
    import torch

    ds = _load("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids
    n = min(limit, ids.size(1) // window) if limit else ids.size(1) // window

    nll, count = 0.0, 0
    with torch.no_grad():
        for i in range(n):
            chunk = ids[:, i * window : (i + 1) * window].to(device)
            out = model(chunk, labels=chunk)
            nll += float(out.loss) * (chunk.size(1) - 1)
            count += chunk.size(1) - 1
    return float(np.exp(nll / count)) if count else float("nan")


def _loglik(model, tok, device, context: str, choice: str) -> float:
    """Length-normalized log-likelihood of `choice` continuing `context`."""
    import torch

    ctx = tok(context, return_tensors="pt").input_ids
    full = tok(context + choice, return_tensors="pt").input_ids.to(device)
    n_ctx = ctx.size(1)
    with torch.no_grad():
        logits = model(full).logits[0, :-1].log_softmax(-1)
    tgt = full[0, 1:]
    sel = logits[torch.arange(len(tgt)), tgt][n_ctx - 1 :]
    return float(sel.sum() / max(1, len(sel)))


def eval_mmlu(model, tok, device: str = "cpu", limit: int = 200) -> float:
    """MMLU accuracy, zero-shot, by log-likelihood over the four options."""
    ds = _load("cais/mmlu", "all", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    correct = 0
    for ex in ds:
        prompt = (
            f"{ex['question'].strip()}\n"
            + "\n".join(f"{c}. {t}" for c, t in zip("ABCD", ex["choices"]))
            + "\nAnswer:"
        )
        scores = [_loglik(model, tok, device, prompt, f" {c}") for c in "ABCD"]
        correct += int(np.argmax(scores) == ex["answer"])
    return correct / len(ds) if len(ds) else float("nan")


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def eval_gsm8k(
    model, tok, device: str = "cpu", limit: int = 200, max_new: int = 256
) -> float:
    """GSM8K exact-match on the final number, 8-shot-free greedy decode."""
    import torch

    ds = _load("gsm8k", "main", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    correct = 0
    for ex in ds:
        prompt = (
            f"Question: {ex['question'].strip()}\n" f"Answer: Let's think step by step."
        )
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0, ids.size(1) :], skip_special_tokens=True)
        gold = _NUM.findall(ex["answer"].split("####")[-1])
        got = _NUM.findall(text)
        if gold and got:
            correct += int(got[-1].replace(",", "") == gold[-1].replace(",", ""))
    return correct / len(ds) if len(ds) else float("nan")
