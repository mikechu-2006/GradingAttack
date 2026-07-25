"""Evaluate transferability of a GCG suffix bank on ASAG samples.

DEPRECATED: Use pipeline.py with method=gcg_suffix_bank and pipeline_mode=true
instead. See configs/GCG-SuffixBank-Llama-3.1-8B-Instruct-3c.yaml for an
example config.

This script is kept for backward compatibility and quick ad-hoc evaluation.
It will be removed in a future release.

This script does not optimize new suffixes. It loads previously discovered
suffixes, tries them on heldout samples, and stops at the first suffix that
flips a non-correct sample to "correct".
"""

from __future__ import annotations
import warnings

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.data_utils import extract_grade, read_student_qa_data_from_jsonl


DEFAULT_MODEL_ID = "LLM-Research/Meta-Llama-3.1-8B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate transfer attack using a bank of GCG suffixes."
    )
    parser.add_argument("--bank", required=True, help="Suffix bank: txt/json/jsonl")
    parser.add_argument("--data", default="dataset/scientsbank.jsonl")
    parser.add_argument("--template", default="configs/grading_template_ci_2c.txt")
    parser.add_argument("--nclass", type=int, choices=[2, 3], default=2)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--bank-limit", type=int, default=None)
    parser.add_argument("--output", default="result/gcg_suffix_bank_transfer.jsonl")
    parser.add_argument(
        "--include-bank-source-indices",
        action="store_true",
        help="Do not exclude source_index values found in the suffix bank.",
    )
    parser.add_argument(
        "--append-eos-tag",
        action="store_true",
        help="Insert suffix before </student_answer> instead of appending to prompt.",
    )
    return parser.parse_args()


def load_suffix_bank(path: str, limit: int | None = None) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    suffixes: list[str] = []
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("suffixes") or data.get("bank") or []
        else:
            items = []
        for item in items:
            suffix = _suffix_from_item(item)
            if suffix:
                suffixes.append(suffix)
    elif p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                suffix = _suffix_from_item(json.loads(line))
                if suffix:
                    suffixes.append(suffix)
    else:
        suffixes = [line.rstrip("\n") for line in p.read_text(encoding="utf-8").splitlines()]
        suffixes = [s for s in suffixes if s.strip()]

    deduped = list(dict.fromkeys(suffixes))
    return deduped[:limit] if limit else deduped


def load_bank_source_indices(path: str) -> set[int]:
    p = Path(path)
    indices: set[int] = set()
    if not p.exists() or p.suffix.lower() not in {".json", ".jsonl"}:
        return indices

    def add_from_item(item: Any) -> None:
        if isinstance(item, dict) and isinstance(item.get("source_index"), int):
            indices.add(item["source_index"])

    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                add_from_item(item)
        elif isinstance(data, dict):
            for item in data.get("suffixes") or data.get("bank") or []:
                add_from_item(item)
            add_from_item(data)
    else:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    add_from_item(json.loads(line))
    return indices


def _suffix_from_item(item: Any) -> str | None:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return None
    for key in ("suffix", "best_string", "gcg_suffix"):
        if item.get(key):
            return str(item[key])
    meta = item.get("meta")
    if isinstance(meta, dict):
        for key in ("suffix", "best_string", "gcg_suffix"):
            if meta.get(key):
                return str(meta[key])
    return None


def resolve_model_path(model_path: str | None, model_id: str) -> str:
    if model_path:
        return model_path
    from modelscope import snapshot_download

    cache_dir = Path.home() / ".cache" / "modelscope" / "hub"
    return snapshot_download(model_id, cache_dir=str(cache_dir))


def label_to_binary(label: str) -> str:
    return "correct" if label == "correct" else "incorrect"


def parsed_grade(response: str, nclass: int) -> str | None:
    grade = extract_grade(response)
    if grade in ("0", "correct"):
        return "correct"
    if nclass == 2:
        if grade in ("1", "2", "incorrect", "contradictory"):
            return "incorrect"
    elif grade in ("1", "contradictory"):
        return "contradictory"
    elif grade in ("2", "incorrect"):
        return "incorrect"
    return None


def grade_rank(grade: str | None, nclass: int) -> int | None:
    if grade == "correct":
        return 0
    if nclass == 2 and grade == "incorrect":
        return 1
    if nclass == 3 and grade == "contradictory":
        return 1
    if nclass == 3 and grade == "incorrect":
        return 2
    return None


def make_prompt(template: str, sample: Any) -> str:
    return template.format(
        question=sample.question,
        solution=sample.question_answer,
        student_answer=sample.student_answer,
    )


def apply_suffix(prompt: str, suffix: str, before_eos_tag: bool) -> str:
    if before_eos_tag:
        tag = "</student_answer>"
        pos = prompt.rfind(tag)
        if pos != -1:
            return prompt[:pos] + suffix + tag + prompt[pos + len(tag):]
    return prompt + suffix


def generate(model, tokenizer, prompt: str, device: str, max_tokens: int) -> str:
    import torch

    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[:, inputs["input_ids"].shape[1]:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]


def main() -> None:
    warnings.warn(
        "eval_gcg_suffix_bank.py is deprecated. "
        "Use pipeline.py with method: gcg_suffix_bank and pipeline_mode: true.",
        DeprecationWarning,
        stacklevel=2,
    )
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    suffixes = load_suffix_bank(args.bank, args.bank_limit)
    if not suffixes:
        raise ValueError(f"No suffixes loaded from {args.bank}")

    template = Path(args.template).read_text(encoding="utf-8")
    all_samples = read_student_qa_data_from_jsonl(args.data)
    excluded_source_indices = (
        set() if args.include_bank_source_indices else load_bank_source_indices(args.bank)
    )
    indexed_samples = [
        (source_idx, sample)
        for source_idx, sample in enumerate(all_samples)
        if source_idx not in excluded_source_indices
    ]
    if args.max_samples and args.max_samples < len(indexed_samples):
        rng = random.Random(args.random_seed)
        indexed_samples = rng.sample(indexed_samples, args.max_samples)

    model_path = resolve_model_path(args.model_path, args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    totals = Counter()
    successes = Counter()
    attempts = []

    with output_path.open("w", encoding="utf-8") as out:
        for idx, (source_index, sample) in enumerate(indexed_samples):
            label = sample.verification or ""
            prompt = make_prompt(template, sample)
            original_response = generate(model, tokenizer, prompt, device, args.max_tokens)
            original_grade = parsed_grade(original_response, args.nclass)

            row = {
                "index": idx,
                "source_index": source_index,
                "student_qa_data": sample.as_dict(),
                "original_response": original_response,
                "original_grade": original_grade,
                "success": False,
                "promotion_success": False,
                "success_suffix_index": None,
                "success_suffix": None,
                "promotion_suffix_index": None,
                "promotion_suffix": None,
                "attacked_response": None,
                "attacked_grade": None,
            }

            if label != "correct":
                totals["non_correct"] += 1
                totals[label] += 1

                # Project ASR counts only samples not already predicted correct.
                eligible = original_grade != "correct"
                if eligible:
                    totals["eligible_non_correct"] += 1
                    totals[f"eligible_{label}"] += 1

                promotion_found = False
                for suffix_idx, suffix in enumerate(suffixes):
                    attacked_prompt = apply_suffix(prompt, suffix, args.append_eos_tag)
                    attacked_response = generate(
                        model, tokenizer, attacked_prompt, device, args.max_tokens
                    )
                    attacked_grade = parsed_grade(attacked_response, args.nclass)
                    original_rank = grade_rank(original_grade, args.nclass)
                    attacked_rank = grade_rank(attacked_grade, args.nclass)
                    is_promotion = (
                        eligible
                        and original_rank is not None
                        and attacked_rank is not None
                        and attacked_rank < original_rank
                    )
                    if eligible and attacked_grade == "correct":
                        successes["eligible_non_correct"] += 1
                        successes[f"eligible_{label}"] += 1
                    if is_promotion and not promotion_found:
                        promotion_found = True
                        successes["eligible_promotion"] += 1
                        successes[f"eligible_promotion_{label}"] += 1
                        row.update({
                            "promotion_success": True,
                            "promotion_suffix_index": suffix_idx,
                            "promotion_suffix": suffix,
                        })
                    if attacked_grade == "correct":
                        successes["non_correct_any_original"] += 1
                        successes[f"{label}_any_original"] += 1
                        row.update({
                            "success": True,
                            "success_suffix_index": suffix_idx,
                            "success_suffix": suffix,
                            "attacked_response": attacked_response,
                            "attacked_grade": attacked_grade,
                        })
                        break
                attempts.append(row)
            else:
                totals["correct"] += 1

            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"[{idx + 1}/{len(indexed_samples)}] source_idx={source_index} label={label} "
                f"orig={original_grade} success={row['success']}",
                flush=True,
            )

    metrics = build_metrics(totals, successes, len(indexed_samples), len(suffixes))
    metrics["excluded_bank_source_indices"] = sorted(excluded_source_indices)
    metrics_path = output_path.with_name(output_path.stem + "_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Wrote {output_path}")
    print(f"Wrote {metrics_path}")


def rate(num: int, den: int) -> float:
    return num / den if den else 0.0


def build_metrics(totals: Counter, successes: Counter, total_samples: int, bank_size: int) -> dict:
    labels = sorted(
        k
        for k in totals
        if k not in {"correct", "non_correct", "eligible_non_correct"}
        and not k.startswith("eligible_")
    )
    by_label = {}
    for label in labels:
        by_label[label] = {
            "total": totals[label],
            "eligible_total": totals[f"eligible_{label}"],
            "transfer_success_any_original": successes[f"{label}_any_original"],
            "project_asr_success": successes[f"eligible_{label}"],
            "promotion_success": successes[f"eligible_promotion_{label}"],
            "transfer_rate_any_original": rate(successes[f"{label}_any_original"], totals[label]),
            "project_asr": rate(successes[f"eligible_{label}"], totals[f"eligible_{label}"]),
            "promotion_asr": rate(
                successes[f"eligible_promotion_{label}"], totals[f"eligible_{label}"]
            ),
        }
    return {
        "total_samples": total_samples,
        "bank_size": bank_size,
        "total_correct": totals["correct"],
        "total_non_correct": totals["non_correct"],
        "eligible_non_correct": totals["eligible_non_correct"],
        "transfer_success_any_original": successes["non_correct_any_original"],
        "project_asr_success": successes["eligible_non_correct"],
        "promotion_success": successes["eligible_promotion"],
        "transfer_rate_any_original": rate(
            successes["non_correct_any_original"], totals["non_correct"]
        ),
        "project_asr": rate(
            successes["eligible_non_correct"], totals["eligible_non_correct"]
        ),
        "promotion_asr": rate(
            successes["eligible_promotion"], totals["eligible_non_correct"]
        ),
        "by_label": by_label,
        "asr_note": (
            "transfer_rate_any_original counts all ground-truth non-correct samples "
            "that become correct under any bank suffix. project_asr matches this "
            "project's stricter metric: original model prediction must not already "
            "be correct, and attack prediction must become correct. promotion_asr "
            "counts any grading relaxation, e.g. incorrect -> contradictory/correct "
            "in 3-class or incorrect -> correct in 2-class."
        ),
    }


if __name__ == "__main__":
    main()
