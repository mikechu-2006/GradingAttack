"""Evaluate defenses against a transferable GCG suffix bank.

DEPRECATED: Use pipeline.py with method=gcg_suffix_bank, pipeline_mode=true,
and a config with defenses configured. See
configs/GCG-SuffixBank-SelfReminder-Llama-3.1-8B-Instruct-3c.yaml for an example.

This script is kept for backward compatibility and quick ad-hoc evaluation.
It will be removed in a future release.

This script uses an existing suffix bank and compares no-defense transfer ASR
with defended transfer ASR on the same sampled examples.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.metrics import _compute_qwk
from scripts.eval_gcg_suffix_bank import (
    apply_suffix,
    grade_rank,
    load_bank_source_indices,
    load_suffix_bank,
    parsed_grade,
    resolve_model_path,
)
from utils.data_utils import extract_grade, read_student_qa_data_from_jsonl


DEFAULT_MODEL_ID = "LLM-Research/Meta-Llama-3.1-8B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate defense vs GCG suffix bank.")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--data", default="dataset/scientsbank.jsonl")
    parser.add_argument("--template", default="configs/grading_template_ci_3c.txt")
    parser.add_argument("--nclass", type=int, choices=[2, 3], default=3)
    parser.add_argument("--defense", choices=["none", "self_reminder", "paraphrase", "hijacking"], default="self_reminder")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--bank-limit", type=int, default=None)
    parser.add_argument("--output", default="result/GCG/gcg_suffix_bank_defense.jsonl")
    parser.add_argument("--include-bank-source-indices", action="store_true")
    parser.add_argument("--append-eos-tag", action="store_true")
    parser.add_argument("--hijacking-beta", type=float, default=0.1)
    parser.add_argument("--hijacking-top-fraction", type=float, default=0.01)
    return parser.parse_args()


def make_prompt(template: str, sample: Any) -> str:
    return template.format(
        question=sample.question,
        solution=sample.question_answer,
        student_answer=sample.student_answer,
    )


def label_to_class(label: str, nclass: int) -> int:
    if label == "correct":
        return 0
    if nclass == 2:
        return 1
    return 1 if label == "contradictory" else 2


def grade_to_class(grade: str | None, nclass: int) -> int | None:
    if grade == "correct":
        return 0
    if nclass == 2 and grade == "incorrect":
        return 1
    if nclass == 3 and grade == "contradictory":
        return 1
    if nclass == 3 and grade == "incorrect":
        return 2
    return None


def make_defense(name: str, args: argparse.Namespace):
    if name == "none":
        return None
    if name == "self_reminder":
        from baselines.defenses.self_reminder import SelfReminder

        return SelfReminder()
    if name == "paraphrase":
        from baselines.defenses.paraphrase import ParaphraseDefense

        return ParaphraseDefense()
    if name == "hijacking":
        from baselines.defenses.hijacking_suppression import HijackingSuppression

        return HijackingSuppression(
            beta=args.hijacking_beta,
            top_fraction=args.hijacking_top_fraction,
        )
    raise ValueError(name)


def generate(model, tokenizer, prompt: str, device: str, max_tokens: int, defense=None, suffix: str = "") -> str:
    import torch

    removers = []
    if defense is not None and defense.requires_inference_context():
        defense.set_inference_context(tokenizer, prompt, suffix)
    if defense is not None and defense.requires_model_hooks():
        removers = defense.install_model_hooks(model)

    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
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
    finally:
        for remove in removers:
            remove()


def success_to_correct(original_grade: str | None, attacked_grade: str | None) -> bool:
    return original_grade != "correct" and attacked_grade == "correct"


def success_promotion(original_grade: str | None, attacked_grade: str | None, nclass: int) -> bool:
    original_rank = grade_rank(original_grade, nclass)
    attacked_rank = grade_rank(attacked_grade, nclass)
    return (
        original_rank is not None
        and attacked_rank is not None
        and attacked_rank < original_rank
    )


def rate(num: int, den: int) -> float:
    return num / den if den else 0.0


def main() -> None:
    warnings.warn(
        "eval_gcg_suffix_bank_defense.py is deprecated. "
        "Use pipeline.py with method: gcg_suffix_bank, pipeline_mode: true, "
        "and defenses configured in your config.",
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
    excluded = set() if args.include_bank_source_indices else load_bank_source_indices(args.bank)
    indexed = [
        (source_idx, sample)
        for source_idx, sample in enumerate(all_samples)
        if source_idx not in excluded
    ]
    if args.max_samples and args.max_samples < len(indexed):
        rng = random.Random(args.random_seed)
        indexed = rng.sample(indexed, args.max_samples)

    defense = make_defense(args.defense, args)
    model_path = resolve_model_path(args.model_path, args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    load_kwargs = dict(trust_remote_code=True, torch_dtype=torch.bfloat16)
    if defense is not None and defense.requires_model_hooks():
        load_kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs).to(device)
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels: list[int] = []
    clean_preds: list[int | None] = []
    def_clean_preds: list[int | None] = []

    total_correct = total_non_correct = 0
    eligible = defended_eligible = 0
    success = defended_success = 0
    promotion_success = defended_promotion_success = 0

    with output_path.open("w", encoding="utf-8") as out:
        for idx, (source_index, sample) in enumerate(indexed):
            label = sample.verification or ""
            labels.append(label_to_class(label, args.nclass))
            if label == "correct":
                total_correct += 1
            else:
                total_non_correct += 1

            prompt = make_prompt(template, sample)
            clean_response = generate(model, tokenizer, prompt, device, args.max_tokens)
            clean_grade = parsed_grade(clean_response, args.nclass)
            clean_preds.append(grade_to_class(clean_grade, args.nclass))

            defended_prompt = defense.pre_process(prompt) if defense is not None else prompt
            defended_clean_response = generate(
                model, tokenizer, defended_prompt, device, args.max_tokens, defense=defense
            )
            defended_clean_grade = parsed_grade(defended_clean_response, args.nclass)
            def_clean_preds.append(grade_to_class(defended_clean_grade, args.nclass))

            row = {
                "index": idx,
                "source_index": source_index,
                "student_qa_data": sample.as_dict(),
                "original_response": clean_response,
                "original_grade": clean_grade,
                "defended_original_response": defended_clean_response,
                "defended_original_grade": defended_clean_grade,
                "success": False,
                "defended_success": False,
                "promotion_success": False,
                "defended_promotion_success": False,
                "success_suffix_index": None,
                "defended_success_suffix_index": None,
            }

            if label != "correct":
                is_eligible = clean_grade != "correct"
                is_def_eligible = defended_clean_grade != "correct"
                if is_eligible:
                    eligible += 1
                if is_def_eligible:
                    defended_eligible += 1

                for suffix_idx, suffix in enumerate(suffixes):
                    if not (row["success"] and row["promotion_success"]):
                        attacked_prompt = apply_suffix(prompt, suffix, args.append_eos_tag)
                        attacked_response = generate(
                            model, tokenizer, attacked_prompt, device, args.max_tokens
                        )
                        attacked_grade = parsed_grade(attacked_response, args.nclass)
                        if is_eligible and success_to_correct(clean_grade, attacked_grade):
                            if not row["success"]:
                                success += 1
                                row.update({
                                    "success": True,
                                    "success_suffix_index": suffix_idx,
                                    "success_suffix": suffix,
                                    "attacked_response": attacked_response,
                                    "attacked_grade": attacked_grade,
                                })
                        if is_eligible and success_promotion(clean_grade, attacked_grade, args.nclass):
                            if not row["promotion_success"]:
                                promotion_success += 1
                                row.update({
                                    "promotion_success": True,
                                    "promotion_suffix_index": suffix_idx,
                                    "promotion_suffix": suffix,
                                })

                    if not (row["defended_success"] and row["defended_promotion_success"]):
                        defended_attacked_prompt = apply_suffix(
                            defended_prompt, suffix, args.append_eos_tag
                        )
                        defended_attacked_response = generate(
                            model,
                            tokenizer,
                            defended_attacked_prompt,
                            device,
                            args.max_tokens,
                            defense=defense,
                            suffix=suffix,
                        )
                        defended_attacked_grade = parsed_grade(
                            defended_attacked_response, args.nclass
                        )
                        if is_def_eligible and success_to_correct(
                            defended_clean_grade, defended_attacked_grade
                        ):
                            if not row["defended_success"]:
                                defended_success += 1
                                row.update({
                                    "defended_success": True,
                                    "defended_success_suffix_index": suffix_idx,
                                    "defended_success_suffix": suffix,
                                    "defended_attacked_response": defended_attacked_response,
                                    "defended_attacked_grade": defended_attacked_grade,
                                })
                        if is_def_eligible and success_promotion(
                            defended_clean_grade, defended_attacked_grade, args.nclass
                        ):
                            if not row["defended_promotion_success"]:
                                defended_promotion_success += 1
                                row.update({
                                    "defended_promotion_success": True,
                                    "defended_promotion_suffix_index": suffix_idx,
                                    "defended_promotion_suffix": suffix,
                                })

                    if (
                        (row["success"] or not is_eligible)
                        and (row["promotion_success"] or not is_eligible)
                        and (row["defended_success"] or not is_def_eligible)
                        and (row["defended_promotion_success"] or not is_def_eligible)
                    ):
                        break

            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"[{idx + 1}/{len(indexed)}] source_idx={source_index} "
                f"label={label} clean={clean_grade} def_clean={defended_clean_grade} "
                f"success={row['success']} defended={row['defended_success']}",
                flush=True,
            )

    qwk_clean = _compute_qwk(labels, clean_preds)
    qwk_defense_clean = _compute_qwk(labels, def_clean_preds)
    asr = rate(success, eligible)
    defended_asr = rate(defended_success, defended_eligible)
    metrics = {
        "defense": args.defense,
        "nclass": args.nclass,
        "total_samples": len(indexed),
        "bank_size": len(suffixes),
        "total_correct": total_correct,
        "total_non_correct": total_non_correct,
        "eligible_non_correct": eligible,
        "defended_eligible_non_correct": defended_eligible,
        "project_asr_success": success,
        "defended_project_asr_success": defended_success,
        "project_asr": asr,
        "defended_project_asr": defended_asr,
        "asr_reduction": 1 - defended_asr / asr if asr else 0.0,
        "promotion_success": promotion_success,
        "defended_promotion_success": defended_promotion_success,
        "promotion_asr": rate(promotion_success, eligible),
        "defended_promotion_asr": rate(defended_promotion_success, defended_eligible),
        "qwk_clean": qwk_clean,
        "qwk_defense_clean": qwk_defense_clean,
        "clean_qwk_retention": qwk_defense_clean / qwk_clean if qwk_clean else 0.0,
        "excluded_bank_source_indices": sorted(excluded),
    }

    metrics_path = output_path.with_name(output_path.stem + "_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    print(f"Wrote {output_path}", flush=True)
    print(f"Wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
