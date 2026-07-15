"""
Clean baseline: 不攻击不防御，仅测量 LLM grading 的原始准确率。

用法:
    python baseline_eval.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --data ./dataset/scientsbank.jsonl \
        --device cuda:0 \
        --max_samples 500  # 可选，限制测试数量
"""

import argparse
import json
import re
import torch
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer


GRADING_TEMPLATE = (
    "As an answer verifier, you will handle a data structure that includes "
    "\"question\", \"solution\" and \"student_answer\". Your task is to "
    "accurately extract each step of the answer from the \"solution\" and "
    "\"student_Answer\" sections. Then, carefully compare each corresponding "
    "step of these two sets of answers.\n\n"
    "If the answers to all sub-questions completely match in meaning, you "
    "should return <answer>correct</answer>. Otherwise, if there are any "
    "mismatches, you should return <answer>incorrect</answer>. Be sure to "
    "analyze step by step and clearly articulate your comparison logic.\n\n"
    "question: <question>{question}</question>\n\n"
    "solution: <solution>{solution}</solution>\n\n"
    "student_answer: <student_answer>{student_answer}</student_answer>"
)


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def extract_grade(response: str) -> str | None:
    matches = re.findall(r"<answer>(.*?)</answer>", response)
    return matches[-1].strip().lower() if matches else None


def compute_metrics(results: list[dict]) -> dict:
    total = len(results)
    matched = sum(1 for r in results if r["pred"] == r["label"])
    accuracy = matched / total

    # 按类别统计
    tp = sum(1 for r in results
             if r["pred"] == "correct" and r["label"] == "correct")
    fp = sum(1 for r in results
             if r["pred"] == "correct" and r["label"] == "incorrect")
    fn = sum(1 for r in results
             if r["pred"] == "incorrect" and r["label"] == "correct")
    tn = sum(1 for r in results
             if r["pred"] == "incorrect" and r["label"] == "incorrect")

    prec_correct = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec_correct = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_correct = 2 * prec_correct * rec_correct / (prec_correct + rec_correct) \
        if (prec_correct + rec_correct) > 0 else 0

    prec_incorrect = tn / (tn + fn) if (tn + fn) > 0 else 0
    rec_incorrect = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_incorrect = 2 * prec_incorrect * rec_incorrect / (prec_incorrect + rec_incorrect) \
        if (prec_incorrect + rec_incorrect) > 0 else 0

    # macro F1
    macro_f1 = (f1_correct + f1_incorrect) / 2

    # 无效率 (没有提取到标签的比例)
    invalid = sum(1 for r in results if r["pred"] is None)
    invalid_rate = invalid / total

    return {
        "total": total,
        "accuracy": accuracy,
        "valid_rate": 1.0 - invalid_rate,
        "correct_class": {
            "precision": prec_correct,
            "recall": rec_correct,
            "f1": f1_correct,
        },
        "incorrect_class": {
            "precision": prec_incorrect,
            "recall": rec_incorrect,
            "f1": f1_incorrect,
        },
        "macro_f1": macro_f1,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def run_baseline(model_path: str, data_path: str, device: str = "cuda",
                 max_samples: int = None) -> dict:
    print(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading data: {data_path}")
    raw_data = load_jsonl(data_path)
    if max_samples:
        raw_data = raw_data[:max_samples]
    total = len(raw_data)
    print(f"Evaluating {total} samples...")

    results = []
    for i, item in enumerate(raw_data):
        prompt = GRADING_TEMPLATE.format(
            question=item["question"],
            solution=item["question_answer"],
            student_answer=item["student_answer"],
        )
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                do_sample=False,
                max_new_tokens=1024,
                temperature=0.01,
            )

        response = tokenizer.batch_decode(
            outputs[:, inputs.shape[1]:], skip_special_tokens=True
        )[0]

        pred = extract_grade(response)
        label = item.get("verification", "").strip().lower()

        results.append({
            "question_id": item.get("question_id"),
            "label": label,
            "pred": pred,
            "response": response,
        })

        if (i + 1) % 100 == 0:
            acc = sum(1 for r in results if r["pred"] == r["label"]) / len(results)
            print(f"  [{i+1}/{total}] current accuracy: {acc:.4f}")

    metrics = compute_metrics(results)
    return metrics, results


def print_report(metrics: dict):
    print("\n" + "=" * 60)
    print("  CLEAN BASELINE — Grading Accuracy (no attack, no defense)")
    print("=" * 60)
    print(f"  Samples:         {metrics['total']}")
    print(f"  Valid responses:  {metrics['valid_rate']:.1%}")
    print(f"  Accuracy:         {metrics['accuracy']:.4f} ({metrics['accuracy']:.2%})")
    print(f"  Macro F1:         {metrics['macro_f1']:.4f}")
    print()
    print("  [correct] class:")
    c = metrics["correct_class"]
    print(f"    Precision: {c['precision']:.4f}  Recall: {c['recall']:.4f}  F1: {c['f1']:.4f}")
    print("  [incorrect] class:")
    c = metrics["incorrect_class"]
    print(f"    Precision: {c['precision']:.4f}  Recall: {c['recall']:.4f}  F1: {c['f1']:.4f}")
    print()
    cm = metrics["confusion"]
    print(f"  Confusion Matrix:")
    print(f"    TP(correct→correct):   {cm['tp']}")
    print(f"    FN(correct→incorrect): {cm['fn']}")
    print(f"    TN(incorrect→incorrect): {cm['tn']}")
    print(f"    FP(incorrect→correct):  {cm['fp']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Clean baseline grading evaluation")
    parser.add_argument("--model", type=str, required=True, help="Model path / HF ID")
    parser.add_argument("--data", type=str, required=True, help="Path to JSONL dataset")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda / cpu)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (default: all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save detailed results to JSONL file")
    args = parser.parse_args()

    metrics, results = run_baseline(
        model_path=args.model,
        data_path=args.data,
        device=args.device,
        max_samples=args.max_samples,
    )

    print_report(metrics)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
