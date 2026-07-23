"""
Clean baseline: evaluate LLM grading accuracy without attack or defense.

Examples:
    python baseline_eval.py --config configs/GCG-Llama-3.1-8B-Instruct.yaml --max_samples 200

    python baseline_eval.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --data ./dataset/scientsbank.jsonl \
        --device cuda:0 \
        --max_samples 200 \
        --output ./result_baseline/scientsbank_baseline.jsonl
"""

import argparse
import json
import os
import random
from dataclasses import asdict
from datetime import datetime
from typing import Optional

from utils.config_utils import AttackConfig, parse_config
from utils.data_utils import extract_grade, read_student_qa_data_from_jsonl


def resolve_model_path(config: AttackConfig) -> str:
    """Return a local/HF model path, downloading from ModelScope when needed."""
    if config.model_config.path:
        return config.model_config.path
    if config.model_config.model_id:
        from modelscope import snapshot_download

        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub")
        print(f"[baseline] Downloading {config.model_config.model_id} from ModelScope...")
        return snapshot_download(config.model_config.model_id, cache_dir=cache_dir)
    raise ValueError("Either model.path or model.model_id must be set.")


def load_template(path: Optional[str], config: Optional[AttackConfig]) -> str:
    if config is not None:
        return config.grading_template
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    default_path = os.path.join("configs", "grading_template_ci.txt")
    with open(default_path, "r", encoding="utf-8") as f:
        return f.read()


def normalize_prediction(pred: Optional[str]) -> Optional[str]:
    if pred is None:
        return None
    pred = str(pred).strip().lower()
    if pred in ("correct", "0"):
        return "correct"
    if pred in ("contradictory", "1"):
        return "contradictory"
    if pred in ("incorrect", "2"):
        return "incorrect"
    return None


def binary_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    return "correct" if label.strip().lower() == "correct" else "incorrect"


def class_id(label: Optional[str]) -> Optional[int]:
    label = normalize_prediction(label)
    if label == "correct":
        return 0
    if label == "contradictory":
        return 1
    if label == "incorrect":
        return 2
    return None


def compute_qwk(labels: list[int], preds: list[Optional[int]]) -> float:
    valid = [(label, pred) for label, pred in zip(labels, preds) if pred is not None]
    if not valid:
        return 0.0
    from sklearn.metrics import cohen_kappa_score

    y_true, y_pred = zip(*valid)
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def compute_baseline_metrics(results: list[dict]) -> dict:
    total = len(results)
    if total == 0:
        raise ValueError("No results to evaluate.")

    binary_matches = sum(
        1 for r in results
        if binary_label(r["pred"]) == binary_label(r["label"])
    )
    valid = sum(1 for r in results if r["pred"] is not None)

    tp = sum(1 for r in results if binary_label(r["pred"]) == "correct"
             and binary_label(r["label"]) == "correct")
    fp = sum(1 for r in results if binary_label(r["pred"]) == "correct"
             and binary_label(r["label"]) == "incorrect")
    fn = sum(1 for r in results if binary_label(r["pred"]) == "incorrect"
             and binary_label(r["label"]) == "correct")
    tn = sum(1 for r in results if binary_label(r["pred"]) == "incorrect"
             and binary_label(r["label"]) == "incorrect")

    prec_correct = tp / (tp + fp) if tp + fp else 0.0
    rec_correct = tp / (tp + fn) if tp + fn else 0.0
    f1_correct = (
        2 * prec_correct * rec_correct / (prec_correct + rec_correct)
        if prec_correct + rec_correct else 0.0
    )

    prec_incorrect = tn / (tn + fn) if tn + fn else 0.0
    rec_incorrect = tn / (tn + fp) if tn + fp else 0.0
    f1_incorrect = (
        2 * prec_incorrect * rec_incorrect / (prec_incorrect + rec_incorrect)
        if prec_incorrect + rec_incorrect else 0.0
    )

    labels_3way = [0 if binary_label(r["label"]) == "correct" else 2 for r in results]
    preds_3way = [class_id(r["pred"]) for r in results]

    return {
        "total": total,
        "accuracy": binary_matches / total,
        "valid_rate": valid / total,
        "qwk": compute_qwk(labels_3way, preds_3way),
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
        "macro_f1": (f1_correct + f1_incorrect) / 2,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def run_baseline(
    model_path: str,
    data_path: str,
    template: str,
    device: str = "cuda",
    max_samples: Optional[int] = None,
    random_seed: Optional[int] = None,
    max_tokens: int = 1024,
    temperature: float = 0.01,
) -> tuple[dict, list[dict]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[baseline] Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"[baseline] Loading data: {data_path}")
    data_list = read_student_qa_data_from_jsonl(data_path)
    if max_samples and max_samples < len(data_list):
        if random_seed is None:
            data_list = data_list[:max_samples]
        else:
            data_list = random.Random(random_seed).sample(data_list, max_samples)

    total = len(data_list)
    print(f"[baseline] Evaluating {total} samples on {device}...")

    results = []
    for i, item in enumerate(data_list, start=1):
        prompt = template.format(
            question=item.question,
            solution=item.question_answer,
            student_answer=item.student_answer,
        )
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                do_sample=False,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )

        response = tokenizer.batch_decode(
            outputs[:, inputs.shape[1]:], skip_special_tokens=True
        )[0]
        pred = normalize_prediction(extract_grade(response))
        label = binary_label(item.verification)

        results.append({
            "question_id": item.question_id,
            "student_id": item.student_id,
            "label": label,
            "pred": pred,
            "response": response,
            "student_qa_data": asdict(item),
        })

        if i == total or i % 25 == 0:
            metrics = compute_baseline_metrics(results)
            print(
                f"  [{i}/{total}] accuracy={metrics['accuracy']:.4f} "
                f"valid={metrics['valid_rate']:.1%}"
            )

    return compute_baseline_metrics(results), results


def print_report(metrics: dict):
    print("\n" + "=" * 60)
    print("  CLEAN BASELINE - Grading Accuracy (no attack, no defense)")
    print("=" * 60)
    print(f"  Samples:          {metrics['total']}")
    print(f"  Valid responses:  {metrics['valid_rate']:.1%}")
    print(f"  Accuracy:         {metrics['accuracy']:.4f} ({metrics['accuracy']:.2%})")
    print(f"  QWK:              {metrics['qwk']:.4f}")
    print(f"  Macro F1:         {metrics['macro_f1']:.4f}")
    print()
    c = metrics["correct_class"]
    print(f"  [correct]   P={c['precision']:.4f} R={c['recall']:.4f} F1={c['f1']:.4f}")
    c = metrics["incorrect_class"]
    print(f"  [incorrect] P={c['precision']:.4f} R={c['recall']:.4f} F1={c['f1']:.4f}")
    print()
    cm = metrics["confusion"]
    print("  Confusion Matrix (binary)")
    print(f"    TP correct->correct:     {cm['tp']}")
    print(f"    FN correct->incorrect:   {cm['fn']}")
    print(f"    TN incorrect->incorrect: {cm['tn']}")
    print(f"    FP incorrect->correct:   {cm['fp']}")
    print("=" * 60)


def save_outputs(output_path: str, metrics: dict, results: list[dict]):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    metrics_path = output_path.replace(".jsonl", "_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n[baseline] Detailed results saved to: {output_path}")
    print(f"[baseline] Metrics saved to: {metrics_path}")


def build_default_output(config: Optional[AttackConfig], data_path: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    model_name = config.model_config.name if config else "manual-model"
    data_name = os.path.splitext(os.path.basename(data_path))[0]
    return os.path.join(
        "result_baseline",
        model_name,
        f"{data_name}_clean_baseline_{timestamp}.jsonl",
    )


def main():
    parser = argparse.ArgumentParser(description="Clean baseline grading evaluation")
    parser.add_argument("--config", type=str, help="YAML config to reuse model/data/template settings")
    parser.add_argument("--model", type=str, help="Model path / HF ID, used without --config")
    parser.add_argument("--data", type=str, help="JSONL dataset path, used without --config")
    parser.add_argument("--template", type=str, help="Prompt template path, default: configs/grading_template_ci.txt")
    parser.add_argument("--device", type=str, default=None, help="Device, e.g. cuda:0 or cpu")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit sample count")
    parser.add_argument("--random_seed", type=int, default=None, help="Sample randomly with this seed")
    parser.add_argument("--max_tokens", type=int, default=None, help="Max generated tokens")
    parser.add_argument("--temperature", type=float, default=None, help="Generation temperature")
    parser.add_argument("--output", type=str, default=None, help="Save detailed JSONL results here")
    args = parser.parse_args()

    config = parse_config(args.config) if args.config else None
    template = load_template(args.template, config)

    if config:
        model_path = resolve_model_path(config)
        device = args.device or config.params.get("device") or (
            "cuda" if cuda_available() else "cpu"
        )
        max_tokens = args.max_tokens or config.generation_config.max_tokens
        temperature = (
            args.temperature
            if args.temperature is not None
            else config.generation_config.temperature
        )
        data_configs = config.data_config
    else:
        if not args.model or not args.data:
            parser.error("Either --config or both --model and --data are required.")
        model_path = args.model
        device = args.device or ("cuda" if cuda_available() else "cpu")
        max_tokens = args.max_tokens or 1024
        temperature = args.temperature if args.temperature is not None else 0.01
        data_configs = [type("ManualDataConfig", (), {
            "name": os.path.splitext(os.path.basename(args.data))[0],
            "path": args.data,
            "max_samples": None,
            "random_seed": None,
        })()]

    all_metrics = {}
    for data_config in data_configs:
        max_samples = args.max_samples if args.max_samples is not None else data_config.max_samples
        random_seed = args.random_seed if args.random_seed is not None else data_config.random_seed
        metrics, results = run_baseline(
            model_path=model_path,
            data_path=data_config.path,
            template=template,
            device=device,
            max_samples=max_samples,
            random_seed=random_seed,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        print_report(metrics)

        output_path = args.output or build_default_output(config, data_config.path)
        if len(data_configs) > 1 and args.output:
            root, ext = os.path.splitext(args.output)
            output_path = f"{root}_{data_config.name}{ext or '.jsonl'}"
        save_outputs(output_path, metrics, results)
        all_metrics[data_config.name] = metrics

    if len(all_metrics) > 1:
        print("\n[baseline] Summary")
        for name, metrics in all_metrics.items():
            print(
                f"  {name}: accuracy={metrics['accuracy']:.4f}, "
                f"qwk={metrics['qwk']:.4f}, valid={metrics['valid_rate']:.1%}"
            )


if __name__ == "__main__":
    main()

