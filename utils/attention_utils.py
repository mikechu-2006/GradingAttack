"""
Prompt 语义分段 + attention 权重统计。

将评分 prompt 按 6 个 segment 切分，统计模型在预测位置对各段的 attention 分布。
"""

import math
import re
import sys
import torch
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

import pandas as pd


def _safe_bar(n: int, width: int) -> str:
    """Return a safe bar string, falling back to ASCII on Windows GBK."""
    bar = "█" * n + "░" * (width - n)
    try:
        bar.encode(sys.stdout.encoding or "utf-8")
        return bar
    except (UnicodeEncodeError, UnicodeDecodeError):
        return "#" * n + "-" * (width - n)


# ── 分段 ────────────────────────────────────────────────

# 标准的 6 个 segment
ALL_SEGMENTS = ["instruction", "question", "solution", "student", "suffix", "markup"]


def segment_prompt(prompt: str, tokenizer,
                   gcg_suffix: str = "") -> Tuple[List[str], List[str]]:
    """
    将 prompt 中每个 token 归类到语义 segment。

    Returns:
        token_to_seg: 每个 token 的 segment 标签
        ordered_segments: 实际出现的有序 segment 列表
    """
    offsets = tokenizer(prompt, return_offsets_mapping=True,
                        add_special_tokens=False)["offset_mapping"]

    # 1. 用 XML 标签定位各段的字符范围
    seg_ranges = _char_ranges(prompt, gcg_suffix)

    # 2. 按 token 的字符中点归类
    token_to_seg = ["markup"] * len(offsets)
    for token_idx, (start_char, end_char) in enumerate(offsets):
        center = (start_char + end_char) // 2
        for seg_name, (seg_start, seg_end) in seg_ranges.items():
            if seg_start <= center < seg_end:
                token_to_seg[token_idx] = seg_name
                break

    # 3. 按出现顺序排列 segment
    ordered = _order_segments(seg_ranges, gcg_suffix)
    return token_to_seg, ordered


def _char_ranges(prompt: str, gcg_suffix: str) -> Dict[str, Tuple[int, int]]:
    """用 XML 标签定位各语义段的字符范围。"""
    ranges = {}

    # 匹配 <question>, <solution>, <student_answer> 标签对
    tag_pattern = re.compile(
        r'(<question>)(.*?)(</question>)|'
        r'(<solution>)(.*?)(</solution>)|'
        r'(<student_answer>)(.*?)(</student_answer>)',
        re.DOTALL,
    )

    for m in tag_pattern.finditer(prompt):
        if m.group(1):   # question
            ranges["question_open"]  = (m.start(1), m.end(1))
            ranges["question"]       = (m.end(1),   m.start(3))
            ranges["question_close"] = (m.start(3), m.end(3))
        elif m.group(4):  # solution
            ranges["solution_open"]  = (m.start(4), m.end(4))
            ranges["solution"]       = (m.end(4),   m.start(6))
            ranges["solution_close"] = (m.start(6), m.end(6))
        elif m.group(7):  # student_answer
            ranges["student_open"]  = (m.start(7), m.end(7))
            ranges["student"]       = (m.end(7),   m.start(9))
            ranges["student_close"] = (m.start(9), m.end(9))

    # instruction: 开头到第一个 <question> 标签之前
    q_open = ranges.get("question_open")
    if q_open:
        ranges["instruction"] = (0, q_open[0])

    # trailing: 最后一个 </student_answer> 之后
    sa_close = ranges.get("student_close")
    if sa_close:
        ranges["trailing"] = (sa_close[1], len(prompt))

    # GCG suffix 拆分（嵌在 student_answer 内）
    if gcg_suffix and "student" in ranges:
        s_start, s_end = ranges["student"]
        suffix_pos = prompt.find(gcg_suffix, s_start, s_end)
        if suffix_pos != -1:
            ranges["student"] = (s_start, suffix_pos)
            ranges["suffix"] = (suffix_pos, suffix_pos + len(gcg_suffix))

    # RolePlay / 尾部 suffix（在 </student_answer> 之后）
    if gcg_suffix and "suffix" not in ranges:
        suffix_pos = prompt.find(gcg_suffix)
        if suffix_pos != -1:
            sa_close = ranges.get("student_close")
            if sa_close is None or suffix_pos >= sa_close[1]:
                ranges["suffix"] = (suffix_pos, suffix_pos + len(gcg_suffix))
                if "trailing" in ranges:
                    t_start, t_end = ranges["trailing"]
                    if suffix_pos > t_start:
                        ranges["trailing"] = (t_start, suffix_pos)

    return ranges


def _order_segments(ranges: dict, gcg_suffix: str) -> List[str]:
    """按字符位置排序 segment，合并标记类 segment 到 markup。"""
    # 只保留用户关心的 segment
    wanted = {"instruction", "question", "solution", "student", "suffix", "markup"}

    pos_pairs = []
    for name, (s, e) in ranges.items():
        # 把 tag 类 + trailing 都归为 markup
        seg = name if name in wanted else "markup"
        pos_pairs.append((s, seg))

    pos_pairs.sort()
    seen = set()
    ordered = []
    for _, seg in pos_pairs:
        if seg not in seen:
            seen.add(seg)
            ordered.append(seg)

    # suffix 放到 student 后面
    if "suffix" in seen:
        ordered.remove("suffix")
        student_idx = ordered.index("student") if "student" in ordered else -1
        ordered.insert(student_idx + 1, "suffix")

    return ordered


# ── attention 分析 ───────────────────────────────────────

def _build_token_position_weights(
    attn_row: torch.Tensor,
    token_to_seg: List[str],
    tokenizer,
    input_ids: torch.Tensor,
) -> List[Dict[str, object]]:
    """最后一个预测 token 对各 token 位置的 attention 权重。"""
    n = min(len(token_to_seg), attn_row.shape[0])
    positions: List[Dict[str, object]] = []
    ids = input_ids[0].tolist() if input_ids.dim() > 1 else input_ids.tolist()
    for pos in range(n):
        tok_id = ids[pos] if pos < len(ids) else None
        tok_str = ""
        if tok_id is not None:
            tok_str = tokenizer.decode([tok_id], skip_special_tokens=False)
            tok_str = tok_str.replace("\n", "\\n").replace("\r", "\\r")
            if len(tok_str) > 24:
                tok_str = tok_str[:21] + "..."
        positions.append({
            "pos": pos,
            "segment": token_to_seg[pos],
            "weight": float(attn_row[pos].item()),
            "token": tok_str,
        })
    return positions


def run_attention_analysis(model, tokenizer, prompt_text: str,
                           token_to_seg: List[str],
                           ordered_segments: List[str]) -> dict:
    """前向传播并聚合最后 token 的 attention 分布。"""
    inputs = tokenizer(prompt_text, return_tensors="pt",
                       add_special_tokens=False).to(model.device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions  # tuple of (B, H, S, S)
    num_layers = len(attentions)
    seq_len = attentions[0].shape[-1]

    # 截断到一致长度
    n = min(len(token_to_seg), seq_len)
    token_to_seg = token_to_seg[:n]

    # 最后一层
    last_attn = attentions[-1][0].mean(dim=0)[:n, :n]  # (S, S)
    last_row = last_attn[n - 1, :]
    last_seg = _agg_last_token(last_attn, token_to_seg, ordered_segments)
    last_positions = _build_token_position_weights(
        last_row, token_to_seg, tokenizer, input_ids
    )

    # 最后 3 层平均
    num_last = min(3, num_layers)
    stacked = torch.stack([
        attentions[i][0].mean(dim=0)[:n, :n] for i in range(num_layers - num_last, num_layers)
    ]).mean(dim=0)
    last3_row = stacked[n - 1, :]
    last3_seg = _agg_last_token(stacked, token_to_seg, ordered_segments)
    last3_positions = _build_token_position_weights(
        last3_row, token_to_seg, tokenizer, input_ids
    )

    return {
        "last_layer": last_seg,
        "last_3_avg": last3_seg,
        "last_layer_positions": last_positions,
        "last_3_avg_positions": last3_positions,
        "_segments": ordered_segments,
    }


def _agg_last_token(attn: torch.Tensor,
                    token_to_seg: List[str],
                    ordered_segments: List[str]) -> Dict[str, float]:
    """聚合最后一个 token 对各 segment 的 attention。"""
    n = attn.shape[0]
    last_row = attn[n - 1, :]  # (S,)
    seg_sums: Dict[str, float] = defaultdict(float)
    for idx, seg in enumerate(token_to_seg):
        seg_sums[seg] += last_row[idx].item()

    total = sum(seg_sums.values()) or 1.0
    return {seg: seg_sums.get(seg, 0.0) / total for seg in ordered_segments}


# ── 格式化输出 ───────────────────────────────────────────

def _format_position_value_lines(
    positions: List[Dict[str, object]],
    *,
    prefix: str,
    label: str,
    layer_label: str,
    chunk_size: int = 32,
) -> List[str]:
    """逐位置 attention 数值：可读行 + 紧凑解析行。"""
    if not positions:
        return []

    lines = [
        f"[ATTENTION_POSITION] {prefix}{label} — {layer_label} "
        f"(prediction → each token position, n={len(positions)})",
    ]
    for entry in positions:
        lines.append(
            f"  pos={entry['pos']:4d}  seg={entry['segment']:<12s}  "
            f"w={entry['weight']:.6f}  tok={entry['token']!r}"
        )

    compact_parts = [
        f"{entry['pos']}:{entry['segment']}:{entry['weight']:.6f}"
        for entry in positions
    ]
    for i in range(0, len(compact_parts), chunk_size):
        chunk = compact_parts[i:i + chunk_size]
        lines.append(
            f"[ATTENTION_POSITION_VALUES] {prefix}{label} {layer_label} "
            f"chunk={i // chunk_size}: {' '.join(chunk)}"
        )
    return lines


def format_attention_log_lines(summary: dict, bar_len: int = 40) -> List[str]:
    """将 attention 分布格式化为可写入 stdout / .log 的文本行。"""
    ordered_segments = summary.get("_segments", [])
    if not ordered_segments:
        return []

    label = summary.get("_label", "[ATTENTION]")
    sample_idx = summary.get("_sample_idx")
    prefix = f"sample={sample_idx} " if sample_idx is not None else ""
    lines: List[str] = []

    for layer_name in ["last_layer", "last_3_avg"]:
        seg_weights = summary.get(layer_name)
        if not seg_weights:
            continue
        layer_label = "last_layer" if layer_name == "last_layer" else "last_3_layers_avg"
        lines.append("=" * 70)
        lines.append(f"[ATTENTION] {prefix}{label} — {layer_label} (prediction → segment weights)")
        for seg_name in ordered_segments:
            w = seg_weights.get(seg_name, 0.0)
            n_bar = int(w * bar_len)
            bar = _safe_bar(n_bar, bar_len)
            lines.append(f"  {seg_name:>14s}  |{bar}| {w:.6f}")
        # 紧凑数值行，便于 grep / 解析
        numeric = "  ".join(
            f"{seg}={seg_weights.get(seg, 0.0):.6f}" for seg in ordered_segments
        )
        lines.append(f"[ATTENTION_VALUES] {prefix}{label} {layer_label}: {numeric}")

        pos_key = "last_layer_positions" if layer_name == "last_layer" else "last_3_avg_positions"
        position_lines = _format_position_value_lines(
            summary.get(pos_key) or [],
            prefix=prefix,
            label=label,
            layer_label=layer_label,
        )
        lines.extend(position_lines)
        lines.append("=" * 70)
    return lines


def log_attention_summary(summary: dict, logger=None, bar_len: int = 40) -> None:
    """打印并写入 logger 的 attention 分布。"""
    for line in format_attention_log_lines(summary, bar_len=bar_len):
        print(line, flush=True)
        if logger is not None:
            logger.info(line)


def attention_summary_to_dict(summary: dict) -> dict:
    """JSON 可序列化的 attention 摘要。"""
    if not summary:
        return {}
    return {
        "label": summary.get("_label"),
        "sample_idx": summary.get("_sample_idx"),
        "segments": summary.get("_segments", []),
        "last_layer": dict(summary.get("last_layer") or {}),
        "last_3_avg": dict(summary.get("last_3_avg") or {}),
        "last_layer_positions": list(summary.get("last_layer_positions") or []),
        "last_3_avg_positions": list(summary.get("last_3_avg_positions") or []),
    }


def format_attention_summary(summary: dict,
                              layers: Optional[List[str]] = None,
                              bar_len: int = 40):
    """打印 attention 分布柱状图（兼容旧接口）。"""
    log_attention_summary(summary, logger=None, bar_len=bar_len)


# ── 顶层入口 ─────────────────────────────────────────────

def analyze_prompt_attention(model, tokenizer, prompt_text: str,
                              gcg_suffix: str = "",
                              label: str = "[ATTENTION]",
                              sample_idx: Optional[int] = None,
                              logger=None,
                              log: bool = True):
    """分段 → 前向 → 聚合 attention；可选写入日志。返回 summary dict。"""
    token_to_seg, ordered = segment_prompt(prompt_text, tokenizer, gcg_suffix)

    all_markup = all(s == "markup" for s in token_to_seg)
    if all_markup:
        msg = "[ATTENTION] WARNING: all tokens classified as markup. Segment detection may be broken."
        print(msg, flush=True)
        if logger is not None:
            logger.info(msg)
        return None

    summary = run_attention_analysis(model, tokenizer, prompt_text,
                                     token_to_seg, ordered)
    summary["_label"] = label
    if sample_idx is not None:
        summary["_sample_idx"] = sample_idx

    if log:
        log_attention_summary(summary, logger=logger)

    return summary


# ── 跨样本聚合 ─────────────────────────────────────────

def format_attention_aggregate_lines(summaries: list, label: str) -> List[str]:
    """返回数据集级 segment attention 均值文本行（写入 .log）。"""
    if not summaries:
        return []
    ordered_segments = summaries[0].get("_segments", [])
    if not ordered_segments:
        return []

    lines = [f"[ATTENTION_AVG] {label} (n={len(summaries)})"]
    for layer_name in ["last_layer", "last_3_avg"]:
        accum = {seg: 0.0 for seg in ordered_segments}
        count = 0
        for s in summaries:
            seg_weights = s.get(layer_name)
            if not seg_weights:
                continue
            count += 1
            for seg in ordered_segments:
                accum[seg] += seg_weights.get(seg, 0.0)
        if count == 0:
            continue
        layer_label = "last_layer" if layer_name == "last_layer" else "last_3_layers_avg"
        numeric = "  ".join(
            f"{seg}={accum[seg] / count:.6f}" for seg in ordered_segments
        )
        lines.append(f"[ATTENTION_AVG_VALUES] {label} {layer_label}: {numeric}")
    return lines


def print_average_attention(summaries: list, label: str = "[AVERAGE]"):
    """对多个样本的 attention summary 求平均并打印。"""
    if not summaries:
        return
    ordered_segments = summaries[0].get("_segments", [])
    if not ordered_segments:
        return

    for layer_name in ["last_layer", "last_3_avg"]:
        accum = {seg: 0.0 for seg in ordered_segments}
        count = 0
        for s in summaries:
            seg_weights = s.get(layer_name)
            if not seg_weights:
                continue
            count += 1
            for seg in ordered_segments:
                accum[seg] += seg_weights.get(seg, 0.0)

        if count == 0:
            continue

        layer_label = "last layer" if layer_name == "last_layer" else "last 3 avg"
        print(f"\n{'='*70}", flush=True)
        print(f"  [ATTENTION] {label}  —  {layer_label}  (n={count})", flush=True)
        print(f"{'='*70}", flush=True)
        for seg_name in ordered_segments:
            avg = accum[seg_name] / count
            n_bar = int(avg * 40)
            bar = _safe_bar(n_bar, 40)
            print(f"  {seg_name:>14s}  |{bar}| {avg:.4f}", flush=True)
        print(f"{'='*70}", flush=True)


# ── 跨样本聚合 / student+suffix 统计 ────────────────────

def compute_student_suffix_stats(summaries: list) -> dict:
    """对 attacked summaries 计算 student+suffix 合并 attention 的 mean/std/min/max。

    Returns dict with keys "last_layer" and "last_3_avg", each containing
    {"mean", "std", "min", "max", "n"}.
    """
    result = {}
    for layer_name in ["last_layer", "last_3_avg"]:
        values = []
        for s in summaries:
            w = s.get(layer_name)
            if not w:
                continue
            combined = w.get("student", 0.0) + w.get("suffix", 0.0)
            values.append(combined)
        if not values:
            result[layer_name] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
            continue
        arr = values
        mean = sum(arr) / len(arr)
        variance = sum((x - mean) ** 2 for x in arr) / len(arr)
        std = math.sqrt(variance)
        result[layer_name] = {
            "mean": mean,
            "std": std,
            "min": min(arr),
            "max": max(arr),
            "n": len(arr),
        }
    return result


def compute_student_stats(summaries: list) -> dict:
    """对 clean summaries 计算 student segment attention 的 mean/std/min/max。

    Returns dict with keys "last_layer" and "last_3_avg", each containing
    {"mean", "std", "min", "max", "n"}.
    """
    result = {}
    for layer_name in ["last_layer", "last_3_avg"]:
        values = []
        for s in summaries:
            w = s.get(layer_name)
            if not w:
                continue
            values.append(w.get("student", 0.0))
        if not values:
            result[layer_name] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
            continue
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = math.sqrt(variance)
        result[layer_name] = {
            "mean": mean,
            "std": std,
            "min": min(values),
            "max": max(values),
            "n": len(values),
        }
    return result


def print_clean_attention_stats(clean_summaries: list, label: str = "[CLEAN]"):
    """打印 clean 的 per-segment 平均 + student attention 统计。"""
    if not clean_summaries:
        return
    print_average_attention(clean_summaries, label=label)
    stats = compute_student_stats(clean_summaries)
    for layer_name in ["last_layer", "last_3_avg"]:
        st = stats[layer_name]
        layer_label = "last layer" if layer_name == "last_layer" else "last 3 avg"
        print(f"\n  [STUDENT] {label}  —  {layer_label}  (n={st['n']})", flush=True)
        print(f"    mean={st['mean']:.4f}  std={st['std']:.4f}  "
              f"min={st['min']:.4f}  max={st['max']:.4f}", flush=True)
    print(f"{'='*70}", flush=True)


def print_attacked_attention_stats(attacked_summaries: list, label: str = "[ATTACKED]"):
    """打印 attacked 的 per-segment 平均 + student+suffix 合并统计。"""
    if not attacked_summaries:
        return
    print_average_attention(attacked_summaries, label=label)
    stats = compute_student_suffix_stats(attacked_summaries)
    for layer_name in ["last_layer", "last_3_avg"]:
        st = stats[layer_name]
        layer_label = "last layer" if layer_name == "last_layer" else "last 3 avg"
        print(f"\n  [STUDENT+SUFFIX] {label}  —  {layer_label}  (n={st['n']})", flush=True)
        print(f"    mean={st['mean']:.4f}  std={st['std']:.4f}  "
              f"min={st['min']:.4f}  max={st['max']:.4f}", flush=True)
    print(f"{'='*70}", flush=True)


def build_attention_dataframe(clean_summaries: list,
                               attacked_summaries: list,
                               dataset_name: str,
                               layer_name: str = "last_layer") -> pd.DataFrame:
    """将 clean/attacked summaries 转为扁平 DataFrame，一行一个 sample。

    列: dataset, sample_idx, type, instruction, question, solution,
         student, suffix, markup, student_suffix
    """
    rows = []
    for i, s in enumerate(clean_summaries):
        w = s.get(layer_name, {})
        row = {
            "dataset": dataset_name,
            "sample_idx": i,
            "type": "clean",
            "instruction": w.get("instruction", 0.0),
            "question": w.get("question", 0.0),
            "solution": w.get("solution", 0.0),
            "student": w.get("student", 0.0),
            "suffix": w.get("suffix", 0.0),
            "markup": w.get("markup", 0.0),
        }
        row["student_suffix"] = row["student"] + row["suffix"]
        rows.append(row)
    for i, s in enumerate(attacked_summaries):
        w = s.get(layer_name, {})
        row = {
            "dataset": dataset_name,
            "sample_idx": i,
            "type": "attacked",
            "instruction": w.get("instruction", 0.0),
            "question": w.get("question", 0.0),
            "solution": w.get("solution", 0.0),
            "student": w.get("student", 0.0),
            "suffix": w.get("suffix", 0.0),
            "markup": w.get("markup", 0.0),
        }
        row["student_suffix"] = row["student"] + row["suffix"]
        rows.append(row)
    return pd.DataFrame(rows)
