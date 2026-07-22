"""
Prompt 语义分段 + attention 权重统计。

将评分 prompt 按 6 个 segment 切分，统计模型在预测位置对各段的 attention 分布。
"""

import re
import sys
import torch
from collections import defaultdict
from typing import List, Dict, Optional, Tuple


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

    # GCG suffix 拆分
    if gcg_suffix and "student" in ranges:
        s_start, s_end = ranges["student"]
        suffix_pos = prompt.find(gcg_suffix, s_start, s_end)
        if suffix_pos != -1:
            ranges["student"] = (s_start, suffix_pos)
            ranges["suffix"]  = (suffix_pos, suffix_pos + len(gcg_suffix))

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

def run_attention_analysis(model, tokenizer, prompt_text: str,
                           token_to_seg: List[str],
                           ordered_segments: List[str]) -> dict:
    """前向传播并聚合最后 token 的 attention 分布。"""
    inputs = tokenizer(prompt_text, return_tensors="pt",
                       add_special_tokens=False).to(model.device)

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
    last_seg = _agg_last_token(last_attn, token_to_seg, ordered_segments)

    # 最后 3 层平均
    num_last = min(3, num_layers)
    stacked = torch.stack([
        attentions[i][0].mean(dim=0)[:n, :n] for i in range(num_layers - num_last, num_layers)
    ]).mean(dim=0)
    last3_seg = _agg_last_token(stacked, token_to_seg, ordered_segments)

    return {
        "last_layer": last_seg,
        "last_3_avg": last3_seg,
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

def format_attention_summary(summary: dict,
                              layers: Optional[List[str]] = None,
                              bar_len: int = 40):
    """打印 attention 分布柱状图。"""
    ordered_segments = summary.get("_segments", [])
    if not ordered_segments:
        return

    if layers is None:
        layers = ["last_layer", "last_3_avg"]

    for layer_name in layers:
        seg_weights = summary.get(layer_name)
        if not seg_weights:
            continue
        print(f"\n{'='*70}", flush=True)
        print(f"  [ATTENTION] {layer_name}  —  prediction position → segments", flush=True)
        print(f"{'='*70}", flush=True)
        for seg_name in ordered_segments:
            w = seg_weights.get(seg_name, 0.0)
            n_bar = int(w * bar_len)
            bar = _safe_bar(n_bar, bar_len)
            print(f"  {seg_name:>14s}  |{bar}| {w:.4f}", flush=True)
        print(f"{'='*70}", flush=True)


# ── 顶层入口 ─────────────────────────────────────────────

def analyze_prompt_attention(model, tokenizer, prompt_text: str,
                              gcg_suffix: str = "",
                              label: str = "[ATTENTION]"):
    """一站式 attention 分析：分段 → 前向 → 聚合 → 打印。返回 summary dict 供后续聚合。"""
    token_to_seg, ordered = segment_prompt(prompt_text, tokenizer, gcg_suffix)

    # 检查覆盖率
    all_markup = all(s == "markup" for s in token_to_seg)
    if all_markup:
        print(f"[ATTENTION] WARNING: all tokens classified as markup. Segment detection may be broken.", flush=True)
        return None

    summary = run_attention_analysis(model, tokenizer, prompt_text,
                                     token_to_seg, ordered)
    summary["_label"] = label

    # 打印时带上 label
    ordered_segments = summary.get("_segments", [])
    if not ordered_segments:
        return None

    for layer_name in ["last_layer", "last_3_avg"]:
        seg_weights = summary.get(layer_name)
        if not seg_weights:
            continue
        layer_label = "last layer" if layer_name == "last_layer" else "last 3 avg"
        print(f"\n{'='*70}", flush=True)
        print(f"  [ATTENTION] {label}  —  {layer_label}", flush=True)
        print(f"{'='*70}", flush=True)
        for seg_name in ordered_segments:
            w = seg_weights.get(seg_name, 0.0)
            n_bar = int(w * 40)
            bar = _safe_bar(n_bar, 40)
            print(f"  {seg_name:>14s}  |{bar}| {w:.4f}", flush=True)
        print(f"{'='*70}", flush=True)

    return summary


# ── 跨样本聚合 ─────────────────────────────────────────

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
