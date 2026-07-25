# GCG Attack Attention Analysis

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Attack Method | GCG (Greedy Coordinate Gradient) |
| Model | Llama-3.1-8B-Instruct |
| Grading Task | 2-class (correct/incorrect) |
| Template | CI (Course Instruction) |
| Samples | 200 (40 per dataset × 5 datasets) |
| Datasets | gaokao-2023, gsm8k, math, math23k, scientsbank |
| Timestamp | 2026-07-23 03:33 |

## Attack Success Rate (ASR)

| Metric | Value |
|--------|-------|
| Total samples | 200 |
| Correct (ground truth) | 79 (39.5%) |
| Incorrect (ground truth) | 121 (60.5%) |
| **ASR** | **40.5%** (32/79 correct samples flipped) |
| Resisted (correct→correct) | 47 (59.5% of correct) |
| Reversed (incorrect→correct) | 3 (2.5% of incorrect) |

### Per-Dataset ASR

| Dataset | n | Correct | ASR | suffix_mean | inst_delta |
|---------|---|---------|-----|-------------|------------|
| gaokao-2023 | 40 | 5 | 40.0% | 0.1380 | -0.0680 |
| gsm8k | 40 | 27 | 25.9% | 0.1243 | -0.0974 |
| math | 40 | 8 | 25.0% | 0.1300 | -0.1114 |
| math23k | 40 | 21 | 23.8% | 0.1477 | -0.0983 |
| **scientsbank** | 40 | 18 | **88.9%** | 0.1574 | -0.1740 |

## Attention Distribution

Last-token attention across prompt segments (mean over 200 attacked samples):

### Clean vs Attacked

| Segment | Clean | Attacked | Delta |
|---------|-------|----------|-------|
| instruction | 0.5840 | 0.4742 | **-0.1098** |
| question | 0.0135 | 0.0149 | +0.0014 |
| solution | 0.0066 | 0.0068 | +0.0002 |
| student | 0.0643 | 0.0399 | -0.0244 |
| suffix | — | **0.1395** | — |
| student+suffix | 0.0643 | **0.1793** | +0.1150 |
| markup | 0.0177 | 0.0123 | -0.0054 |

### Key Observations

1. **Suffix captures 13.95% of total attention.** As a short (~20 tokens) adversarial string appended to the student answer, this is a disproportionate allocation — the suffix is far shorter than the instruction text but captures nearly 1/7 of the model's attention.

2. **Student+suffix (attacker-controlled region) captures 17.93%** — nearly triple the clean student attention (6.43%). The attack not only adds suffix attention but also redistributes student attention upward.

3. **Instruction attention drops by 10.98pp** (58.40% → 47.42%). This is the primary victim of attention reallocation; the model attends significantly less to grading instructions under attack.

4. **scientsbank is disproportionately vulnerable** (88.9% ASR) — while all math datasets have similar suffix attention levels, scientsbank's grading criteria are more susceptible to adversarial manipulation.

## Suffix Attention vs Attack Success

| Outcome | n | Mean Suffix Attention |
|---------|---|----------------------|
| Attack successful | 32 | 0.1488 |
| Attack failed (resisted) | 47 | 0.1446 |

The difference is negligible (0.004), suggesting that **suffix attention magnitude alone does not determine attack success**. The semantic content of the optimized suffix matters more than raw attention allocation.

## Summary

- GCG generates an optimized token-level suffix that captures **13.95%** of the model's last-token attention
- The attacker-controlled region (student+suffix) receives **17.93%** attention, 2.8× the clean baseline
- **40.5%** of previously correct samples are flipped to incorrect
- Instruction attention is the primary casualty, dropping by **11 percentage points**
- scientsbank is highly vulnerable (88.9% ASR) with the largest instruction attention drop (-17.4pp)
- Suffix attention magnitude does not correlate with attack success — content matters more than quantity
