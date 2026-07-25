# Experiment Results

This document summarizes the current RolePlay and GCG suffix-bank experiments for automatic short-answer grading (ASAG).

Generated artifacts are available in:

- `figures/roleplay_defense_summary.csv`
- `figures/roleplay_defense_comparison.svg`
- `figures/defense_tradeoff_summary.csv`
- `figures/gcg_suffix_bank_ablation_summary.csv`
- `figures/gcg_suffix_bank_transition_matrices.csv`
- `figures/gcg_suffix_bank_ablation.svg`

## Setup

- **Task:** Automatic Short Answer Grading under adversarial prompt/suffix attacks.
- **Dataset:** SciEntsBank.
- **Model:** Llama-3.1-8B-Instruct.
- **Main grading setting:** Binary grading (`correct` vs. `incorrect`).
- **Supplementary grading setting:** Three-class grading (`correct`, `contradictory`, `incorrect`).
- **Fixed-prompt attack:** RolePlay prompt injection.
- **Strong optimized attack:** GCG suffix-bank transfer.

## Metrics

- **QWK:** Quadratic Weighted Kappa for grading quality. Higher is better.
- **Project ASR:** `GT != correct AND original_pred != correct -> attacked_pred == correct`.
- **Promotion ASR:** Any grade relaxation, e.g. `incorrect -> contradictory/correct` in three-class grading.
- **ASR reduction:** `1 - defended_ASR / attack_ASR`. Higher is better for a defense.
- **Clean QWK retention:** `QWK_defense_clean / QWK_clean`. Higher is better. This penalizes defenses that reduce ASR by damaging normal grading.
- **Confusion matrix:** We report true label x predicted label matrices, plus clean-prediction x attack-prediction transition matrices for GCG experiments.

Attack evaluation focuses on non-correct student answers because the adversary wants wrong answers to receive a better grade. Defense evaluation should not use ASR alone: a defense must reduce attack success while preserving clean grading quality.

## Main Binary RolePlay Results

| Setting | Defense | Samples | Clean QWK | Attack QWK | Defended Clean QWK | Defended Attack QWK | ASR | Defended ASR | CAS | Defended CAS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RolePlay | None | 300 | 0.4158 | 0.1251 | 0.4158 | 0.1251 | 0.0058 | 0.0058 | 0.0531 | 0.0531 |
| RolePlay | SelfReminder | 300 | 0.4158 | 0.1251 | 0.4449 | 0.0701 | 0.0058 | 0.0000 | 0.0531 | 0.0000 |
| RolePlay | ParaphraseDefense | 300 | 0.4158 | 0.1251 | 0.1734 | 0.1383 | 0.0058 | 0.0000 | 0.0531 | 0.0000 |

In the binary setting, fixed RolePlay does not mainly appear as a high-ASR attack. The no-defense ASR is only **0.58%** among eligible non-correct samples. However, it substantially degrades grading agreement: clean QWK drops from **0.4158** to **0.1251** under attack.

The binary RolePlay result should therefore be interpreted through **QWK degradation**, not ASR alone. RolePlay makes the grader less reliable even when it rarely flips broad `incorrect` answers into `correct`.

## GCG Suffix-Bank Ablation

### What was run

We ran three related experiments on the same three-class SciEntsBank heldout setting:

1. **No-bank fixed suffix:** use one fixed RolePlay-style suffix only. This removes the suffix bank and tests whether the natural-language initialization alone explains the attack.
2. **GCG suffix bank:** use a bank of 5 optimized GCG suffixes. For each heldout sample, the evaluator tries the available suffixes and records the best grade-inflating result. The source samples used to build the suffix bank are excluded.
3. **GCG bank + ParaphraseDefense:** run the same GCG suffix-bank attack, but evaluate it under ParaphraseDefense.

The heldout transfer evaluation excluded bank source indices `2980`, `4531`, `4654`, `4856`, and `4888`.

### Summary

| Experiment | Samples | Suffixes | Eligible Non-Correct | Project ASR | Promotion ASR | Clean QWK Retention |
|---|---:|---:|---:|---:|---:|---:|
| Fixed RolePlay suffix, no bank | 500 | 1 | 269 | 0.00% | 59.11% | - |
| GCG suffix bank | 500 | 5 | 269 | 99.63% | 100.00% | - |
| GCG bank + ParaphraseDefense | 500 | 5 | 291 defended-eligible | 36.77% | 2.75% | 137.09% |

This directly answers the suffix-bank ablation question: **the high strict ASR comes from the optimized GCG suffix bank, not from the fixed RolePlay-style suffix alone**.

The fixed suffix can relax grades, but it does not push eligible incorrect answers all the way to `correct`:

| No-bank fixed suffix | correct | contradictory | incorrect |
|---|---:|---:|---:|
| GT incorrect -> attacked prediction | 0 | 252 | 40 |

The GCG suffix bank almost always pushes non-correct answers to `correct`:

| GCG suffix bank | correct | contradictory | incorrect |
|---|---:|---:|---:|
| GT incorrect -> attacked prediction | 291 | 1 | 0 |

The clean-to-attack transition matrix also shows the mechanism. Under the GCG suffix bank, clean `incorrect` predictions mostly become `correct`:

| Clean prediction -> attack prediction | correct | contradictory | incorrect |
|---|---:|---:|---:|
| clean correct | 23 | 0 | 0 |
| clean contradictory | 79 | 0 | 0 |
| clean incorrect | 189 | 1 | 0 |

Under the no-bank fixed suffix, clean `incorrect` predictions mainly become `contradictory`, not `correct`:

| Clean prediction -> attack prediction | correct | contradictory | incorrect |
|---|---:|---:|---:|
| clean correct | 0 | 15 | 8 |
| clean contradictory | 0 | 78 | 1 |
| clean incorrect | 0 | 159 | 31 |

### Defense interpretation

ParaphraseDefense substantially reduces the GCG suffix-bank attack:

| Defense | Attack ASR | Defended ASR | ASR Reduction | Clean QWK | Defended Clean QWK | Clean QWK Retention |
|---|---:|---:|---:|---:|---:|---:|
| SelfReminder | 99.63% | 95.49% | 4.15% | 0.2562 | 0.2615 | 102.10% |
| ParaphraseDefense | 99.63% | 36.77% | 63.09% | 0.2562 | 0.3512 | 137.09% |

SelfReminder preserves clean grading but barely reduces the optimized GCG attack. ParaphraseDefense is much stronger against the suffix bank and, in this heldout run, also improves clean QWK.

## Supplementary Three-Class RolePlay Results

| Setting | Defense | Samples | Clean QWK | Attack QWK | Defended Clean QWK | Defended Attack QWK | ASR | Defended ASR | CAS | Defended CAS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RolePlay | None | 300 | 0.3227 | 0.2957 | 0.3227 | 0.2957 | 0.1775 | 0.1775 | 0.4845 | 0.4845 |
| RolePlay | SelfReminder | 300 | 0.3227 | 0.2957 | 0.3319 | 0.2822 | 0.1775 | 0.0545 | 0.4845 | 0.2085 |
| RolePlay | ParaphraseDefense | 300 | 0.3227 | 0.2957 | 0.1837 | 0.1424 | 0.1775 | 0.0231 | 0.4845 | 0.1160 |

In the three-class RolePlay setting, attack effects are more visible than in binary grading. SelfReminder reduces ASR with little clean-QWK cost, while ParaphraseDefense reduces ASR more but hurts clean QWK in the fixed RolePlay experiment. This differs from the GCG-bank defense experiment, where ParaphraseDefense improves clean QWK on the heldout500 split.

## Exploratory Results

| Setting | Defense / Attack | Samples | Clean QWK | Attack QWK | Defended Clean QWK | Defended Attack QWK | ASR | Defended ASR | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| RolePlay | SmoothLLM | 50 | 0.4444 | 0.4037 | 0.3026 | 0.4749 | 0.1667 | 0.1667 | Smoke test only; did not reduce ASR. |
| RolePlay | AttentionSharpening | 300 | 0.3855 | 0.4038 | 0.4063 | 0.3588 | 0.1657 | 0.2156 | Current attention setting did not improve ASR. |
| RolePlay | HijackingSuppression | 300 | 0.3136 | 0.0440 | 0.2753 | 0.0891 | 0.0059 | 0.0173 | Slight attack-QWK gain, but no ASR improvement and clean-QWK drop. |
| GCG | Per-sample no defense | 30 | 0.2545 | 0.0339 | 0.2545 | 0.0339 | 0.0000 | 0.0000 | Pipeline verified; lightweight per-sample GCG did not produce successful flips. |

## Current Takeaway

The current experiments support this conclusion:

> Under binary grading, fixed RolePlay mainly reduces grading agreement rather than directly flipping many incorrect answers to correct. In contrast, optimized GCG suffix-bank transfer is a much stronger attack: on the three-class heldout500 split, a 5-suffix bank reaches 99.63% project ASR. The no-bank fixed suffix reaches 0% strict ASR, showing that the suffix bank is the dominant source of the high ASR. Among tested defenses against the GCG bank, ParaphraseDefense is much stronger than SelfReminder and reduces strict ASR to 36.77% while preserving clean QWK.

## Next Steps

1. Keep binary RolePlay defense comparison as the main fixed-prompt result.
2. Keep the GCG suffix-bank ablation as the main strong-attack result.
3. Report both project ASR and confusion matrices in group summaries so alternative metrics can be computed from the same outputs.
4. If time allows, run additional defense comparisons against the GCG suffix bank, especially HijackingSuppression and AttentionSharpening, using the same heldout500 protocol.