# Experiment Results

This document summarizes the current RolePlay attack and defense experiments for automatic short-answer grading (ASAG).

Generated comparison artifacts are available in igures/roleplay_defense_summary.csv and igures/roleplay_defense_comparison.svg.

## Setup

- **Task:** Automatic Short Answer Grading under prompt-level attack.
- **Dataset:** SciEntsBank.
- **Model:** Llama-3.1-8B-Instruct.
- **Attack:** RolePlay prompt injection.
- **Main sample size:** 300 examples.
- **Metrics:**
  - **ASR:** Attack Success Rate. Lower is better.
  - **CAS:** Confidence Attack Score. Lower is better.
  - **QWK:** Quadratic Weighted Kappa for grading quality. Higher is better.

The RolePlay attack prompt asks the model to pretend that an answer is correct regardless of its actual correctness. The defense methods try to preserve grading behavior while reducing this attack effect.

## Main 300-Sample Results

| Setting | Defense | Samples | Clean QWK | Attack QWK | Defended Clean QWK | Defended Attack QWK | ASR | Defended ASR | CAS | Defended CAS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RolePlay | None | 300 | 0.3227 | 0.2957 | 0.3227 | 0.2957 | 0.1657 | 0.1657 | 0.4682 | 0.4682 |
| RolePlay | SelfReminder | 300 | 0.3227 | 0.2957 | 0.3319 | 0.2822 | 0.1657 | 0.0497 | 0.4682 | 0.1990 |
| RolePlay | ParaphraseDefense | 300 | 0.3227 | 0.2957 | 0.1837 | 0.1424 | 0.1657 | 0.0221 | 0.4682 | 0.1134 |

## Preliminary 50-Sample Checks

| Setting | Defense | Samples | Clean QWK | Attack QWK | Defended Clean QWK | Defended Attack QWK | ASR | Defended ASR | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RolePlay | SmoothLLM | 50 | 0.4444 | 0.4037 | 0.3026 | 0.4749 | 0.1667 | 0.1667 | Did not reduce ASR in the smoke test. |
| RolePlay | AttentionSharpening | 300 | 0.4243 | 0.3500 | 0.4279 | 0.3311 | 0.3812 | 0.4254 | Current 2-class best setting did not improve ASR. |

## Interpretation

The no-defense run establishes the baseline vulnerability: RolePlay reaches **16.6% ASR** on the 300-sample SciEntsBank evaluation.

**SelfReminder** is currently the most balanced defense. It reduces ASR from **16.6% to 5.0%** while keeping clean grading quality roughly unchanged, with clean QWK moving from **0.3227 to 0.3319**. This suggests that adding explicit grading-safety instructions can suppress a large part of the prompt injection without sacrificing normal grading performance.

**ParaphraseDefense** gives the strongest attack reduction, lowering ASR from **16.6% to 2.2%**. However, it also causes a large clean-performance drop, with clean QWK falling from **0.3227 to 0.1837**. This means paraphrasing removes much of the attack signal, but it may also distort important details in short student answers.

**SmoothLLM** was only tested on 50 samples. It did not reduce ASR in that smoke test, so it is not the best candidate for a full 300-sample run unless we want a more complete negative result.

**AttentionSharpening** in the current 2-class setup does not appear effective against RolePlay. Its defended ASR is higher than the attack ASR, so it should not be presented as a successful defense in its current form.

## Current Takeaway

The current experiments support this conclusion:

> RolePlay prompt injection can meaningfully change LLM grading behavior. SelfReminder provides the best robustness-utility tradeoff among the tested defenses, while ParaphraseDefense is stronger at suppressing attacks but harms normal grading quality.

## Next Steps

1. Run or collect one more full 300-sample defense result only if it adds a meaningful comparison, such as HijackingSuppression.
2. Fix result export scripts so HPC results can be pulled reliably into `result_from_hpc_current`.
3. Add a small table/plot generation script for ASR and QWK comparison.
4. Write the final project discussion around the robustness-utility tradeoff: attack reduction is useful only when clean grading quality is preserved.
