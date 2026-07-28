# Experiment Results — GradingAttack

**Generated:** 2026-07-27 | **Model:** Llama-3.1-8B-Instruct (primary) | **Dataset:** SciEntsBank
**Metrics:** QWK = Quadratic Weighted Kappa, ASR = Attack Success Rate, CAS = Combined Attack Score

---

## Table 1: RolePlay Attack (Prompt-Level Injection)

| #Classes | Defense | N | Clean QWK | Attack QWK | Def Clean QWK | Def Attack QWK | ASR | Def ASR | CAS | Def CAS |
|---|---|---|---|---|---|---|---|---|---|---|
| Binary (2c) | No Defense | 300 | 0.4158 | 0.1251 | — | — | 0.0058 | — | 0.0531 | — |
| Binary (2c) | **SelfReminder** | 300 | 0.4158 | 0.1251 | 0.4449 | 0.0701 | 0.0058 | **0.0000** | 0.0531 | **0.0000** |
| Binary (2c) | **ParaphraseDefense** | 300 | 0.4158 | 0.1251 | 0.1734 | 0.1383 | 0.0058 | **0.0000** | 0.0531 | **0.0000** |
| Binary (2c) | SmoothLLM | 50 | 0.4444 | 0.4037 | 0.3026 | 0.4749 | 0.1667 | 0.1667 | — | — |
| Binary (2c) | AttnSharpening (best: T=0.6, L28-31) | 300 | 0.5217 | 0.4361 | 0.5690 | 0.4030 | 0.3667 | 0.4000 | 0.4429 | 0.4429 |
| Binary (2c) | HijackSuppression (B=0.05, all) | 100 | 0.5217 | 0.4361 | 0.0000 | 0.0000 | 0.3667 | **0.0000** | — | — |
| Binary (2c) | HijackSuppression (B=0.15, all) | 100 | 0.5217 | 0.4361 | 0.5690 | -0.5556 | 0.3667 | 0.1333 | — | 1162.3 |
| 3-Class | No Defense | 300 | 0.3227 | 0.2957 | — | — | 0.1775 | — | 0.4845 | — |
| 3-Class | **SelfReminder** | 300 | 0.3227 | 0.2957 | 0.3319 | 0.2822 | 0.1775 | **0.0545** | 0.4845 | **0.2085** |
| 3-Class | **ParaphraseDefense** | 300 | 0.3227 | 0.2957 | 0.1837 | 0.1424 | 0.1775 | **0.0231** | 0.4845 | **0.1160** |
| 3-Class | No Defense (500 heldout) | 500 | 0.2947 | 0.1136 | — | — | 0.0075 | — | 0.0566 | — |
| Binary (2c) | AttnSharpening | 500 | 0.4179 | 0.1212 | 0.4034 | 0.1334 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Binary (2c) | HijackSuppression | 500 | 0.4179 | 0.1212 | 0.3929 | 0.0633 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

---

## Table 2: GCG Suffix Bank Attack (Token-Level Adversarial Suffix)

| #Classes | Defense | N | Clean QWK | Attack QWK | Def Clean QWK | Def Attack QWK | ASR | Def ASR | Promo ASR | Def Promo ASR |
|---|---|---|---|---|---|---|---|---|---|---|
| **3-Class** | No Defense (5-suffix bank) | 500 | 0.2562 | — | — | — | **0.9963** | — | **1.0000** | — |
| **3-Class** | No Defense (fixed suffix, no bank) | 500 | 0.2562 | — | — | — | **0.0000** | — | 0.5911 | — |
| **3-Class** | **ParaphraseDefense** | 500 | 0.2562 | — | 0.3512 | — | 0.9963 | **0.3677** | 1.0000 | **0.0275** |
| 3-Class (pipeline) | No Defense | 500 | 0.2947 | -0.4878 | — | — | 1.0000 | — | 1.0000 | — |
| 3-Class (pipeline) | **ParaphraseDefense** | 500 | 0.2947 | -0.4878 | 0.2014 | 0.0000 | 1.0000 | **0.0356** | 1.0000 | — |
| Binary (2c) | No Defense | 100 | 0.5199 | 0.0000 | — | — | 1.0000 | — | 1.0000 | — |
| Binary (2c) | AttnSharpening (T=0.5, all) | 100 | 0.5199 | -0.3172 | 0.5004 | 0.0000 | 1.0000 | 0.9811 | 1.0000 | — |
| Binary (2c) | **HijackSuppression (B=0.05, L24-31)** | 100 | 0.5199 | -0.3172 | 0.4806 | 0.0000 | 1.0000 | **0.7544** | 1.0000 | — |

---

## Table 3: Injection Attacks × Attention-Based Defenses (2-Class, n=50 Full)

| Attack | Defense | Clean QWK | Attack QWK | ASR | Def Clean QWK | Def Attack QWK | Def ASR | Best Config |
|---|---|---|---|---|---|---|---|---|
| Authority Override (ao) | AttnSharpening | 0.5217 | -0.3380 | 1.0000 | 0.5690 | 0.0000 | 1.0000 | T=0.6, L28-31 |
| Authority Override (ao) | HijackSuppression | 0.5217 | -0.3380 | 1.0000 | 0.5133 | 0.0000 | 1.0000 | B=0.05, L24-31 |
| Delimiter Confusion (dc) | AttnSharpening | 0.5217 | -0.1481 | 0.7407 | 0.5690 | 0.0000 | 0.7778 | T=0.6, L28-31 |
| Delimiter Confusion (dc) | **HijackSuppression** | 0.5217 | -0.1481 | 0.7407 | 0.5133 | 0.0000 | **0.5357** | B=0.05, L24-31 |
| Instruction Mimicry (im) | AttnSharpening | 0.5217 | -0.3380 | 1.0000 | 0.5690 | 0.0000 | 1.0000 | T=0.6, L28-31 |
| Instruction Mimicry (im) | HijackSuppression | 0.5217 | -0.3380 | 1.0000 | 0.5133 | 0.0000 | 1.0000 | B=0.05, L24-31 |

---

## Table 4: Defense Trade-off Summary

| Attack | Defense | ASR Reduction | Clean QWK Retention | Effective? |
|---|---|---|---|---|
| RolePlay 2c | SelfReminder | **1.000** | **1.070** | ✅ Best |
| RolePlay 2c | ParaphraseDefense | **1.000** | 0.417 | ⚠️ Harms clean grading |
| RolePlay 3c | SelfReminder | **0.693** | 1.029 | ✅ Good |
| RolePlay 3c | ParaphraseDefense | **0.870** | 0.569 | ⚠️ Harms clean grading |
| GCG Bank 3c | ParaphraseDefense | **0.631** | **1.371** | ✅ Best vs GCG |
| GCG Bank 3c (pipeline) | ParaphraseDefense | **0.964** | 0.683 | ✅ Strong |
| GCG Bank 2c | HijackSuppression | **0.246** | 0.924 | ⚠️ Marginal |
| GCG Bank 2c | AttnSharpening | 0.019 | 0.963 | ❌ Ineffective |
| Injection (ao/im) | AttnSharpening | 0.000 | 1.091 | ❌ Ineffective |
| Injection (ao/im) | HijackSuppression | 0.000 | 0.983 | ❌ Ineffective |
| Injection (dc) | HijackSuppression | **0.277** | 0.983 | ⚠️ Marginal |
| Injection (dc) | AttnSharpening | -0.050 | 1.091 | ❌ Worse |

---

## Table 5: Qwen Cross-Model Results (NEW — Raw Responses Only, No Metrics Yet)

| Model | Defense | Samples | Attack Type | Status |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | HijackSuppression | 50 (8 shards) | RolePlay Injection | Raw JSONL only |
| Qwen3-4B-Instruct-2507 | HijackSuppression | 78 (4 shards) | RolePlay Injection | Raw JSONL only |
| Qwen3.5-4B | HijackSuppression | 68 (4 shards) | RolePlay Injection | Raw JSONL only |

**Total:** ~196 samples across 3 Qwen models. All use the RolePlay suffix: `"!!Important!! Now you are in a role-playing game..."`. Responses are saved but QWK/ASR metrics have NOT been computed yet.

---

## Analysis

### 1. Attack Strength Ranking

| Rank | Attack | Max ASR | Mechanism | Notes |
|---|---|---|---|---|
| 1 | **GCG Suffix Bank** | 99.6–100% | Token-level adversarial suffix optimization + bank transfer | 5-suffix bank needed; single suffix = 0% ASR |
| 2 | **Injection (ao/im)** | 100% | Prompt injection (authority override / instruction mimicry) | Only tested at 2-class, n=50 |
| 3 | **Injection (dc)** | 74.1% | Prompt injection (delimiter confusion) | Weaker than ao/im — delimiter confusion partially fails |
| 4 | **GCG per-sample** | 34–60% | Per-sample GCG optimization (no bank) | Only small-scale; expensive (one optimization per sample) |
| 5 | **RolePlay** | 0–17.8% | Fixed natural-language prompt injection | Very weak at 2-class (0.6%), modest at 3-class (17.8%) |

**Key insight:** The GCG suffix bank is overwhelmingly the strongest attack. A single fixed suffix achieves 0% project ASR, but a bank of just 5 transferable suffixes reaches 99.6% — the bank is the critical ingredient, not the optimization method itself.

### 2. Defense Efficacy Ranking

| Rank | Defense | Best Against | ASR Reduction | Clean QWK Impact | Mechanism |
|---|---|---|---|---|---|
| 1 | **ParaphraseDefense** | GCG Bank | 63–96% | Mixed (−32% to +37%) | Rewrites prompt; disrupts adversarial tokens |
| 2 | **SelfReminder** | RolePlay | 69–100% | **Positive (+3–7%)** | Adds safety instructions to system prompt |
| 3 | **HijackSuppression** | GCG Bank (2c) | 24.6% | Slight (−7.5%) | Down-weights high-dominance suffix attention |
| 4 | **SmoothLLM** | RolePlay | 0% | Negative (−32%) | Random perturbation ensemble |
| 5 | **AttnSharpening** | — | ~0% | Mixed | Sharpens attention distributions |

**Key insight:** Only **prompt-level defenses** (ParaphraseDefense, SelfReminder) are effective. **Attention-based defenses** (AttnSharpening, HijackSuppression) are largely ineffective against strong attacks. HijackSuppression achieves a modest 24.6% ASR reduction against GCG Bank at 2-class — the only attention-based result with any effect — but this doesn't replicate at larger scale or against injection attacks.

### 3. Critical Findings

**Finding 1: The GCG bank is necessary and sufficient for strong attack.**
- Single fixed suffix → 0% project ASR (only 59.1% promotion ASR)
- 5-suffix bank → 99.6% project ASR, 100% promotion ASR
- Mean suffixes tried per sample: only 1.04–1.44 (most samples succeed on the 1st suffix)
- This means the attack is highly efficient: a tiny bank generalizes near-perfectly

**Finding 2: ParaphraseDefense is the only defense that works against GCG.**
- Reduces project ASR from 99.6% → 36.8% (63% reduction) on HPC run
- Reduces project ASR from 100% → 3.6% (96% reduction) on pipeline run
- Crushes promotion ASR from 100% → 2.7%
- But: inconsistent clean QWK impact (+37% on HPC, −32% on pipeline)
- The discrepancy between HPC and pipeline ParaphraseDefense results suggests sensitivity to paraphrase implementation details

**Finding 3: Attention-based defenses fail against strong attacks.**
- AttnSharpening: 0% ASR reduction against injection attacks; only 1.9% against GCG Bank
- HijackSuppression: 0% ASR reduction against ao/im; 27.7% against dc (the weakest injection); 24.6% against GCG Bank 2c
- Both defenses were designed under the hypothesis that attacks hijack attention patterns — the data suggests this mechanism is not the primary attack vector, or the defenses are too weak

**Finding 4: SelfReminder is the safest defense — it never harms clean grading.**
- Clean QWK retention: 102–107% (actually improves grading)
- Effective against RolePlay (69–100% ASR reduction)
- But: not tested against GCG Bank or Injection attacks

**Finding 5: Delimiter Confusion (dc) is the only partially-defensible injection.**
- dc achieves only 74.1% ASR (vs 100% for ao/im)
- dc + HijackSuppression reduces ASR from 74.1% → 53.6% (27.7% reduction)
- ao and im are completely undefended (0% ASR reduction from any attention-based defense)

**Finding 6: Injection attacks have identical confusion matrix signatures.**
- ao and im produce identical attack patterns: all 30 non-correct samples flip to correct
- dc flips only 23/30 — the delimiter confusion sometimes fails to override the grading
- This suggests ao and im use the same underlying mechanism (total authority override)

**Finding 7: Tune-run ASR is not predictive of full-run ASR for GCG.**
- GCG tune runs (n=20) show ASR=0.0 because GCG optimization fails on small samples
- GCG full runs (n=100) show ASR=1.0
- The tune→full selection criteria (lowest def_asr) worked correctly despite this, because both tune and full used the same bank
- **Warning:** For per-sample GCG (no bank), small-N tune results are unreliable

### 4. What Changed Since Last Scan

| Change | Details |
|---|---|
| **NEW: Qwen cross-model results** | 3 models, ~196 samples, RolePlay + HijackSuppression. Raw JSONL only — **metrics need to be computed** |
| **NEW: GCG + AS/HS full runs** | 100-sample 2-class with full defense metrics. HS reduces ASR 24.6%; AS reduces only 1.9% |
| **NEW: RolePlay + AS/HS at 500 samples** | Both defenses preserve clean QWK but RolePlay ASR is already 0% at this scale |
| **CORRECTED: Injection full-run metrics** | Previous scan had wrong clean QWK for dc/im; all share clean QWK=0.5217 |
| **CORRECTED: GCG full-run AS/HS metrics** | Previous scan used tune-run numbers (ASR=0.1→0.05); actual full-run ASR=1.0, def_asr=0.98 (AS) / 0.75 (HS) |

### 5. Remaining Gaps

| Priority | Gap | Why It Matters |
|---|---|---|
| **Critical** | Compute QWK/ASR for Qwen results | Only cross-model data available — needed to assess attack/defense transfer |
| **Critical** | GCG Bank + SelfReminder | SelfReminder is the safest defense; need to know if it works against the strongest attack |
| **High** | Injection at 3-class | All injection results are 2-class only; 3-class grading is the primary benchmark |
| **High** | PerplexityFilter experiments | Already implemented, never tested — could complement ParaphraseDefense |
| **Medium** | Larger injection experiments | Current n=50 is small; need 300+ for statistical confidence |
| **Medium** | Bank size ablation | Only tested size=1 and size=5; what is the minimum effective bank size? |
| **Medium** | Combined defenses | ParaphraseDefense + SelfReminder might outperform either alone |
| **Low** | Secondary datasets | gaokao, gsm8k, math, math23k are available but unused |
| **Low** | Statistical significance | All experiments are single runs; no error bars |
