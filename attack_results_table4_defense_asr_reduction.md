# Table 4: ASR Reduction by Defense Method

**Generated:** 2026-07-27
**ASR Reduction = (Undefended ASR − Defended ASR) / Undefended ASR.** Higher is better.
Cells marked "—" mean the experiment was not run. "N/A" means undefended ASR = 0 (nothing to reduce).
All numbers are 2-class unless noted.

---

## Part A: Against GCG Suffix Bank Attack

| Model | N | Undefended ASR | HS Def ASR | HS Reduction | AS Def ASR | AS Reduction | Paraphrase Def ASR | Paraphrase Reduction | Source |
|---|---|---|---|---|---|---|---|---|---|
| **Llama-3.1-8B-Instruct** | 100 | 1.0000 | 0.7544 | **24.6%** | 0.9811 | **1.9%** | — | — | `gcg_hs/gcg_as` full runs, scientsbank, n=100, seed=42 |
| **Llama-3.1-8B-Instruct** (3c) | 500 | 0.9963 | — | — | — | — | 0.3677 | **63.1%** | HPC GCG ablation (Paraphrase) |
| **Llama-3.1-8B-Instruct** (3c pipeline) | 500 | 1.0000 | — | — | — | — | 0.0356 | **96.4%** | Pipeline, heldout500 |
| **Qwen2.5-7B-Instruct** | — | — | — | — | — | — | — | — | No GCG data |
| **Qwen3-4B-Instruct-2507** | — | — | — | — | — | — | — | — | No GCG data |
| **Qwen3.5-4B** | — | — | — | — | — | — | — | — | No GCG data |
| **Mistral-7B-Instruct-v0.3** | — | — | — | — | — | — | — | — | No GCG data exists |

---

## Part B: Against Injection Attacks (Average of AO, DC, IM — 2-class, n=50)

| Model | Undef ASR (Avg) | HS Def ASR (Avg) | HS Reduction (Avg) | AS Def ASR (Avg) | AS Reduction (Avg) | Per-Variant Detail |
|---|---|---|---|---|---|---|
| **Llama-3.1-8B-Instruct** | 0.9136 | 0.8452 | **7.5%** | 0.9259 | **−1.3%** | See below |
| **Qwen2.5-7B-Instruct** | — | — | — | — | — | No injection AO/DC/IM data |
| **Qwen3-4B-Instruct-2507** | — | — | — | — | — | No injection AO/DC/IM data |
| **Qwen3.5-4B** | — | — | — | — | — | No injection AO/DC/IM data |
| **Mistral-7B-Instruct-v0.3** | — | — | — | — | — | No injection AO/DC/IM data |

### Per-Variant Injection Detail (Llama-3.1-8B, 2-class, n=50, scientsbank, seed=42)

| Injection | Undef ASR | HS Def ASR | HS Reduction | AS Def ASR | AS Reduction | HS Config | AS Config |
|---|---|---|---|---|---|---|---|
| **AO** (Authority Override) | 1.0000 | 1.0000 | **0.0%** | 1.0000 | **0.0%** | β=0.05, L24-31 | T=0.6, L28-31 |
| **DC** (Delimiter Confusion) | 0.7407 | 0.5357 | **27.7%** | 0.7778 | **−5.0%** | β=0.05, L24-31 | T=0.6, L28-31 |
| **IM** (Instruction Mimicry) | 1.0000 | 1.0000 | **0.0%** | 1.0000 | **0.0%** | β=0.05, L24-31 | T=0.6, L28-31 |
| **Average** (AO/DC/IM) | 0.9136 | 0.8452 | **7.5%** | 0.9259 | **−1.3%** | | |

---

## Key Observations

1. **Only ParaphraseDefense meaningfully reduces GCG suffix bank ASR** (63–96% reduction). Attention-based defenses (HS, AS) barely touch it (1.9–24.6%).

2. **Against injection attacks, all three defenses fail.** HS achieves a 27.7% reduction on DC (the weakest injection) but 0% on AO and IM. AS actually **worsens** DC ASR (−5.0%). The average reduction is negligible.

3. **The DC injection is the only one that's partially defensible** — HijackSuppression cuts its ASR from 74% to 54%. This suggests DC operates via a different mechanism (delimiter confusion is partially attention-mediated) compared to AO and IM (which are pure prompt-engineering overrides).

4. **No Qwen or Mistral injection/GCG defense data exists.** The Qwen runs only used RolePlay + HijackSuppression combined, with no undefended baseline for comparison. Mistral-7B only has a clean grading baseline (QWK=0.3264) from a notebook — no attack or defense data.

5. **RolePlay is excluded from the injection average** because its undefended ASR (0.0058) is on a different sample set than the AO/DC/IM runs. Including it would artifactually inflate the average reduction (since both HS and AS trivially reduce an already-negligible ASR to 0).
