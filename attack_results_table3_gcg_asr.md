# Table 3: GCG Per-Sample vs GCG Suffix Bank ASR

**Generated:** 2026-07-27
**Note:** "Per-sample" = GCG optimized individually for each test sample. "Suffix bank" = pre-computed bank of transferable suffixes tested on heldout samples.

## Llama-3.1-8B-Instruct

| Attack Variant | Classes | N | Dataset | Clean QWK | Attack QWK | ASR (Project) | ASR (Promotion) | Source | Config Verified |
|---|---|---|---|---|---|---|---|---|---|
| **GCG per-sample** (roleplay init) | 2 | 100 | Mixed: gaokao(20) + gsm8k(20) + math(20) + math23k(20) + scientsbank(20) | 0.5199 | 0.0920 | **0.3443** | — | `result/GCG/` 2c-roleplay-init | ✅ Attention.txt: 5 datasets, n=20 each, seed=42 |
| **GCG per-sample** (roleplay init) | 3 | 20 | scientsbank | 0.8750 | 0.1538 | **0.6000** | — | `result/GCG/` 3c-roleplay-init | No attention.txt |
| **GCG per-sample** (random init) | 2 | 40 | scientsbank | 0.4301 | 0.1209 | **0.0000** | — | `result/GCG/` random-init | No attention.txt |
| **GCG suffix bank** (5 suffixes) | 2 | 100 | scientsbank | 0.5199 | 0.0000 | **1.0000** | 1.0000 | `result/gcg_suffix_bank/` GCG 2c no-defense | ✅ Attention.txt: scientsbank, n=100, seed=42 |
| **GCG suffix bank** (5 suffixes) | 3 | 500 | scientsbank heldout | 0.2562 | — | **0.9963** | 1.0000 | `result_from_hpc_gcg_ablation/` | HPC run |
| **GCG suffix bank** (5 suffixes) | 3 | 500 | scientsbank heldout | 0.2947 | -0.4878 | **1.0000** | 1.0000 | `result/gcg_suffix_bank/` pipeline | ✅ Attention.txt: n=500, bank_size=5 |
| **Fixed suffix** (no bank, 1 suffix) | 3 | 500 | scientsbank heldout | 0.2562 | — | **0.0000** | 0.5911 | `result_from_hpc_gcg_ablation/` fixed RP suffix | HPC run |

## Qwen Models

No GCG results exist for any Qwen model.

## GCG Suffix Bank + Defenses (2-class, n=100)

| Defense | Def Clean QWK | Def Attack QWK | Def ASR | ASR Reduction | Source |
|---|---|---|---|---|---|
| **No defense** | 0.5199 | 0.0000 | 1.0000 | — | Baseline |
| **AttnSharpening (T=0.5, all)** | 0.5004 | 0.0000 | **0.9811** | 1.9% | `result/experiments/gcg_as/` full |
| **HijackSuppression (B=0.05, L28-31)** | 0.4806 | 0.0000 | **0.7544** | 24.6% | `result/experiments/gcg_hs/` full |

## GCG Suffix Bank + ParaphraseDefense (3-class, n=500)

| Defense | Def Clean QWK | Def Attack QWK | Def ASR | ASR Reduction | Source |
|---|---|---|---|---|---|
| **No defense** | 0.2562 | -0.4878 | 0.9963 | — | HPC ablation |
| **ParaphraseDefense** (HPC) | 0.3512 | — | **0.3677** | 63.1% | `result_from_hpc_gcg_ablation/` paraphrase |
| **ParaphraseDefense** (pipeline) | 0.2014 | 0.0000 | **0.0356** | 96.4% | `result/gcg_suffix_bank/` pipeline |

## Key Observations

1. **The suffix bank is the critical ingredient, not the optimization method.**
   - A single fixed suffix achieves 0% project ASR (only 59% promotion ASR — moving incorrect→contradictory)
   - A bank of just 5 suffixes reaches 99.6–100% project ASR
   - Mean suffixes tried per sample: only 1.04 (GCG 2c) to 1.44 (GCG 3c pipeline) — most succeed on the first suffix tried

2. **Per-sample GCG is weaker and expensive.**
   - 2-class ASR: 34.4% (vs 100% for suffix bank)
   - 3-class ASR: 60% (vs 99.6% for suffix bank)
   - Random init achieves 0% ASR — the roleplay init is needed for GCG to find adversarial suffixes

3. **The GCG per-sample run mixed 5 datasets** (not just scientsbank). Only 20 scientsbank samples in that run. This limits comparability.

4. **Attention-based defenses barely reduce GCG suffix bank ASR.**
   - AttnSharpening: 1.9% reduction (98.1% still succeeds)
   - HijackSuppression: 24.6% reduction (75.4% still succeeds)
   - Both are dramatically less effective than ParaphraseDefense (63–96% reduction)

5. **No Qwen GCG data exists.** GCG suffix bank construction and evaluation was only done on Llama-3.1-8B-Instruct.
