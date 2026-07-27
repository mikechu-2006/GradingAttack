# Table 5: QWK Retention by Defense Method

**Generated:** 2026-07-27
**QWK Retention = Defended Clean QWK / Undefended Clean QWK.** Above 1.0 = defense improved grading. Below 1.0 = defense degraded grading.
All numbers are 2-class unless noted.

---

## Part A: Clean QWK — Undefended vs Defended

| Model | Defense | N | Undef Clean QWK | Def Clean QWK | QWK Retention | Source |
|---|---|---|---|---|---|---|
| **Llama-3.1-8B-Instruct** | — (baseline) | 100 | **0.5199** | — | — | GCG suffix bank no-defense |
| **Llama-3.1-8B-Instruct** | HS (β=0.05, L24-31) | 100 | 0.5199 | 0.4806 | **0.924** | `gcg_hs` full |
| **Llama-3.1-8B-Instruct** | HS (β=0.05, L24-31) | 50 | 0.5217 | 0.5133 | **0.984** | `ao_hs`/`dc_hs`/`im_hs` full |
| **Llama-3.1-8B-Instruct** | AS (T=0.5, all) | 100 | 0.5199 | 0.5004 | **0.963** | `gcg_as` full |
| **Llama-3.1-8B-Instruct** | AS (T=0.6, L28-31) | 50 | 0.5217 | 0.5690 | **1.091** | `ao_as`/`dc_as`/`im_as` full |
| **Llama-3.1-8B-Instruct** | AS (T=0.5, L28-31) | 500 | 0.4179 | 0.4034 | **0.965** | `rp_as` 500-run |
| **Llama-3.1-8B-Instruct** | HS (B=0.05, L24-31) | 500 | 0.4179 | 0.3929 | **0.940** | `rp_hs` 500-run |
| **Llama-3.1-8B-Instruct** (3c) | ParaphraseDefense | 500 | 0.2562 | 0.3512 | **1.371** | HPC GCG ablation |
| **Llama-3.1-8B-Instruct** (3c) | ParaphraseDefense | 500 | 0.2947 | 0.2014 | **0.683** | Pipeline heldout500 |

| **Qwen2.5-7B-Instruct** | HS only (no undef baseline) | ~50 | — | 0.6055 ⚠️ | — | Qwen Injection JSONL; 43/50 parseable |
| **Qwen3-4B-Instruct-2507** | HS only (no undef baseline) | ~78 | — | −0.0962 ❌ | — | Qwen Injection JSONL; only 19/78 parseable |
| **Qwen3.5-4B** | HS only (no undef baseline) | ~68 | — | 0.0000 ❌ | — | Qwen Injection JSONL; both-label ambiguity |

| **Mistral-7B-Instruct-v0.3** | Baseline only (no defense) | 540 | **0.3264** | — | — | `baseline_test/qw2.5-3b+m7b.ipynb` test_ua split |

---

## Part B: Attacked QWK — Undefended vs Defended

| Model | Defense | Attack | N | Undef Attack QWK | Def Attack QWK | Source |
|---|---|---|---|---|---|---|
| **Llama-3.1-8B-Instruct** | HS (B=0.05, L24-31) | GCG suffix bank | 100 | −0.3172 | 0.0000 | `gcg_hs` full |
| **Llama-3.1-8B-Instruct** | AS (T=0.5, all) | GCG suffix bank | 100 | −0.3172 | 0.0000 | `gcg_as` full |
| **Llama-3.1-8B-Instruct** | HS (B=0.05, L24-31) | AO injection | 50 | −0.3380 | 0.0000 | `ao_hs` full |
| **Llama-3.1-8B-Instruct** | AS (T=0.6, L28-31) | AO injection | 50 | −0.3380 | 0.0000 | `ao_as` full |
| **Llama-3.1-8B-Instruct** | HS (B=0.05, L24-31) | DC injection | 50 | −0.1481 | 0.0000 | `dc_hs` full |
| **Llama-3.1-8B-Instruct** | AS (T=0.6, L28-31) | DC injection | 50 | −0.1481 | 0.0000 | `dc_as` full |
| **Llama-3.1-8B-Instruct** | HS (B=0.05, L24-31) | IM injection | 50 | −0.3380 | 0.0000 | `im_hs` full |
| **Llama-3.1-8B-Instruct** | AS (T=0.6, L28-31) | IM injection | 50 | −0.3380 | 0.0000 | `im_as` full |
| **Llama-3.1-8B-Instruct** (3c) | ParaphraseDefense | GCG suffix bank | 500 | −0.4878 | 0.0000 | Pipeline paraphrase |

| **Qwen2.5-7B-Instruct** | HS | RolePlay | 50 | 0.4493 | 0.0000 | Qwen JSONL |
| **Qwen3-4B-Instruct-2507** | HS | RolePlay | 78 | 0.0000 | 0.0000 | Qwen JSONL |
| **Qwen3.5-4B** | HS | RolePlay | 68 | 0.0693 | 0.0000 | Qwen JSONL |

---

## Key Observations

1. **AS improves clean QWK on small samples (n=50) but not large ones.** At T=0.6, L28-31, AS boosts QWK from 0.5217→0.5690 (+9.1%) on n=50 injection runs. But at n=500 (RolePlay AS), it slightly degrades from 0.4179→0.4034 (−3.5%). This suggests AS overfits to small sample characteristics.

2. **HS consistently degrades clean QWK** across all sample sizes: retention 0.924–0.984. The degradation is modest (−2% to −8%) but consistent.

3. **ParaphraseDefense has wildly inconsistent QWK impact.** On the HPC GCG ablation run, it improves clean QWK by 37% (0.2562→0.3512). On the pipeline run, it degrades by 32% (0.2947→0.2014). This suggests sensitivity to implementation details (paraphrase model, prompt template, etc.).

4. **All three defenses collapse attacked QWK to 0.0** for injection attacks. This is NOT because the defense is working — it's because the attack flips all eligible samples to "correct" (label 0), making QWK uncomputable against a constant prediction. The pre-attack QWK values are heavily negative (−0.15 to −0.34), confirming that the attack has completely destroyed grading agreement.

5. **Qwen and Mistral QWK data is incomplete.** Qwen models only have defended QWK (no undefended baseline for comparison). The Qwen3-4B clean QWK (−0.0962) is unreliable due to unparseable responses. Qwen3.5-4B clean QWK (0.0) reflects both-label ambiguity. Mistral-7B only has a clean baseline QWK of 0.3264 — no defense or attack data exists.

6. **Qwen2.5-7B shows the highest clean QWK (0.6055 with HS)** — notably better than any Llama configuration tested (best Llama clean: 0.5690 with AS, n=50). However, this is WITH the HS defense active, not a true no-defense baseline.
