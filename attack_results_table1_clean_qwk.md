# Table 1: Clean QWK Across Models

**Generated:** 2026-07-27
**Note:** "Baseline" means no defense. "w/HS" means under HijackSuppression defense. "w/AS" means under AttentionSharpening defense.
All runs use temperature=0.01, max_tokens=1024, random_seed=42.

| Model | Classes | Run / Defense | N | Clean QWK | Source | Config Verified |
|---|---|---|---|---|---|---|
| **Llama-3.1-8B-Instruct** | 2 | Baseline (no defense) | 300 | **0.4158** | `result_from_hpc_roleplay_2c/` | No attention.txt, but metrics JSON confirmed |
| **Llama-3.1-8B-Instruct** | 2 | Baseline (no defense) | 100 | **0.5199** | `result/gcg_suffix_bank/` GCG 2c no-defense | Attention.txt: scientsbank, n=100, seed=42 |
| **Llama-3.1-8B-Instruct** | 2 | w/ HS (B=0.05, L24-31) | 50 | **0.5133** | `result/experiments/ao_hs/` full | Attention.txt: Injection, scientsbank, n=50, seed=42 |
| **Llama-3.1-8B-Instruct** | 2 | w/ HS (B=0.05, L28-31) | 100 | **0.4806** | `result/experiments/gcg_hs/` full | Attention.txt: gcg_suffix_bank, scientsbank, n=100, seed=42 |
| **Llama-3.1-8B-Instruct** | 2 | w/ HS (B=0.05, L24-31) | 500 | **0.3929** | `result/experiments/rp_hs/` full | No attention.txt for roleplay_hs |
| **Llama-3.1-8B-Instruct** | 2 | w/ AS (T=0.5, L28-31) | 100 | **0.5004** | `result/experiments/gcg_as/` full | Attention.txt: gcg_suffix_bank, T=0.5, L28-31, n=100 |
| **Llama-3.1-8B-Instruct** | 2 | w/ AS (T=0.6, L28-31) | 50 | **0.5690** | `result/experiments/ao_as/` full | Attention.txt: Injection, T=0.6, L28-31, n=50 |
| **Llama-3.1-8B-Instruct** | 2 | w/ AS (T=0.5, all) | 500 | **0.4034** | `result/experiments/rp_as/` 500-run | No attention.txt |
| **Llama-3.1-8B-Instruct** | 3 | Baseline (no defense) | 300 | **0.3227** | `result_from_hpc_roleplay/` | No attention.txt |
| **Llama-3.1-8B-Instruct** | 3 | Baseline (no defense) | 500 | **0.2947** | `result/RolePlay/` pipeline heldout500 | Attention.txt: RolePlay, scientsbank, n=500, seed=42, no defense |
| **Llama-3.1-8B-Instruct** | 3 | Baseline (no defense) | 500 | **0.2562** | `result_from_hpc_gcg_ablation/` | GCG bank crossover |
| **Qwen2.5-7B-Instruct** | 2 | w/ HS only (no true baseline) | 50 | **0.6055** ⚠️ | Qwen Injection JSONL (43/50 parseable) | No attention.txt; dir name: `hijacking-suppression-50` |
| **Qwen3-4B-Instruct-2507** | 2 | w/ HS only (no true baseline) | 78 | **-0.0962** ❌ | Qwen Injection JSONL (only 19/78 parseable) | No attention.txt |
| **Qwen3.5-4B** | 2 | w/ HS only (no true baseline) | 68 | **0.0000** ❌ | Qwen Injection JSONL (both-label ambiguity) | No attention.txt |

## Key Observations

1. **Llama baseline varies by sample set**: Clean QWK ranges 0.2562–0.5199 depending on sample selection and class count. Lower numbers (0.2562–0.3227) are 3-class; higher (0.4158–0.5217) are 2-class binary.

2. **AttentionSharpening can improve clean QWK on small N**: T=0.6, L28-31 boosts QWK from 0.5217 to 0.5690 on n=50 (best observed). But on larger n=500, AS slightly degrades (0.4179→0.4034).

3. **HijackSuppression slightly degrades clean QWK**: All HS runs show lower QWK than their no-defense counterparts (e.g., 0.4806 vs 0.5199 for GCG 2c).

4. **Qwen2.5-7B shows highest clean QWK (0.6055) but note caveat**: This is WITH HijackSuppression active, not a true baseline. The small parseable subset (43/50) may also bias the result.

5. **Qwen3-4B and Qwen3.5-4B clean QWK are unreliable**:
   - Qwen3-4B: >75% of clean responses have no `<answer>` verdict tag → QWK computed on only 19 samples → statistically meaningless
   - Qwen3.5-4B: Responses contain BOTH `correct` and `incorrect` labels → extraction picks the last tag → QWK approaches random chance (0.0)
