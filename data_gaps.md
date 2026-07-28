# Data Gaps — What Remains to Be Filled

**Generated:** 2026-07-27 | **Class count:** 2c (default), noted where 3c only

---

## ✅ Filled

| Gap | Value | Source |
|---|---|---|
| Llama-3.1-8B undefended clean QWK | 0.5199 | Pipeline metrics |
| Qwen2.5-7B undefended clean QWK | **0.5529** | User-provided (notebook) |
| Qwen3-4B-2507 undefended clean QWK | **0.6741** | User-provided (notebook) |
| Qwen3.5-4B undefended clean QWK | ❌ still missing | — |
| Mistral-7B undefended clean QWK | 0.3264 | User-provided (notebook) |

---

## Table 1: Clean QWK (by model)

| Model | Undefended | HS | AS | Paraphrase |
|---|---|---|---|---|
| **Llama-3.1-8B** | ✅ 0.5199 | ✅ 0.4806 | ✅ 0.5004 | ✅ (3c only: 0.3512) |
| **Qwen2.5-7B** | ✅ 0.5529 | ⚠️ 0.6055¹ | ❌ | ❌ |
| **Qwen3-4B-2507** | ✅ 0.6741 | ❌ unparseable² | ❌ | ❌ |
| **Qwen3.5-4B** | ❌ | ❌ ambiguous³ | ❌ | ❌ |
| **Mistral-7B** | ✅ 0.3264 | ❌ | ❌ | ❌ |

¹ Qwen2.5-7B with HS gave 0.6055 but clean baseline 0.5529 → retention 1.095 (improved!).
² Qwen3-4B clean responses lack `<answer>` tags → defended QWK meaningless.
³ Qwen3.5-4B responses have both "correct" and "incorrect" → defended QWK ~0.0.

---

## Table 2: Injection ASR (by model × injection prompt)

| Model | RolePlay | AO | DC | IM |
|---|---|---|---|---|
| **Llama-3.1-8B** | ✅ 0.0058 | ✅ 1.0000 | ✅ 0.7407 | ✅ 1.0000 |
| **Qwen2.5-7B** | ❌¹ | ❌ | ❌ | ❌ |
| **Qwen3-4B-2507** | ❌¹ | ❌ | ❌ | ❌ |
| **Qwen3.5-4B** | ❌¹ | ❌ | ❌ | ❌ |
| **Mistral-7B** | ❌ | ❌ | ❌ | ❌ |

¹ RolePlay was run but with HS defense active → no undefended ASR available.

---

## Table 3: GCG vs GCG Suffix Bank ASR (attack only)

| Model | GCG Per-Sample | GCG Suffix Bank |
|---|---|---|
| **Llama-3.1-8B** | ✅ 0.3443 (2c, n=100) | ✅ 1.0000 (2c, n=100) / 0.9963 (3c, n=500) |
| **Qwen2.5-7B** | ❌ | ❌ |
| **Qwen3-4B-2507** | ❌ | ❌ |
| **Qwen3.5-4B** | ❌ | ❌ |
| **Mistral-7B** | ❌ | ❌ (only needed in Table 4) |
| Bank size sweep | — | ❌ (only 1 and 5 tested) |

---

## Table 4: ASR Reduction by Defense (only HS, AS, Paraphrase)

### Against GCG Suffix Bank

| Model | HS Reduction | AS Reduction | Paraphrase Reduction |
|---|---|---|---|
| **Llama-3.1-8B** (2c) | ✅ 24.6% | ✅ 1.9% | ❌ (only 3c) |
| **Llama-3.1-8B** (3c) | ❌ | ❌ | ✅ 63–96% |
| **Qwen2.5-7B** | ❌ | ❌ | ❌ |
| **Qwen3-4B-2507** | ❌ | ❌ | ❌ |
| **Qwen3.5-4B** | ❌ | ❌ | ❌ |
| **Mistral-7B** | ❌ | ❌ | ❌ |

### Against Injection (avg of AO, DC, IM)

| Model | HS Reduction | AS Reduction | Paraphrase Reduction |
|---|---|---|---|
| **Llama-3.1-8B** (2c) | ✅ 7.5% | ✅ −1.3% (worse) | ❌ |
| **Qwen2.5-7B** | ❌ | ❌ | ❌ |
| **Qwen3-4B-2507** | ❌ | ❌ | ❌ |
| **Qwen3.5-4B** | ❌ | ❌ | ❌ |
| **Mistral-7B** | ❌ | ❌ | ❌ |

---

## Table 5: QWK Retention by Defense

| Model | HS Retention | AS Retention | Paraphrase Retention |
|---|---|---|---|
| **Llama-3.1-8B** | ✅ 0.924 (n=100) / 0.940 (n=500) | ✅ 0.963 (n=100) / 1.091 (n=50) | ✅ 0.683–1.371 (3c only) |
| **Qwen2.5-7B** | ⚠️ 1.095¹ | ❌ | ❌ |
| **Qwen3-4B-2507** | ❌ unparseable | ❌ | ❌ |
| **Qwen3.5-4B** | ❌ ambiguous | ❌ | ❌ |
| **Mistral-7B** | ❌ | ❌ | ❌ |

¹ Qwen2.5-7B HS retention: 0.6055/0.5529 = 1.095 — HS actually improved grading. Only 43/50 samples parseable though.

---

## Cross-Cutting: Defenses Never Tested

| Defense | Tested Against | Never Tested Against |
|---|---|---|
| **PerplexityFilter** | Nothing | All attacks |
| **SystemPromptChange** | Nothing | All attacks |
| **SmoothLLM** | RolePlay (n=50 2c) | GCG bank, injection |
| **SelfReminder** | RolePlay (2c/3c) | GCG bank, injection |
| **Combined defenses** | Nothing | All |

---

## What's Still Needed — Actionable List

### 🔴 Critical (needed for Tables 2–5)

| # | What | Why | How |
|---|---|---|---|
| 1 | **Qwen2.5-7B undefended RolePlay ASR** | Table 2 missing; also blocks Table 4/5 HC | Re-run pipeline: Qwen2.5-7B + RolePlay, no defense |
| 2 | **Qwen3.5-4B undefended baseline** | Table 1 missing | Get clean QWK from notebook or run pipeline |
| 3 | **Qwen3-4B clean verdict parsing** | All defended metrics uncomputable | Fix prompt template to emit `<answer>` tag |
| 4 | **Mistral × GCG suffix bank** | Table 4: user specifically requested | Build GCG suffix bank on Mistral; evaluate ASR |
| 5 | **Mistral × Paraphrase vs GCG** | Table 4: main defense against GCG | Run Mistral + Paraphrase on GCG bank |

### 🟠 Medium (fills out Tables 2–4)

| # | What | Why | How |
|---|---|---|---|
| 6 | **All Qwen × AO/DC/IM injection** | Table 2/4: only RolePlay tested | Run pipeline: each Qwen + AO/DC/IM |
| 7 | **Llama: Paraphrase vs GCG at 2c** | Table 4: only 3c exists | Run Paraphrase on GCG bank 2c pipeline |
| 8 | **Llama: Paraphrase vs injection** | Table 4: missing entirely | Run Paraphrase + injection pipeline |
| 9 | **Llama: SelfReminder vs GCG bank** | Table 4: best defense vs strongest attack | Add SelfReminder to GCG bank pipeline |
| 10 | **GCG bank size sweep** | Table 3: understand minimum viable bank | Build banks of sizes 2,3,4,10 on Llama |

### 🟢 Low (nice to have)

| # | What | Why |
|---|---|---|
| 11 | PerplexityFilter / SystemPromptChange | Already implemented, zero results |
| 12 | Combined defenses (Paraphrase+SelfReminder) | Could outperform either alone |
| 13 | Injection at n=300+ | Current n=50 may not generalize |
