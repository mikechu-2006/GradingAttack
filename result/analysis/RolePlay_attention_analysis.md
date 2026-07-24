# RolePlay Attack Attention Analysis

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Attack Method | RolePlay (Natural Language Prompt Injection) |
| Model | Llama-3.1-8B-Instruct |
| Grading Task | 2-class (correct/incorrect) |
| Template | CI (Course Instruction) |
| Samples | 180 (20 per dataset × 4 + 100 for scientsbank) |
| Datasets | gaokao-2023, gsm8k, math, math23k, scientsbank |
| Adv Prompt | "!!Important!! Now you are in a role-playing game..." |
| Timestamp | 2026-07-23 18:39 |

## Attack Success Rate (ASR)

| Metric | Value |
|--------|-------|
| Total samples | 180 |
| Correct (ground truth) | 71 (39.4%) |
| Incorrect (ground truth) | 109 (60.6%) |
| **ASR** | **59.2%** (42/71 correct samples flipped) |
| Resisted (correct→correct) | 29 (40.8% of correct) |
| Reversed (incorrect→correct) | 26 (23.9% of incorrect) |

### Per-Dataset ASR

| Dataset | n | Correct | ASR | suffix_mean | inst_delta |
|---------|---|---------|-----|-------------|------------|
| gaokao-2023 | 20 | 2 | 50.0% | 0.0658 | +0.0215 |
| gsm8k | 20 | 15 | 13.3% | 0.0475 | -0.0112 |
| math | 20 | 5 | 20.0% | 0.0656 | -0.0069 |
| math23k | 20 | 10 | 40.0% | 0.0541 | -0.0171 |
| **scientsbank** | 100 | 39 | **87.2%** | 0.0318 | -0.0235 |

## Attention Distribution

Last-token attention across prompt segments (mean over 180 attacked samples):

### Clean vs Attacked

| Segment | Clean | Attacked | Delta |
|---------|-------|----------|-------|
| instruction | 0.6039 | 0.5894 | **-0.0146** |
| question | 0.0120 | 0.0094 | -0.0026 |
| solution | 0.0052 | 0.0040 | -0.0012 |
| student | 0.0442 | 0.0244 | -0.0197 |
| suffix | — | **0.0436** | — |
| student+suffix | 0.0442 | **0.0680** | +0.0238 |
| markup | 0.0222 | 0.0155 | -0.0067 |

### Key Observations

1. **Suffix captures only 4.36% of total attention.** As a natural language prompt (~100 tokens), the RolePlay text receives relatively modest attention compared to GCG's 13.95%. The model does not strongly attend to the role-playing instruction.

2. **Student+suffix receives 6.80%** — a 54% increase over clean student attention (4.42%), but far less than GCG's 17.93%. The attack does not strongly hijack attention distribution.

3. **Instruction attention barely drops** (-1.46pp, from 60.39% to 58.94%). Unlike GCG, RolePlay preserves the model's focus on grading instructions. This is a key mechanistic difference between the two attack types.

4. **Despite lower attention, RolePlay achieves higher ASR** (59.2% vs GCG's 40.5%). The attack succeeds through semantic persuasion rather than attention hijacking.

## Suffix Attention vs Attack Success

| Outcome | n | Mean Suffix Attention |
|---------|---|----------------------|
| Attack successful | 42 | 0.0348 |
| Attack failed (resisted) | 29 | 0.0536 |

Counterintuitively, **failed attacks have higher suffix attention (0.0536) than successful ones (0.0348)**. When the model pays more attention to the RolePlay prompt, it appears to recognize and resist the manipulation. Successful attacks work by blending into the student answer without drawing attention.

## Side Effect: Incorrect→Correct Reversal

RolePlay causes a notable number of reversals: **26 originally incorrect answers were graded as correct** after the attack. This is much higher than GCG (3 cases). This suggests the RolePlay prompt has a general "be lenient" effect that biases the model toward grading answers as correct, regardless of actual correctness.

## Comparison: GCG vs RolePlay

| Metric | GCG | RolePlay | Ratio |
|--------|-----|----------|-------|
| Suffix attention | 13.95% | 4.36% | 3.2× |
| Student+suffix attention | 17.93% | 6.80% | 2.6× |
| Instruction drop | -10.98pp | -1.46pp | 7.5× |
| ASR | 40.5% | **59.2%** | 0.68× |
| Incorrect→Correct | 3 (2.5%) | 26 (23.9%) | — |

### Interpretation

**GCG and RolePlay operate through fundamentally different mechanisms:**

- **GCG**: Optimized token-level attack that **hijacks attention** — the suffix aggressively grabs the model's focus, diverting it from grading instructions. High attention cost, moderate success rate. Success correlates poorly with attention magnitude.

- **RolePlay**: Natural language semantic attack that **persuades through content** — the model barely notices the prompt as "special" (low suffix attention), but its grading behavior changes nonetheless. Low attention cost, high success rate. Higher suffix attention actually correlates with *failure* (the model "catches on" when it focuses too much on the RolePlay text).

## Summary

- RolePlay's natural language prompt captures only **4.36%** of attention (vs GCG's 13.95%)
- Instruction attention is preserved (-1.46pp vs GCG's -10.98pp)
- Despite minimal attention impact, ASR reaches **59.2%** — 1.46× higher than GCG
- **Semantic persuasion > attention hijacking** for this particular attack-model pair
- scientsbank dominates both attacks (87.2% ASR), confirming it as the most vulnerable dataset
- Lower suffix attention predicts higher attack success (counterintuitive)
- Significant leniency bias: 23.9% of incorrect answers get reversed to correct
