# GCG Suffix-Bank Attack Guide

This document explains the GCG attack workflow used in this project, including
the threat model, code structure, HPC commands, and how to interpret the
metrics.

## What GCG Does

GCG is a token-level adversarial suffix attack. Instead of using a fixed natural
language prompt such as RolePlay, it optimizes a suffix that is appended or
inserted into the student answer field so that the grader is more likely to
output:

```json
{"verdict": "correct"}
```

In this project, GCG is used in a **blind attacker** setting:

- The attacker optimizes the suffix without seeing the reference solution.
- The suffix optimization prompt uses `solution=""`.
- During evaluation, the normal grader still receives the reference solution,
  because ASAG grading requires it.

This means the GCG attacker only controls the injected suffix, not the grading
rubric or the reference answer.

## Main GCG Workflow

The recommended workflow is **suffix-bank transfer**:

1. Build a small bank of successful GCG suffixes.
2. Evaluate whether those suffixes transfer to heldout samples.
3. Evaluate defenses against the same suffix bank.

This is more useful than running per-sample GCG from scratch on every test item,
because per-sample GCG is slower and less stable.

## Important Files

| File | Purpose |
|---|---|
| `scripts/build_gcg_suffix_bank.py` | Builds a suffix bank by running GCG on eligible samples. |
| `scripts/eval_gcg_suffix_bank.py` | Tests an existing suffix bank on heldout samples. |
| `scripts/eval_gcg_suffix_bank_defense.py` | Tests defenses against an existing suffix bank. |
| `hpc/build_gcg_suffix_bank.sh` | HPC job script for building a bank and optionally running transfer. |
| `hpc/run_gcg_suffix_bank_transfer.sh` | HPC job script for transfer-only evaluation. |
| `hpc/run_gcg_suffix_bank_defense.sh` | HPC job script for defense evaluation. |
| `baselines/gcg/gcg.py` | Original per-sample GCG implementation using `nanogcg`. |

## Metrics

### Project ASR

The main attack metric is:

```text
GT != correct AND original_pred != correct -> attacked_pred == correct
```

In words: among non-correct student answers that the clean grader did not
already mark as correct, how many become `correct` after attack?

### Promotion ASR

Promotion ASR counts any grading relaxation:

```text
incorrect -> contradictory
incorrect -> correct
contradictory -> correct
```

This is useful in the three-class setting because attacks may inflate grades
without always reaching `correct`.

### Transfer Rate

`transfer_rate_any_original` counts all ground-truth non-correct samples that
can be made `correct` by any suffix in the bank, including samples that the
clean grader had already marked correct.

### Defense Metrics

Defense evaluation should not use ASR alone. A defense can lower ASR by damaging
normal grading behavior. We therefore report:

```text
ASR reduction = 1 - defended_ASR / attack_ASR
Clean QWK retention = QWK_defense_clean / QWK_clean
```

Clean QWK retention penalizes defenses that misgrade normal, unattacked student
answers, including correct answers being changed to incorrect.

## Build a GCG Suffix Bank

On HPC2:

```bash
cd ~/GradingAttack
chmod +x hpc/build_gcg_suffix_bank.sh

NCLASS=3 TEMPLATE_PATH=configs/grading_template_ci_3c.txt SUCCESS_MODE=promotion \
MAX_CANDIDATES=50 TARGET_BANK_SIZE=5 NUM_STEPS=100 TRANSFER_MAX_SAMPLES=500 \
OUTPUT_BANK=result/GCG/gcg_suffix_bank_3c_promotion.jsonl \
TRANSFER_OUTPUT=result/GCG/gcg_suffix_bank_transfer_3c_promotion_heldout500.jsonl \
sbatch hpc/build_gcg_suffix_bank.sh
```

Useful environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `NCLASS` | `2` or `3` class grading. | `2` in transfer script, `3` in defense script |
| `TEMPLATE_PATH` | Prompt template path. | `configs/grading_template_ci_2c.txt` or `3c` |
| `SUCCESS_MODE` | `correct` or `promotion` while building bank. | `correct` |
| `MAX_CANDIDATES` | Number of non-correct candidates to try. | `50` |
| `TARGET_BANK_SIZE` | Stop after this many successful suffixes. | `10` |
| `NUM_STEPS` | GCG optimization steps per candidate. | `100` |
| `OUTPUT_BANK` | JSONL suffix bank output. | `gcg_suffix_bank.jsonl` |

Check status:

```bash
squeue -u $USER
sacct -j JOBID --format=JobID,JobName,Partition,State,ExitCode,Elapsed,NodeList%20
tail -100 hpc/logs/build_gcg_bank_JOBID.out
tail -100 hpc/logs/build_gcg_bank_JOBID.err
```

## Run Transfer with an Existing Bank

If the suffix bank already exists, run transfer directly:

```bash
cd ~/GradingAttack
chmod +x hpc/run_gcg_suffix_bank_transfer.sh

NCLASS=3 TEMPLATE_PATH=configs/grading_template_ci_3c.txt \
BANK_PATH=result/GCG/gcg_suffix_bank_3c_promotion.jsonl \
MAX_SAMPLES=500 \
OUTPUT_PATH=result/GCG/gcg_suffix_bank_transfer_3c_promotion_heldout500.jsonl \
sbatch hpc/run_gcg_suffix_bank_transfer.sh
```

Read the metrics:

```bash
cat result/GCG/gcg_suffix_bank_transfer_3c_promotion_heldout500_metrics.json
```

The current heldout500 result is:

```text
bank_size = 5
eligible_non_correct = 269
project_asr = 268 / 269 = 99.63%
promotion_asr = 269 / 269 = 100.00%
transfer_rate_any_original = 291 / 292 = 99.66%
```

The transfer script excludes suffix-bank source samples by default when the bank
contains `source_index`.

## Run Defense Evaluation

Run SelfReminder against the GCG suffix bank:

```bash
cd ~/GradingAttack
chmod +x hpc/run_gcg_suffix_bank_defense.sh

DEFENSE=self_reminder \
BANK_PATH=result/GCG/gcg_suffix_bank_3c_promotion.jsonl \
MAX_SAMPLES=500 \
OUTPUT_PATH=result/GCG/gcg_suffix_bank_defense_self_reminder_heldout500.jsonl \
sbatch hpc/run_gcg_suffix_bank_defense.sh
```

Other defenses:

```bash
DEFENSE=paraphrase \
BANK_PATH=result/GCG/gcg_suffix_bank_3c_promotion.jsonl \
MAX_SAMPLES=500 \
OUTPUT_PATH=result/GCG/gcg_suffix_bank_defense_paraphrase_heldout500.jsonl \
sbatch hpc/run_gcg_suffix_bank_defense.sh
```

```bash
DEFENSE=hijacking \
BANK_PATH=result/GCG/gcg_suffix_bank_3c_promotion.jsonl \
MAX_SAMPLES=500 \
OUTPUT_PATH=result/GCG/gcg_suffix_bank_defense_hijacking_heldout500.jsonl \
sbatch hpc/run_gcg_suffix_bank_defense.sh
```

Read metrics:

```bash
cat result/GCG/gcg_suffix_bank_defense_self_reminder_heldout500_metrics.json
```

Current SelfReminder result:

```text
project_asr = 99.63%
defended_project_asr = 95.49%
asr_reduction = 4.15%
clean_qwk_retention = 102.10%
```

Interpretation: SelfReminder preserves clean grading quality, but it barely
reduces the optimized GCG suffix-bank attack.

## Local Result Files

Pulled HPC results are stored in:

```text
result_from_hpc_gcg_suffix_bank/
```

Key files:

```text
gcg_suffix_bank_3c_promotion.jsonl
gcg_suffix_bank_transfer_3c_promotion_heldout500.jsonl
gcg_suffix_bank_transfer_3c_promotion_heldout500_metrics.json
```

The high-level experiment summary is in:

```text
RESULTS.md
```

## Short Explanation for Reports

Use this wording:

> We first build a small bank of successful GCG suffixes in a blind setting
> without access to the reference solution. We then evaluate transferability on
> heldout SciEntsBank samples, excluding the bank source samples. A bank of 5
> suffixes reaches 99.63% project ASR on 500 heldout samples, showing that
> optimized suffixes transfer much more strongly than a fixed RolePlay prompt.
> Defense evaluation should report both ASR reduction and clean QWK retention,
> because a defense that damages normal grading is not useful.
