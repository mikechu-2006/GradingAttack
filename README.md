# GradingAttack

LLM-based ASAG (Automatic Short Answer Grading) adversarial attack and defense evaluation framework.

This project studies whether prompt-level attacks can manipulate LLM grading behavior, and whether lightweight defenses can reduce that effect while preserving normal grading quality.

## Current Focus

The current experiments focus on:

- **Task:** Automatic Short Answer Grading
- **Dataset:** SciEntsBank
- **Model:** Llama-3.1-8B-Instruct
- **Main attack:** RolePlay prompt injection
- **Main defenses:** SelfReminder, ParaphraseDefense, SmoothLLM, AttentionSharpening, HijackingSuppression

See [RESULTS.md](RESULTS.md) for the current experiment summary.

## Project Structure

```text
GradingAttack/
|-- main.py                 # CLI entrypoint
|-- grading_attack.py       # Attack/defense dispatcher
|-- pipeline.py             # Attack -> defense -> evaluation pipeline
|-- baseline_eval.py        # Clean baseline evaluation
|-- baselines/
|   |-- gcg/                # Token-level GCG attack
|   |-- roleplay/           # Prompt-level RolePlay attack
|   `-- defenses/           # Defense implementations
|-- configs/                # YAML experiment configs
|-- dataset/                # JSONL datasets
|-- eval/metrics.py         # QWK / ASR / CAS metrics
|-- hpc/                    # HKUST-GZ HPC2 helper scripts
|-- scripts/                # Grid search, plotting, and helper scripts
|-- result_from_hpc_roleplay/
|-- result_from_hpc_self_reminder/
`-- RESULTS.md              # Current experiment conclusions
```

## Attacks

| Attack | Type | Purpose |
|---|---|---|
| RolePlay | Prompt-level | Injects a role-playing instruction into the grading prompt, asking the model to pretend the student answer is correct. |
| GCG | Token-level | Uses adversarial suffix optimization through `nanogcg`. |

## Defenses

| Defense | Stage | Idea |
|---|---|---|
| SelfReminder | Prompt pre-processing | Adds explicit grading and safety instructions before the task prompt. |
| ParaphraseDefense | Prompt pre-processing | Rewrites the request before grading to remove suspicious or irrelevant injected content. |
| SmoothLLM | Inference-time ensemble | Applies random perturbations and majority voting. |
| PerplexityFilter | Pre-processing | Filters suspicious high-perplexity input. |
| AttentionSharpening | Model hook | Sharpens attention distributions during inference. |
| HijackingSuppression | Model hook | Down-weights high-dominance suffix attention during inference. |

## Key Results

Main 300-sample RolePlay results:

| Defense | Clean QWK | ASR | Defended Clean QWK | Defended ASR | Takeaway |
|---|---:|---:|---:|---:|---|
| None | 0.3227 | 0.1657 | 0.3227 | 0.1657 | Establishes baseline vulnerability. |
| SelfReminder | 0.3227 | 0.1657 | 0.3319 | 0.0497 | Best robustness-utility tradeoff so far. |
| ParaphraseDefense | 0.3227 | 0.1657 | 0.1837 | 0.0221 | Strongest ASR reduction, but large clean-QWK drop. |

Current conclusion:

> RolePlay prompt injection can meaningfully change LLM grading behavior. SelfReminder provides the best tradeoff among the tested defenses, while ParaphraseDefense is stronger at suppressing attacks but harms normal grading quality.

## Installation

```bash
git clone https://github.com/mikechu-2006/GradingAttack.git
cd GradingAttack
pip install -r requirements.txt
```

For HPC2, use the setup notes in [hpc/HPC2_QUICKSTART.md](hpc/HPC2_QUICKSTART.md).

## Running Experiments

Run a configured attack-defense pipeline:

```bash
python main.py --pipeline configs/RolePlay-Llama-3.1-8B-Instruct-cluster.yaml
```

Run a clean baseline:

```bash
python baseline_eval.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data ./dataset/scientsbank.jsonl \
  --device cuda \
  --max_samples 200 \
  --output ./result/scientsbank_baseline.jsonl
```

On HPC2, a typical workflow is:

```bash
cd ~/GradingAttack
module load anaconda3
module load cuda/12.4
eval "$(conda shell.bash hook)"
conda activate gradingattack
sbatch hpc/overnight_self_reminder.sh
```

Check job status and logs:

```bash
squeue -u $USER
sacct -j JOBID --format=JobID,JobName,Partition,State,ExitCode,Elapsed,NodeList%20
tail -100 hpc/logs/<job_log>.out
tail -100 hpc/logs/<job_log>.err
```

## Plotting Results

Generate the current ASR/QWK comparison figure:

```bash
python scripts/plot_results.py
```

The generated files are written to `figures/`:

- `figures/roleplay_defense_summary.csv`
- `figures/roleplay_defense_comparison.svg`

## Notes

- The most complete current result summary is in [RESULTS.md](RESULTS.md).
- Some attention-based defense experiments are exploratory and should not be presented as successful defenses unless their ASR improves on the no-defense baseline.
- The current repository is ready for report writing; additional experiments are only needed if the final report must include a new proposed defense such as HijackingSuppression.

## License

MIT