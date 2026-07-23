# HPC2 quick commands

After uploading and unzipping this project to `~/GradingAttack`, use only these short commands in the HPC2 web Terminal.

```bash
cd ~/GradingAttack
bash hpc/setup_hpc.sh
sbatch hpc/test_gpu.sh
squeue -u $USER
```

After the GPU test finishes, inspect logs:

```bash
ls -lh hpc/logs
cat hpc/logs/test_gpu_<jobid>.out
cat hpc/logs/test_gpu_<jobid>.err
```

Then submit clean baseline:

```bash
sbatch hpc/run_clean_baseline_hpc.sh
squeue -u $USER
```

Inspect clean baseline logs/results:

```bash
ls -lh hpc/logs
cat hpc/logs/clean_baseline_<jobid>.out
cat hpc/logs/clean_baseline_<jobid>.err
ls -lh result_baseline
```

Useful overrides:

```bash
MAX_SAMPLES=20 sbatch hpc/run_clean_baseline_hpc.sh
CONFIG_PATH=configs/GCG-Qwen3-4B-Instruct.yaml MAX_SAMPLES=20 sbatch hpc/run_clean_baseline_hpc.sh
```

Default modules/settings:

- Conda module: `anaconda3`
- Conda env: `gradingattack`
- CUDA module: `cuda/12.4`
- PyTorch wheel index: `https://download.pytorch.org/whl/cu124`
- Test partition: `debug`
- Baseline partition: `emergency_gpua40`
