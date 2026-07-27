# GCG attention runs on five models

This helper runs the GCG suffix-bank pipeline with `log_attention: true`.
The current Llama-3.1-8B-Instruct heldout500 attention run already exists, so
the default submit script runs four additional models:

- Qwen2.5-7B-Instruct
- Qwen3-4B-Instruct-2507
- Qwen3.5-4B
- Qwen2.5-3B-Instruct

Together with the existing Llama run, this gives five models.

## 1. Upload scripts

From local PowerShell:

```powershell
cd "E:\Documents\SummerSchool\AI Security\Project\GradingAttack"
& "D:\Git\usr\bin\scp.exe" .\hpc\download_gcg_attention_models.sh pwang176@hpc2login.hpc.hkust-gz.edu.cn:~/GradingAttack/hpc/
& "D:\Git\usr\bin\scp.exe" .\hpc\submit_gcg_attention_models.sh pwang176@hpc2login.hpc.hkust-gz.edu.cn:~/GradingAttack/hpc/
```

## 2. Download models on the login node

On HPC:

```bash
cd ~/GradingAttack
sed -i 's/\r$//' hpc/download_gcg_attention_models.sh hpc/submit_gcg_attention_models.sh
chmod +x hpc/download_gcg_attention_models.sh hpc/submit_gcg_attention_models.sh
bash hpc/download_gcg_attention_models.sh
```

If a model ID fails, replace that line through `MODEL_SPECS` or use a local
`model_path`.

## 3. Submit GPU jobs

```bash
cd ~/GradingAttack
bash hpc/submit_gcg_attention_models.sh
squeue -u $USER
```

For a quick smoke test first:

```bash
MAX_SAMPLES=20 bash hpc/submit_gcg_attention_models.sh
```

## 4. Check finished jobs

```bash
ls -lt hpc/logs/merged_pipeline_*.out hpc/logs/merged_pipeline_*.err | head -20
sacct -j JOBID1,JOBID2,JOBID3,JOBID4 --format=JobID,JobName,Partition,State,ExitCode,Elapsed,NodeList%20
find result/gcg_suffix_bank -type f -name '*attention.csv' -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort | tail -20
```

## 5. Run the detector

After pulling each `*_attention.csv` back locally:

```bash
python scripts/analyze_attention_detector.py \
  --input path/to/model_attention.csv \
  --output-dir result/analysis/attention_detector/model_name
```
