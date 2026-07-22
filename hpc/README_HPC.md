# HPC run notes for GradingAttack

These notes follow the HKUST-GZ HPC connection and Slurm job-submission docs.

## 1. Log in

Use SSH from PowerShell, replacing `<username>` with your SSO username/email prefix. On this machine, `hpc2login.hpc.hkust-gz.edu.cn` did not resolve, while HPC1 resolved and was reachable, so start with HPC1:

```bash
ssh <username>@hpc1login.hpc.hkust-gz.edu.cn
```

If HPC2 is available for your account/network, you can also try:

```bash
ssh <username>@hpc2login.hpc.hkust-gz.edu.cn
```

Because Windows OpenSSH is not installed on this machine, use Git's SSH client locally:

```powershell
& "D:\Git\usr\bin\ssh.exe" <username>@hpc1login.hpc.hkust-gz.edu.cn
```

The web route is also available from the HPC portal: open the portal, go to the app repository, then open Terminal.

## 2. Put the project on HPC

From your local machine, sync this project to your HPC home directory:

```bash
scp -r "E:/Documents/SummerSchool/AI Security/Project/GradingAttack" <username>@hpc1login.hpc.hkust-gz.edu.cn:~/GradingAttack
```

If `scp` has trouble with the Windows path spaces, run the command from the parent directory and use the folder name. Since this machine uses Git's SSH client, the matching SCP client is usually:

```powershell
& "D:\Git\usr\bin\scp.exe" -r "E:\Documents\SummerSchool\AI Security\Project\GradingAttack" <username>@hpc1login.hpc.hkust-gz.edu.cn:~/GradingAttack
```

## 3. Prepare the environment on HPC

On HPC:

```bash
cd ~/GradingAttack
conda create -n gradingattack python=3.11 -y
conda activate gradingattack
pip install -r requirements.txt
```

For clean baseline only, you need at least `torch`, `transformers`, `scikit-learn`, `pyyaml`, and optionally `modelscope` if you want ModelScope downloads.

## 4. Check queues and edit partition

Inspect available partitions:

```bash
sinfo
```

Then edit `hpc/run_clean_baseline_hpc.sh` and set the `#SBATCH -p ...` line to a partition you can use. The current value follows this repo's existing cluster script: `emergency_gpua40`.

## 5. Submit clean baseline

```bash
cd ~/GradingAttack
sbatch hpc/run_clean_baseline_hpc.sh
```

## 6. Watch the job

```bash
squeue -u $USER
scontrol show job <jobid>
tail -f hpc/logs/clean_baseline_<jobid>.out
tail -f hpc/logs/clean_baseline_<jobid>.err
```

After it finishes:

```bash
sacct -u $USER
ls -lh result_baseline
```

## 7. Cancel if needed

```bash
scancel <jobid>
```

## 8. Copy results back

From local PowerShell:

```powershell
& "D:\Git\usr\bin\scp.exe" -r <username>@hpc1login.hpc.hkust-gz.edu.cn:~/GradingAttack/result_baseline "E:\Documents\SummerSchool\AI Security\Project\GradingAttack\"
```