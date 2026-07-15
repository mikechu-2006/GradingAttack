# GradingAttack

LLM-based ASAG (Automatic Short Answer Grading) 对抗攻击与防御评估框架。

## 项目结构

```
GradingAttack/
├── main.py                 # 入口：argparse → GradingAttack.run()
├── grading_attack.py       # 工厂类：根据 method 分发到 GCG / RolePlay
├── pipeline.py             # 攻防流水线：attack → defense → evaluate
├── baseline_eval.py        # Clean baseline：不攻击不防御，测原始准确率
├── colab_pipeline.ipynb    # Google Colab 一键运行 notebook
├── baselines/
│   ├── gcg/gcg.py          # Token-level GCG 攻击 (nanogcg)
│   ├── roleplay/roleplay.py # Prompt-level RolePlay 攻击 (vLLM)
│   └── defenses/
│       ├── base.py              # BaseDefense 抽象基类
│       ├── perplexity_filter.py # PPL 过滤 (Jain et al. 2023)
│       ├── smooth_llm.py        # 随机扰动 + 多数投票 (Robey et al. 2023)
│       ├── self_reminder.py     # 安全指令前缀 (Wu et al. 2024)
│       └── paraphrase.py        # 改写防御 (Zhang et al. 2024)
├── eval/
│   └── metrics.py          # ASR / CAS / 混淆矩阵
├── utils/
│   ├── config_utils.py     # AttackConfig 等 dataclass + YAML 解析
│   ├── data_utils.py       # StudentQAData, AttackResult + JSONL 读取
│   └── log_utils.py        # 日志与结果写入
├── configs/
│   ├── GCG-Llama-3.1-8B-Instruct.yaml
│   ├── GCG-Llama-3.1-8B-Instruct-defense.yaml
│   ├── GCG-Llama-3.1-8B-Instruct-smoothllm.yaml
│   └── RolePlay-Llama-3.1-8B-Instruct.yaml
└── dataset/
```

## 安装

```bash
git clone https://github.com/mikechu-2006/GradingAttack.git
cd GradingAttack
pip install -r requirements.txt
```

## 快速开始

### 1. Clean Baseline

不攻击不防御，测量 LLM 原始 grading 准确率：

```bash
python baseline_eval.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --data ./dataset/scientsbank.jsonl \
    --device cuda \
    --max_samples 200 \
    --output ./result/scientsbank_baseline.jsonl
```

### 2. 攻击（无防御）

```bash
# GCG token-level 攻击
python main.py configs/GCG-Llama-3.1-8B-Instruct.yaml

# RolePlay prompt-level 攻击
python main.py configs/RolePlay-Llama-3.1-8B-Instruct.yaml
```

### 3. 攻击 + 防御 Pipeline

```bash
# SelfReminder + PerplexityFilter 防御
python main.py --pipeline configs/GCG-Llama-3.1-8B-Instruct-defense.yaml

# SmoothLLM 防御
python main.py --pipeline configs/GCG-Llama-3.1-8B-Instruct-smoothllm.yaml
```

## 攻击方法

| 方法 | 类型 | 说明 |
|------|------|------|
| GCG | Token-level (白盒) | 基于梯度优化对抗 suffix，使用 nanogcg 库 |
| RolePlay | Prompt-level | 在 grading prompt 中注入角色扮演字符串 |

## 防御方法

| 方法 | 阶段 | 说明 |
|------|------|------|
| PerplexityFilter | Pre-processing | 检测高困惑度输入并拒绝 |
| SmoothLLM | Inference | 随机字符扰动 + 多次生成多数投票 |
| SelfReminder | Pre-processing | 在 prompt 前追加安全 grading 指令 |
| ParaphraseDefense | Pre-processing | 要求模型先改写再评判 |

## 数据集

| 数据集 | 说明 |
|--------|------|
| SciEntsBank | 5000 样本，15 个科学领域，SemEval-2013 |
| MATH | 数学题 |
| GSM8K | 小学数学应用题 |
| Math23K | 中文数学题 |
| Gaokao-2023 | 2023 高考题 |

所有数据集为 JSONL 格式，字段：`question_id`, `student_id`, `question`, `question_answer`, `student_answer`, `verification`。

## Colab 运行

直接打开 `colab_pipeline.ipynb` 在 Google Colab 中运行，包含：
1. Drive 挂载
2. 代码克隆
3. 依赖安装
4. 模型下载
5. Baseline 评估
6. GCG 攻击
7. GCG + 防御 Pipeline

## License

MIT
