# GradingAttack: A Unified Adversarial Evaluation Framework for LLM-based Automatic Short Answer Grading

## Abstract

Automatic short answer grading (ASAG) aims to automatically evaluate the correctness of students' short written answers to objective questions without requiring manual work from teachers. Large language models (LLMs) have demonstrated remarkable potential for ASAG, significantly boosting student assessment efficiency and the scalability of Web-deployed educational applications. However, their vulnerability to adversarial manipulation raises critical concerns about grading fairness and reliability. In this paper, we introduce a unified adversarial evaluation framework that systematically evaluates the vulnerability of LLM-based ASAG models. Specifically, we align general-purpose adversarial attack methods with the specific objectives of ASAG to achieve effective evaluation in the ASAG task. Experiments on multiple datasets demonstrate that both attack methods effectively mislead LLM-based grading models, with token-level attack methods providing stronger camouflage, whereas prompt-level attack methods achieve higher success rates. Our findings highlight critical robustness issues in LLM-based ASAG models and underscore the need for stronger defenses to ensure fair, transparent and trustworthy AI-driven assessment.

## Installation

```bash
git clone path/to/repo
cd GradingAttack
pip install -r requirements.txt
```

## Quick Start

1. Create a config file (see example in `configs/GCG-Llama3.1-8B-Instruct.yaml`)

2. Run attack:
    ```bash
    python main.py configs/GCG-Llama3.1-8B-Instruct.yaml
    ```

## License

MIT
