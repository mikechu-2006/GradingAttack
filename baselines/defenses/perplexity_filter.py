import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseDefense, DefenseRejectException


class PerplexityFilter(BaseDefense):
    """基于 GCG suffix 高困惑度特性进行检测。

    对应论文:
    - Jain et al. (2023) "Baseline Defenses for Adversarial Attacks Against Aligned Language Models"
    - Alon & Kamfonas (2023) "Detecting Language Model Attacks with Perplexity"

    原理: GCG 生成的后缀是梯度优化的无意义 token 序列，
    在受害模型上的条件困惑度远高于正常文本。
    """

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                 threshold: float = 1000.0, device: str = "cuda"):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.device = device

    def pre_process(self, prompt: str) -> str:
        tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(tokens, labels=tokens)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()

        if perplexity > self.threshold:
            raise DefenseRejectException(
                self.name,
                f"Perplexity {perplexity:.1f} > threshold {self.threshold}"
            )
        return prompt
