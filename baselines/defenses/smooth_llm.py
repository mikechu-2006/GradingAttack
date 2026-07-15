import random

from .base import BaseDefense


class SmoothLLM(BaseDefense):
    """通过随机字符扰动生成多个 prompt 副本，取多数投票结果。

    对应论文:
    - Robey et al. (2023) "SmoothLLM: Defending Large Language Models Against
      Jailbreaking Attacks"

    原理: 攻击后缀对字符级扰动高度敏感。将原始 prompt 随机扰动 N 份后推理，
    若 N 份输出的多数结果与原始不同，说明 prompt 不可靠，返回修正结果。
    """

    PERTURB_RATE = 0.1  # 随机修改字符的比例

    def __init__(self, num_copies: int = 5, perturb_rate: float = None):
        super().__init__()
        self.num_copies = num_copies
        if perturb_rate is not None:
            self.PERTURB_RATE = perturb_rate

    def requires_multiple_generations(self) -> bool:
        return True

    def generate_variants(self, prompt: str) -> list[str]:
        """除原始 prompt 外，生成 num_copies 个扰动版本"""
        return [prompt] + [self._perturb(prompt) for _ in range(self.num_copies)]

    def pre_process(self, prompt: str) -> str:
        return prompt  # 扰动在 generate_variants 中完成

    def _perturb(self, text: str) -> str:
        """三种扰动策略随机混合：替换 / 交换 / 插入"""
        chars = list(text)
        n = max(1, int(len(chars) * self.PERTURB_RATE))

        for _ in range(n):
            idx = random.randint(0, len(chars) - 1)
            op = random.choice(["replace", "swap", "insert"])
            if op == "replace" and chars:
                chars[idx] = random.choice("abcdefghijklmnopqrstuvwxyz")
            elif op == "swap" and idx < len(chars) - 1:
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            elif op == "insert":
                chars.insert(idx, random.choice("abcdefghijklmnopqrstuvwxyz"))

        return "".join(chars)
