import functools
from typing import List, Union

import torch.nn.functional as F

from .base import BaseDefense


class AttentionSharpening(BaseDefense):
    """推理期 sharpen attention 分布，缓解 Attention Slipping。

    对应论文:
    - Hu et al. (2025) "Attention Slipping: A Mechanistic Understanding of
      Jailbreak Attacks and Defenses in LLMs"

    原理: 在 attention softmax 前对 logits 除以 T (T < 1)，使模型更聚焦
    输入中的 unsafe prototype / 核心 grading 内容。
    """

    def __init__(
        self,
        temperature: float = 0.5,
        layers: Union[str, List[int]] = "all",
    ):
        super().__init__()
        if temperature <= 0 or temperature >= 1:
            raise ValueError(
                f"attention sharpening temperature must be in (0, 1), got {temperature}"
            )
        self.temperature = temperature
        self.layers = layers
        self._patched_forwards = []

    def requires_model_hooks(self) -> bool:
        return True

    def install_model_hooks(self, model) -> list:
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise ValueError("AttentionSharpening expects a causal LM with model.layers")

        removers = []
        target_layers = self._resolve_layer_indices(model)
        for layer_idx in target_layers:
            attn = model.model.layers[layer_idx].self_attn
            original_forward = attn.forward
            attn.forward = _make_sharpened_forward(original_forward, self.temperature)
            self._patched_forwards.append((attn, original_forward))
            removers.append(lambda attn=attn, orig=original_forward: setattr(attn, "forward", orig))

        return removers

    def _resolve_layer_indices(self, model) -> List[int]:
        num_layers = len(model.model.layers)
        if self.layers == "all":
            return list(range(num_layers))
        if isinstance(self.layers, list):
            return [i for i in self.layers if 0 <= i < num_layers]
        raise ValueError(f"Unsupported layers spec: {self.layers!r}")


def _make_sharpened_forward(original_forward, temperature: float):
    @functools.wraps(original_forward)
    def forward(*args, **kwargs):
        orig_softmax = F.softmax

        def sharpened_softmax(input, dim=None, dtype=None, **kw):
            if input.dim() == 4:
                input = input / temperature
            return orig_softmax(input, dim=dim, dtype=dtype, **kw)

        F.softmax = sharpened_softmax
        try:
            return original_forward(*args, **kwargs)
        finally:
            F.softmax = orig_softmax

    return forward
