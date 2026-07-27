from baselines.defenses import (
    PerplexityFilter, SmoothLLM, SelfReminder,
    ParaphraseDefense, AttentionSharpening, HijackingSuppression, BaseDefense,
    SystemPromptChange,
)
from utils.config_utils import AttackConfig


def _defense_type_key(defense_type: str) -> str:
    return defense_type.lower().replace("-", "").replace("_", "")


def _needs_eager_attention(config: AttackConfig) -> bool:
    if config.debug or config.log_attention:
        return True
    if not config.defenses:
        return False
    eager_types = {"attentionsharpening", "hijackingsuppression"}
    return any(
        _defense_type_key(dc.type) in eager_types
        for dc in config.defenses
    )


def _has_attention_sharpening(config: AttackConfig) -> bool:
    if not config.defenses:
        return False
    return any(
        _defense_type_key(dc.type) == "attentionsharpening"
        for dc in config.defenses
    )


def _load_pipeline_model(model_path: str, device: str, config: AttackConfig):
    import torch
    from transformers import AutoModelForCausalLM

    kwargs = dict(trust_remote_code=True, torch_dtype=torch.bfloat16)
    if _needs_eager_attention(config):
        kwargs["attn_implementation"] = "eager"
        if config.debug:
            print("[grading_attack] Using attn_implementation=eager for debug (attention analysis)", flush=True)
        else:
            print("[grading_attack] Using attn_implementation=eager for attention hook defense", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    return model.to(device)


def _build_defenses(config: AttackConfig, model=None, tokenizer=None):
    """根据配置构建防御模块列表"""
    if not config.defenses:
        return []
    defenses = []
    device = config.params.get("device", "cuda")
    for dc in config.defenses:
        t = dc.type.lower().replace("-", "").replace("_", "")
        if t == "perplexityfilter":
            defenses.append(PerplexityFilter(
                model=model, tokenizer=tokenizer,
                threshold=dc.params.get("threshold", 1000.0),
                device=device,
            ))
        elif t == "smoothllm":
            defenses.append(SmoothLLM(
                num_copies=dc.params.get("num_copies", 5),
                perturb_rate=dc.params.get("perturb_rate", None),
            ))
        elif t == "selfreminder":
            defenses.append(SelfReminder(
                reminder=dc.params.get("reminder", None),
            ))
        elif t == "paraphrasedefense" or t == "paraphrase":
            defenses.append(ParaphraseDefense())
        elif t == "attentionsharpening":
            defenses.append(AttentionSharpening(
                temperature=dc.params.get("temperature", 0.5),
                layers=dc.params.get("layers", "all"),
            ))
        elif t == "hijackingsuppression":
            defenses.append(HijackingSuppression(
                beta=dc.params.get("beta", 0.1),
                top_fraction=dc.params.get("top_fraction", 0.01),
                layers=dc.params.get("layers", "all"),
            ))
        elif t == "systempromptchange":
            defenses.append(SystemPromptChange(
                pre_instruction=dc.params.get("pre_instruction", None),
                post_reminder=dc.params.get("post_reminder", None),
            ))
        else:
            raise ValueError(f"Unknown defense type: {dc.type}")
    return defenses


class GradingAttack:
    def __init__(self, config: AttackConfig):
        self.config = config
        if config.pipeline_mode:
            # pipeline 模式: 在 pipeline.py 中统一处理
            self.attack = None
        elif config.attack_method.lower() == "gcg":
            from baselines.gcg.gcg import GCG
            self.attack = GCG(config)
        elif config.attack_method.lower() == "roleplay":
            from baselines.roleplay.roleplay import RolePlay
            self.attack = RolePlay(config)
        elif config.attack_method.lower() == "injection":
            from baselines.injection.injection import Injection
            self.attack = Injection(config)
        elif config.attack_method.lower() == "gcg_suffix_bank":
            raise ValueError(
                "gcg_suffix_bank requires pipeline_mode=True. "
                "Set pipeline_mode: true in your config."
            )
        else:
            raise ValueError(f"Not supported attack method {config.attack_method}")

    def run(self):
        if self.config.pipeline_mode:
            import torch
            from pipeline import GradingDefensePipeline, _resolve_model_path
            from transformers import AutoTokenizer

            device = self.config.params.get("device") or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            model_path = _resolve_model_path(self.config)
            print(f"[grading_attack] Using device: {device}, dtype: bfloat16", flush=True)
            model = _load_pipeline_model(model_path, device, self.config)
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            defenses = _build_defenses(self.config, model, tokenizer)
            pipeline = GradingDefensePipeline(
                config=self.config,
                model=model,
                tokenizer=tokenizer,
                defenses=defenses,
            )
            pipeline.run()
        elif self.attack is not None:
            self.attack.run()
        else:
            raise RuntimeError(
                "pipeline_mode=True requires defenses configured, "
                "or set pipeline_mode=False for pure attack."
            )
