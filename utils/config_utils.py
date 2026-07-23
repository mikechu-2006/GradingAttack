import os
import yaml
from typing import Optional, Dict, List
from dataclasses import dataclass, field, asdict
from collections.abc import Iterable


@dataclass
class ModelConfig:
    name: Optional[str] = None
    path: Optional[str] = None
    model_id: Optional[str] = None


@dataclass
class DataConfig:
    name: Optional[str] = None
    path: Optional[str] = None
    max_samples: Optional[int] = None
    random_seed: Optional[int] = None


@dataclass
class GenerationConfig:
    temperature: float = 0.01
    max_tokens: int = 1024

    def as_dict(self):
        return asdict(self)


@dataclass
class LogConfig:
    log_dir: str = "./logs"
    result_dir: str = "./result"


@dataclass
class DefenseConfig:
    type: str = ""
    params: Dict = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class AttackConfig:
    name: Optional[str] = None
    model_config: Optional[ModelConfig] = None
    data_config: Optional[List[DataConfig]] = None
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    log_config: LogConfig = field(default_factory=LogConfig)
    attack_method: str = "GCG"
    params: Optional[Dict] = None
    grading_template: Optional[str] = None
    defenses: Optional[List[DefenseConfig]] = None
    pipeline_mode: bool = False
    debug: bool = False
    log_attention: bool = False
    nclass: int = 3


def parse_config(path: str) -> AttackConfig:
    with open(path, "r", encoding="utf-8") as config_file:
        config_dict = yaml.safe_load(config_file)
    
    if "name" in config_dict:
        name = config_dict["name"]
    else:
        name = os.path.split(os.path.basename(path))[0]
    
    model_config = ModelConfig(**config_dict["model"])
    
    if isinstance(config_dict["data"], Iterable):
        data_config = [DataConfig(**d) for d in config_dict["data"]]
    else:
        data_config = [DataConfig(**config_dict["data"])]

    if "generation" in config_dict:
        generation_config = GenerationConfig(**config_dict["generation"])
    else:
        generation_config = GenerationConfig()

    if "log" in config_dict:
        log_config = LogConfig(**config_dict["log"])
    else:
        log_config = LogConfig()

    if "params" in config_dict:
        params = config_dict["params"]
    else:
        params = {}
    
    generation_config = GenerationConfig(**config_dict["generation"])
    attack_method = config_dict["method"]
    nclass = config_dict.get("nclass", 3)
    grading_template = config_dict.get("grading_template")
    if grading_template is None:
        template_name = config_dict.get("template", "ci")
        template_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "configs", f"grading_template_{template_name}_{nclass}c.txt"
        )
        with open(template_path, "r", encoding="utf-8") as f:
            grading_template = f.read()


    defenses = None
    if "defenses" in config_dict and config_dict["defenses"]:
        defenses = [DefenseConfig(**d) for d in config_dict["defenses"]]

    pipeline_mode = config_dict.get("pipeline_mode", False)
    debug = config_dict.get("debug", False)
    log_attention = config_dict.get("log_attention", False)

    return AttackConfig(
        name=name,
        model_config=model_config,
        data_config=data_config,
        generation_config=generation_config,
        log_config=log_config,
        attack_method=attack_method,
        params=params,
        grading_template=grading_template,
        defenses=defenses,
        pipeline_mode=pipeline_mode,
        debug=debug,
        log_attention=log_attention,
        nclass=nclass,
    )


def _defense_type_key(defense_type: str) -> str:
    return defense_type.lower().replace("-", "").replace("_", "")


def has_attention_sharpening(config: AttackConfig) -> bool:
    if not config.defenses:
        return False
    return any(
        _defense_type_key(dc.type) == "attentionsharpening"
        for dc in config.defenses
    )


def needs_eager_attention(config: AttackConfig) -> bool:
    if config.debug:
        return True
    if not config.defenses:
        return False
    eager_types = {"attentionsharpening", "hijackingsuppression"}
    return any(
        _defense_type_key(dc.type) in eager_types
        for dc in config.defenses
    )


def build_run_metadata(config: AttackConfig, extra: Optional[Dict] = None) -> Dict:
    """Structured experiment metadata for metrics JSON and run logs."""
    mc = config.model_config
    gc = config.generation_config
    metadata = {
        "name": config.name,
        "attack_method": config.attack_method,
        "model": mc.name,
        "model_path": mc.path,
        "model_id": mc.model_id,
        "template": getattr(config, "template", "ci"),
        "nclass": config.nclass,
        "log_attention": config.log_attention,
        "generation": gc.as_dict(),
        "data": [asdict(dc) for dc in config.data_config],
        "params": config.params or {},
        "defenses": [
            {"type": dc.type, "params": dict(dc.params or {})}
            for dc in (config.defenses or [])
        ],
    }
    if extra:
        metadata.update(extra)
    return metadata


def format_run_metadata_lines(metadata: Dict) -> List[str]:
    """Compact, stable log lines shared by stdout and .log files."""
    lines = [
        "=" * 60,
        "[RUN_CONFIG] Experiment metadata",
        "=" * 60,
        f"  name:          {metadata.get('name')}",
        f"  attack_method: {metadata.get('attack_method')}",
        f"  model:         {metadata.get('model')}",
        f"  model_path:    {metadata.get('model_path') or '(auto-download)'}",
        f"  model_id:      {metadata.get('model_id') or '(not set)'}",
        f"  template:      {metadata.get('template')}",
        f"  nclass:        {metadata.get('nclass')}",
        f"  log_attention: {metadata.get('log_attention')}",
        "",
    ]

    gen = metadata.get("generation") or {}
    lines.extend([
        "  [generation]",
        f"    temperature:   {gen.get('temperature')}",
        f"    max_tokens:    {gen.get('max_tokens')}",
        "",
    ])

    for i, dc in enumerate(metadata.get("data") or [], start=1):
        lines.extend([
            f"  [data #{i}]",
            f"    name:          {dc.get('name')}",
            f"    path:          {dc.get('path')}",
            f"    max_samples:   {dc.get('max_samples') or '(all)'}",
            f"    random_seed:   {dc.get('random_seed') or '(not set)'}",
            "",
        ])

    defenses = metadata.get("defenses") or []
    if defenses:
        for i, dc in enumerate(defenses, start=1):
            lines.append(f"  [defense #{i}]")
            lines.append(f"    type:          {dc.get('type')}")
            for k, v in (dc.get("params") or {}).items():
                lines.append(f"    {k}: {v}")
            lines.append("")
    else:
        lines.extend(["  [defenses] (none)", ""])

    runtime = metadata.get("defense_runtime")
    if runtime:
        lines.append("  [defense_runtime]")
        for k, v in runtime.items():
            lines.append(f"    {k}: {v}")
        lines.append("")

    lines.append("=" * 60)
    return lines


def log_run_metadata(config: AttackConfig, logger=None, extra: Optional[Dict] = None) -> Dict:
    """Print and optionally persist experiment metadata at run start."""
    metadata = build_run_metadata(config, extra=extra)
    for line in format_run_metadata_lines(metadata):
        print(line, flush=True)
        if logger is not None:
            logger.info(line)
    return metadata


def print_config_summary(config: AttackConfig):
    """Print all configuration parameters at the start of a run."""
    import json
    print("=" * 60, flush=True)
    print("[CONFIG] Run Configuration Summary", flush=True)
    print("=" * 60, flush=True)
    print(f"  name:            {config.name}", flush=True)
    print(f"  attack_method:   {config.attack_method}", flush=True)
    print(f"  pipeline_mode:   {config.pipeline_mode}", flush=True)
    print(f"  debug:           {config.debug}", flush=True)
    print(f"  log_attention:   {config.log_attention}", flush=True)
    print(f"  nclass:          {config.nclass}-class", flush=True)
    print(f"", flush=True)

    # Model
    mc = config.model_config
    print(f"  [model]", flush=True)
    print(f"    name:          {mc.name}", flush=True)
    print(f"    path:          {mc.path or '(auto-download from ModelScope)'}", flush=True)
    print(f"    model_id:      {mc.model_id or '(not set)'}", flush=True)
    print(f"", flush=True)

    # Generation
    gc = config.generation_config
    print(f"  [generation]", flush=True)
    print(f"    temperature:   {gc.temperature}", flush=True)
    print(f"    max_tokens:    {gc.max_tokens}", flush=True)
    print(f"", flush=True)

    # Log
    lc = config.log_config
    print(f"  [log]", flush=True)
    print(f"    log_dir:       {lc.log_dir}", flush=True)
    print(f"    result_dir:    {lc.result_dir}", flush=True)
    print(f"", flush=True)

    # Data
    for i, dc in enumerate(config.data_config):
        print(f"  [data #{i+1}]", flush=True)
        print(f"    name:          {dc.name}", flush=True)
        print(f"    path:          {dc.path}", flush=True)
        print(f"    max_samples:   {dc.max_samples or '(all)'}", flush=True)
        print(f"    random_seed:   {dc.random_seed or '(not set)'}", flush=True)
        print(f"", flush=True)

    # Params
    if config.params:
        print(f"  [params]", flush=True)
        for k, v in config.params.items():
            if isinstance(v, dict):
                print(f"    {k}:", flush=True)
                for subk, subv in v.items():
                    print(f"      {subk}: {subv}", flush=True)
            else:
                val_str = str(v)
                if len(val_str) > 80:
                    val_str = val_str[:77] + "..."
                print(f"    {k}: {val_str}", flush=True)
        print(f"", flush=True)

    # Defenses
    if config.defenses:
        for i, dc in enumerate(config.defenses):
            print(f"  [defense #{i+1}]", flush=True)
            print(f"    type:          {dc.type}", flush=True)
            if dc.params:
                for k, v in dc.params.items():
                    val_str = str(v)
                    if len(val_str) > 80:
                        val_str = val_str[:77] + "..."
                    print(f"    {k}: {val_str}", flush=True)
            print(f"", flush=True)
    else:
        print(f"  [defenses] (none)", flush=True)
        print(f"", flush=True)

    # Grading template
    print(f"  [grading_template] ({len(config.grading_template)} chars)", flush=True)
    print(f"    {config.grading_template[:200]}...", flush=True)
    print(f"", flush=True)

    # Environment
    import torch, sys, platform
    print(f"  [environment]", flush=True)
    print(f"    Python:    {sys.version.split()[0]}", flush=True)
    print(f"    PyTorch:   {torch.__version__}", flush=True)
    print(f"    CUDA:      {torch.version.cuda or 'N/A'}", flush=True)
    print(f"    GPU:       {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}", flush=True)
    print(f"    Platform:  {platform.platform()}", flush=True)
    print("=" * 60, flush=True)
