"""
Attack-Defense 联合评估 Pipeline。

控制流:
    data → attack (GCG/RolePlay) → defenses → evaluate → metrics

参考论文:
    - SmoothLLM (Robey et al., 2023)
    - Baseline Defenses (Jain et al., 2023)
    - GradingAttack (ICLR 2026)
"""

import os
import torch
import random
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download

from utils.config_utils import (
    AttackConfig,
    print_config_summary,
    build_run_metadata,
    format_run_metadata_lines,
    needs_eager_attention,
)


def _resolve_model_path(config: AttackConfig) -> str:
    """Return local model path, downloading via ModelScope if needed."""
    if config.model_config.path:
        return config.model_config.path
    if config.model_config.model_id:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub")
        print(f"[pipeline] Downloading model {config.model_config.model_id} from ModelScope...", flush=True)
        return snapshot_download(config.model_config.model_id, cache_dir=cache_dir)
    raise ValueError("Either model.path or model.model_id must be set in config")

from utils.log_utils import GradingAttackLogger
from utils.data_utils import read_student_qa_data_from_jsonl, AttackResult
from eval.metrics import compute_metrics, EvalMetrics
from baselines.defenses.base import BaseDefense, DefenseRejectException


class GradingDefensePipeline:
    def __init__(self, config: AttackConfig,
                 model: AutoModelForCausalLM = None,
                 tokenizer: AutoTokenizer = None,
                 defenses: List[BaseDefense] = None):
        """
        config: AttackConfig (含 defense 配置)
        model / tokenizer: 可选外部注入，仅 GCG 方式需要
        defenses: 防御模块列表
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.defenses = defenses or []
        self.device = config.params.get("device") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 检查是否有 SmoothLLM 类需要多次推理的防御
        self.multi_gen_defenses = [d for d in self.defenses
                                   if d.requires_multiple_generations()]
        self.context_aware_defenses = any(
            d.requires_inference_context() for d in self.defenses
        )

    def _should_log_attention(self) -> bool:
        return self.config.debug or self.config.log_attention

    def _analyze_attention(
        self,
        prompt_text: str,
        label: str,
        logger: GradingAttackLogger,
        gcg_suffix: str = "",
        sample_idx: Optional[int] = None,
        with_defense_hooks: bool = False,
        hook_prompt_content: str = "",
        hook_attack_suffix: str = "",
    ):
        removers = []
        if with_defense_hooks and self.defenses:
            removers = self._install_defense_hooks(
                hook_prompt_content or prompt_text,
                hook_attack_suffix,
            )
        try:
            from utils.attention_utils import analyze_prompt_attention
            return analyze_prompt_attention(
                self.model,
                self.tokenizer,
                prompt_text,
                gcg_suffix=gcg_suffix,
                label=label,
                sample_idx=sample_idx,
                logger=logger,
                log=True,
            )
        finally:
            if removers:
                self._remove_defense_hooks(removers)

    def _verdict_label(self, response: str) -> str:
        from eval.metrics import _parse_grade
        grade = _parse_grade(response or "", self.config.nclass)
        if grade is None:
            return "unknown"
        return "correct" if grade == 0 else "incorrect"

    @staticmethod
    def _grade_label(grade_int, nclass: int) -> str:
        """Convert int grade (0/1/2) to human-readable label."""
        if grade_int is None:
            return "unknown"
        labels = ("correct", "incorrect") if nclass == 2 else ("correct", "contradictory", "incorrect")
        if 0 <= grade_int < len(labels):
            return labels[grade_int]
        return "unknown"

    @staticmethod
    def _gt_is_correct(data) -> bool:
        """Ground-truth correct samples are not evaluated on attack paths."""
        return (getattr(data, "verification", None) or "").strip().lower() == "correct"

    def _log_sample_verdict(
        self,
        logger: GradingAttackLogger,
        data_idx: int,
        data,
        original_resp: str,
        attacked_resp: str,
        defended_original_resp: Optional[str],
        defended_attacked_resp: Optional[str],
        skip_attack_eval: bool = False,
    ) -> None:
        lines = [
            f"[SAMPLE] idx={data_idx} qid={data.question_id} gt={data.verification}",
            f"  original={self._verdict_label(original_resp)}",
        ]
        if skip_attack_eval:
            lines.append("  attacked=skipped (GT=correct)")
        else:
            lines.append(f"  attacked={self._verdict_label(attacked_resp)}")
        if defended_original_resp is not None:
            lines.append(f"  defense_clean={self._verdict_label(defended_original_resp)}")
        if skip_attack_eval:
            lines.append("  defense_attacked=skipped (GT=correct)")
        elif defended_attacked_resp is not None:
            lines.append(f"  defense_attacked={self._verdict_label(defended_attacked_resp)}")
        for line in lines:
            print(line, flush=True)
            logger.info(line)

    def _install_defense_hooks(self, prompt_content: str, attack_suffix: str) -> list:
        removers = []
        for d in self.defenses:
            if d.requires_inference_context():
                d.set_inference_context(
                    self.tokenizer, prompt_content, attack_suffix
                )
            if d.requires_model_hooks():
                removers.extend(d.install_model_hooks(self.model))
        return removers

    @staticmethod
    def _remove_defense_hooks(removers: list) -> None:
        for remove in removers:
            remove()

    def _generate_with_defense_hooks(
        self,
        messages: list,
        prompt_content: str,
        attack_suffix: str,
    ) -> str:
        removers = self._install_defense_hooks(prompt_content, attack_suffix)
        try:
            return self._generate(messages)
        finally:
            self._remove_defense_hooks(removers)

    def run(self):
        config = self.config
        logger = GradingAttackLogger(config)

        defense_runtime = None
        if self.defenses:
            defense_runtime = {
                "sharpen_clean": True,
                "sharpen_attacked": True,
                "attn_implementation": "eager" if needs_eager_attention(config) else "default",
            }
        log_extra = {"defense_runtime": defense_runtime} if defense_runtime else None
        run_metadata = build_run_metadata(config, extra=log_extra)
        self.run_metadata = run_metadata
        for line in format_run_metadata_lines(run_metadata):
            print(line, flush=True)
            logger.info(line)
        if config.debug:
            print_config_summary(config)

        model_path = _resolve_model_path(config)

        # 如果未传入 model，在此加载 (RolePlay 用 vLLM 则不需)
        if self.model is None:
            load_kwargs = dict(trust_remote_code=True, torch_dtype=torch.bfloat16)
            if any(d.requires_model_hooks() for d in self.defenses):
                load_kwargs["attn_implementation"] = "eager"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **load_kwargs,
            ).to(self.device)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )

        all_clean_attn_summaries = []   # (summary, dataset_name)
        all_attacked_attn_summaries = []
        all_attn_dfs = []

        # ── 后缀银行加载 (gcg_suffix_bank) ──
        bank = None
        bank_source_indices: set[int] = set()
        if config.attack_method.lower() == "gcg_suffix_bank":
            bank = self._load_suffix_bank()
            if config.params.get("exclude_bank_source_indices", True):
                bank_source_indices = self._load_bank_source_indices()
                print(
                    f"[pipeline] Excluding {len(bank_source_indices)} bank source indices "
                    f"from heldout evaluation.",
                    flush=True,
                )

        for data_config in config.data_config:
            data_list = read_student_qa_data_from_jsonl(data_config.path)
            if data_config.max_samples and data_config.max_samples < len(data_list):
                rng = random.Random(data_config.random_seed)
                data_list = rng.sample(data_list, data_config.max_samples)
            all_results = []
            clean_attn_summaries = []
            attacked_attn_summaries = []
            total_samples = len(data_list)
            sample_count = 0  # running counter for non-excluded samples

            for data_idx, data in enumerate(data_list):
                # Skip bank-source samples for fair transfer evaluation
                if bank_source_indices and data_idx in bank_source_indices:
                    print(
                        f"[pipeline] Skipping sample {data_idx}: "
                        f"found in bank source indices",
                        flush=True,
                    )
                    continue

                try:
                    prompt = config.grading_template.format(
                        question=data.question,
                        solution=data.question_answer,
                        student_answer=data.student_answer,
                    )

                    # ── Step 1: 原始推理 ──
                    messages = [{"role": "user", "content": prompt}]
                    attn_meta = {}
                    if self._should_log_attention():
                        s = self._analyze_attention(
                            prompt, "[CLEAN]", logger,
                            sample_idx=data_idx,
                        )
                        if s is not None:
                            clean_attn_summaries.append(s)
                            from utils.attention_utils import attention_summary_to_dict
                            attn_meta["clean"] = attention_summary_to_dict(s)
                    original_resp = self._generate(messages)

                    # GT=correct: skip attack + defense-attack (ASR/eval only targets incorrect GT)
                    end_tag = "</student_answer>"
                    skip_attack_eval = self._gt_is_correct(data)
                    attacked_messages = [{"role": "user", "content": prompt}]
                    attack_suffix = ""
                    bank_meta = None
                    attacked_resp = original_resp

                    if skip_attack_eval:
                        print(
                            f"[pipeline] Skipping attack/defense-attack for sample {data_idx}: "
                            f"GT=correct",
                            flush=True,
                        )
                    else:
                        # ── Step 2: 攻击推理 ──
                        if config.attack_method.lower() == "gcg":
                            import time
                            import nanogcg
                            target = config.params["target"]
                            gcg_config = nanogcg.GCGConfig(**config.params["gcg_config"])

                            # 过滤 Llama reserved_special_token（默认过滤不覆盖这类 token）
                            import nanogcg.utils as _ncg_utils
                            _orig_get_nonascii_toks = _ncg_utils.get_nonascii_toks
                            def _patched_get_nonascii_toks(tokenizer, device="cpu"):
                                not_allowed = _orig_get_nonascii_toks(tokenizer, device)
                                reserved_ids = [
                                    tid for tok, tid in tokenizer.get_vocab().items()
                                    if tok.startswith("<|reserved_special_token_")
                                ]
                                if reserved_ids:
                                    not_allowed = torch.cat([
                                        not_allowed,
                                        torch.tensor(reserved_ids, device=device),
                                    ])
                                return not_allowed
                            _ncg_utils.get_nonascii_toks = _patched_get_nonascii_toks
                            # Blind prompt for GCG optimization (no solution)
                            blind_prompt = config.grading_template.format(
                                question=data.question,
                                solution="",
                                student_answer=data.student_answer,
                            )
                            blind_insert_pos = blind_prompt.rfind(end_tag)
                            if blind_insert_pos != -1:
                                blind_gcq_prompt = blind_prompt[:blind_insert_pos]
                                t_start = time.perf_counter()
                                gcg_result = nanogcg.run(self.model, self.tokenizer,
                                                         [{"role": "user", "content": blind_gcq_prompt}],
                                                         target, gcg_config)
                                t_end = time.perf_counter()
                                attack_suffix = gcg_result.best_string
                                full_insert_pos = prompt.rfind(end_tag)
                                attacked_messages[0]["content"] = prompt[:full_insert_pos] + attack_suffix + end_tag
                            else:
                                t_start = time.perf_counter()
                                gcg_result = nanogcg.run(self.model, self.tokenizer,
                                                         [{"role": "user", "content": blind_prompt}],
                                                         target, gcg_config)
                                t_end = time.perf_counter()
                                attack_suffix = gcg_result.best_string
                                attacked_messages[0]["content"] = prompt + attack_suffix
                            print(f"[GCG] Optimization done in {t_end - t_start:.1f}s", flush=True)
                        elif config.attack_method.lower() == "roleplay":
                            attack_suffix = config.params["adv_prompt"]
                            insert_pos = prompt.rfind(end_tag)
                            if insert_pos != -1:
                                attacked_messages[0]["content"] = prompt[:insert_pos] + attack_suffix + end_tag
                            else:
                                attacked_messages[0]["content"] = prompt + attack_suffix
                        elif config.attack_method.lower() == "injection":
                            attack_suffix = config.params["injection_prompt"]
                            insert_pos = prompt.rfind(end_tag)
                            if insert_pos != -1:
                                attacked_messages[0]["content"] = prompt[:insert_pos] + attack_suffix + end_tag
                            else:
                                attacked_messages[0]["content"] = prompt + attack_suffix
                        elif config.attack_method.lower() == "gcg_suffix_bank":
                            import time as _sbtime
                            from eval.metrics import _parse_grade

                            t_start = _sbtime.perf_counter()
                            # Defaults: no suffix succeeds
                            attacked_resp = original_resp
                            attack_suffix = ""
                            success_suffix_index = None
                            success_suffix = None
                            promotion_suffix_index = None
                            promotion_suffix = None
                            suffixes_tried = 0
                            insert_pos_sb = prompt.rfind(end_tag)
                            original_grade_for_bank = _parse_grade(original_resp, config.nclass)

                            for suffix_idx, suffix in enumerate(bank):
                                suffixes_tried += 1
                                if insert_pos_sb != -1:
                                    test_content = prompt[:insert_pos_sb] + suffix + end_tag
                                else:
                                    test_content = prompt + suffix

                                test_resp = self._generate(
                                    [{"role": "user", "content": test_content}]
                                )
                                test_grade = _parse_grade(test_resp, config.nclass)

                                # Track first promotion (any grade improvement)
                                if (original_grade_for_bank is not None
                                        and test_grade is not None
                                        and test_grade < original_grade_for_bank
                                        and promotion_suffix_index is None):
                                    promotion_suffix_index = suffix_idx
                                    promotion_suffix = suffix

                                # Early stop on correct flip
                                if test_grade == 0:  # 0 = correct
                                    attacked_resp = test_resp
                                    attack_suffix = suffix
                                    success_suffix_index = suffix_idx
                                    success_suffix = suffix
                                    attacked_messages[0]["content"] = test_content
                                    break

                            t_end = _sbtime.perf_counter()
                            print(
                                f"[SuffixBank] Tried {suffixes_tried}/{len(bank)} suffixes "
                                f"in {t_end - t_start:.1f}s, "
                                f"success_idx={success_suffix_index}, "
                                f"promotion_idx={promotion_suffix_index}",
                                flush=True,
                            )

                            bank_meta = {
                                "success_suffix_index": success_suffix_index,
                                "success_suffix": success_suffix,
                                "promotion_suffix_index": promotion_suffix_index,
                                "promotion_suffix": promotion_suffix,
                                "suffixes_tried": suffixes_tried,
                                "bank_path": config.params.get("bank_path"),
                                "bank_size": len(bank),
                            }

                        if self._should_log_attention():
                            s = self._analyze_attention(
                                attacked_messages[0]["content"],
                                "[ATTACKED]",
                                logger,
                                gcg_suffix=attack_suffix,
                                sample_idx=data_idx,
                            )
                            if s is not None:
                                attacked_attn_summaries.append(s)
                                from utils.attention_utils import attention_summary_to_dict
                                attn_meta["attacked"] = attention_summary_to_dict(s)
                        # gcg_suffix_bank generates inside its branch (multi-suffix iteration)
                        if config.attack_method.lower() != "gcg_suffix_bank":
                            attacked_resp = self._generate(attacked_messages)

                    # ── Step 3: 防御推理 ──
                    defended_original_resp = None
                    defended_attacked_resp = None
                    rejected = False

                    if self.defenses:
                        try:
                            if self.multi_gen_defenses:
                                if self.context_aware_defenses:
                                    defended_original_resp = self._generate_with_defense_hooks(
                                        [{"role": "user", "content": prompt}],
                                        prompt,
                                        "",
                                    )
                                    if not skip_attack_eval:
                                        defended_attacked_resp = self._generate_with_defense_hooks(
                                            attacked_messages,
                                            attacked_messages[0]["content"],
                                            attack_suffix,
                                        )
                                else:
                                    hook_removers = self._install_defense_hooks(
                                        prompt, attack_suffix
                                    )
                                    try:
                                        defended_original_resp = self._generate_with_voting(
                                            [{"role": "user", "content": prompt}]
                                        )
                                        if not skip_attack_eval:
                                            defended_attacked_resp = self._generate_with_voting(
                                                attacked_messages
                                            )
                                    finally:
                                        self._remove_defense_hooks(hook_removers)
                                for d in self.defenses:
                                    defended_original_resp = d.post_process(
                                        defended_original_resp or ""
                                    )
                                    if not skip_attack_eval:
                                        defended_attacked_resp = d.post_process(
                                            defended_attacked_resp or ""
                                        )
                            else:
                                defended_prompt = prompt
                                for d in self.defenses:
                                    defended_prompt = d.pre_process(defended_prompt)

                                if config.attack_method.lower() in ("gcg", "gcg_suffix_bank"):
                                    defended_insert_pos = defended_prompt.rfind(end_tag)
                                    if defended_insert_pos != -1:
                                        defended_attacked_content = (
                                            defended_prompt[:defended_insert_pos]
                                            + attack_suffix + end_tag
                                        )
                                    else:
                                        defended_attacked_content = defended_prompt + attack_suffix
                                elif config.attack_method.lower() == "roleplay":
                                    rp_insert = defended_prompt.rfind(end_tag)
                                    if rp_insert != -1:
                                        defended_attacked_content = (
                                            defended_prompt[:rp_insert]
                                            + attack_suffix + end_tag
                                        )
                                    else:
                                        defended_attacked_content = defended_prompt + attack_suffix
                                elif config.attack_method.lower() == "injection":
                                    rp_insert = defended_prompt.rfind(end_tag)
                                    if rp_insert != -1:
                                        defended_attacked_content = (
                                            defended_prompt[:rp_insert]
                                            + attack_suffix + end_tag
                                        )
                                    else:
                                        defended_attacked_content = defended_prompt + attack_suffix
                                else:
                                    defended_attacked_content = defended_prompt + attack_suffix

                                if self.context_aware_defenses:
                                    defended_original_resp = self._generate_with_defense_hooks(
                                        [{"role": "user", "content": defended_prompt}],
                                        defended_prompt,
                                        "",
                                    )
                                    if not skip_attack_eval:
                                        defended_attacked_resp = self._generate_with_defense_hooks(
                                            [{"role": "user", "content": defended_attacked_content}],
                                            defended_attacked_content,
                                            attack_suffix,
                                        )
                                else:
                                    hook_removers = self._install_defense_hooks(
                                        defended_prompt, attack_suffix
                                    )
                                    try:
                                        defended_original_resp = self._generate(
                                            [{"role": "user", "content": defended_prompt}]
                                        )
                                        if not skip_attack_eval:
                                            defended_attacked_resp = self._generate(
                                                [{"role": "user", "content": defended_attacked_content}]
                                            )
                                    finally:
                                        self._remove_defense_hooks(hook_removers)

                                for d in self.defenses:
                                    defended_original_resp = d.post_process(
                                        defended_original_resp or ""
                                    )
                                    if not skip_attack_eval:
                                        defended_attacked_resp = d.post_process(
                                            defended_attacked_resp or ""
                                        )

                        except DefenseRejectException:
                            rejected = True

                    if self._should_log_attention() and self.defenses and not rejected:
                        defended_prompt_for_attn = prompt
                        for d in self.defenses:
                            defended_prompt_for_attn = d.pre_process(defended_prompt_for_attn)
                        if config.attack_method.lower() in ("gcg", "gcg_suffix_bank"):
                            defended_insert_pos = defended_prompt_for_attn.rfind(end_tag)
                            if defended_insert_pos != -1:
                                defended_attacked_for_attn = (
                                    defended_prompt_for_attn[:defended_insert_pos]
                                    + attack_suffix + end_tag
                                )
                            else:
                                defended_attacked_for_attn = defended_prompt_for_attn + attack_suffix
                        elif config.attack_method.lower() == "roleplay":
                            rp_insert = defended_prompt_for_attn.rfind(end_tag)
                            if rp_insert != -1:
                                defended_attacked_for_attn = (
                                    defended_prompt_for_attn[:rp_insert]
                                    + attack_suffix + end_tag
                                )
                            else:
                                defended_attacked_for_attn = defended_prompt_for_attn + attack_suffix
                        elif config.attack_method.lower() == "injection":
                            rp_insert = defended_prompt_for_attn.rfind(end_tag)
                            if rp_insert != -1:
                                defended_attacked_for_attn = (
                                    defended_prompt_for_attn[:rp_insert]
                                    + attack_suffix + end_tag
                                )
                            else:
                                defended_attacked_for_attn = defended_prompt_for_attn + attack_suffix
                        else:
                            defended_attacked_for_attn = defended_prompt_for_attn + attack_suffix

                        from utils.attention_utils import attention_summary_to_dict
                        s = self._analyze_attention(
                            defended_prompt_for_attn,
                            "[DEFENSE-CLEAN]",
                            logger,
                            sample_idx=data_idx,
                            with_defense_hooks=True,
                            hook_prompt_content=prompt,
                            hook_attack_suffix="",
                        )
                        if s is not None:
                            attn_meta["defense_clean"] = attention_summary_to_dict(s)
                        if not skip_attack_eval:
                            s = self._analyze_attention(
                                defended_attacked_for_attn,
                                "[DEFENSE-ATTACKED]",
                                logger,
                                gcg_suffix=attack_suffix,
                                sample_idx=data_idx,
                                with_defense_hooks=True,
                                hook_prompt_content=prompt,
                                hook_attack_suffix=attack_suffix,
                            )
                            if s is not None:
                                attn_meta["defense_attacked"] = attention_summary_to_dict(s)

                    if self._should_log_attention():
                        self._log_sample_verdict(
                            logger,
                            data_idx,
                            data,
                            original_resp,
                            attacked_resp,
                            defended_original_resp,
                            defended_attacked_resp,
                            skip_attack_eval=skip_attack_eval,
                        )

                    # ── Step 4: 记录结果 ──
                    # Parse grades for output and progress printing
                    from eval.metrics import _parse_grade as _pg
                    orig_grade_int = _pg(original_resp, config.nclass)
                    attacked_grade_int = _pg(attacked_resp, config.nclass)
                    def_orig_grade_int = _pg(defended_original_resp or "", config.nclass)
                    def_attacked_grade_int = _pg(defended_attacked_resp or "", config.nclass)

                    sample_meta = {
                        "defended_original_response": defended_original_resp,
                        "defended_attacked_response": defended_attacked_resp,
                        "rejected": rejected,
                        "attention": attn_meta or None,
                        "skip_attack_eval": skip_attack_eval,
                    }
                    if bank_meta:
                        sample_meta.update(bank_meta)

                    result = AttackResult(
                        student_qa_data=data,
                        original_response=original_resp,
                        attacked_response=attacked_resp,
                        meta=sample_meta,
                    )
                    result_dict = result.as_dict()

                    # Enrich with flat top-level fields (matching HPC output format)
                    result_dict["source_index"] = data_idx
                    result_dict["original_grade"] = self._grade_label(orig_grade_int, config.nclass)
                    result_dict["attacked_grade"] = self._grade_label(attacked_grade_int, config.nclass)
                    if defended_original_resp:
                        result_dict["defended_original_grade"] = self._grade_label(
                            def_orig_grade_int, config.nclass
                        )
                    if defended_attacked_resp:
                        result_dict["defended_attacked_grade"] = self._grade_label(
                            def_attacked_grade_int, config.nclass
                        )

                    # Flatten bank_meta fields to top level for gcg_suffix_bank
                    if bank_meta:
                        result_dict["success"] = bank_meta.get("success_suffix_index") is not None
                        result_dict["promotion_success"] = (
                            bank_meta.get("promotion_suffix_index") is not None
                        )
                        for key in (
                            "success_suffix_index", "success_suffix",
                            "promotion_suffix_index", "promotion_suffix",
                            "suffixes_tried", "bank_path", "bank_size",
                        ):
                            if key in bank_meta:
                                result_dict[key] = bank_meta[key]

                    logger.result(result_dict)
                    all_results.append(result_dict)

                    # ── Per-sample progress print ──
                    sample_count += 1
                    label = data.verification or "?"
                    orig_str = self._grade_label(orig_grade_int, config.nclass)
                    if skip_attack_eval:
                        attacked_str = "skipped"
                    else:
                        attacked_str = self._grade_label(attacked_grade_int, config.nclass)
                    success_str = ""
                    if bank_meta is not None:
                        si = bank_meta.get("success_suffix_index")
                        success_str = f" success={'yes' if si is not None else 'no'}"
                    print(
                        f"[{sample_count}/{total_samples}] source_idx={data_idx} "
                        f"label={label} orig={orig_str} attacked={attacked_str}"
                        f"{success_str}",
                        flush=True,
                    )

                except Exception as e:
                    import traceback
                    print(f"[ERROR] Sample {data_idx} failed: {e}", flush=True)
                    traceback.print_exc()

            # ── 打印 attention 平均统计 ──
            if self._should_log_attention() and (clean_attn_summaries or attacked_attn_summaries):
                from utils.attention_utils import (
                    print_clean_attention_stats,
                    print_attacked_attention_stats,
                    build_attention_dataframe,
                    format_attention_aggregate_lines,
                )
                ds_label = f"[CLEAN AVERAGE {data_config.name}]"
                if clean_attn_summaries:
                    all_clean_attn_summaries.extend(
                        (s, data_config.name) for s in clean_attn_summaries
                    )
                    print_clean_attention_stats(clean_attn_summaries, label=ds_label)
                    for line in format_attention_aggregate_lines(
                        clean_attn_summaries, ds_label
                    ):
                        logger.info(line)
                if attacked_attn_summaries:
                    all_attacked_attn_summaries.extend(
                        (s, data_config.name) for s in attacked_attn_summaries
                    )
                    atk_label = f"[ATTACKED AVERAGE {data_config.name}]"
                    print_attacked_attention_stats(attacked_attn_summaries, label=atk_label)
                    for line in format_attention_aggregate_lines(
                        attacked_attn_summaries, atk_label
                    ):
                        logger.info(line)
                df = build_attention_dataframe(clean_attn_summaries,
                                               attacked_attn_summaries,
                                               data_config.name,
                                               results=all_results)
                all_attn_dfs.append(df)

            # ── Step 5: 计算指标 ──
            if all_results:
                flat_results = []
                for r in all_results:
                    flat_r = {
                        "student_qa_data": r["student_qa_data"],
                        "original_response": r["original_response"],
                        "attacked_response": r["attacked_response"],
                        "defended_original_response": r.get("meta", {}).get(
                            "defended_original_response") or "",
                        "defended_attacked_response": r.get("meta", {}).get(
                            "defended_attacked_response") or "",
                    }
                    flat_results.append(flat_r)

                try:
                    metrics: EvalMetrics = compute_metrics(
                        flat_results,
                        nclass=config.nclass,
                        print_confusion_matrix=config.log_attention,
                    )
                except Exception as e:
                    import traceback
                    print(f"[ERROR] compute_metrics failed: {e}", flush=True)
                    traceback.print_exc()
                    logger.info(f"compute_metrics failed: {e}")
                    all_results.clear()
                    continue

                if self.defenses:
                    logger.info(
                        f"[{config.model_config.name}] [{data_config.name}] "
                        f"QWK_clean={metrics.qwk_clean:.4f} "
                        f"QWK_attack={metrics.qwk_attack:.4f} "
                        f"ASR={metrics.asr:.4f} "
                        f"ASR_defended={metrics.asr_defended:.4f} "
                        f"CAS={metrics.cas:.4f} "
                        f"CAS_defended={metrics.cas_defended:.4f}"
                    )
                    print(
                        f"\n[SUMMARY {data_config.name}] "
                        f"QWK_clean={metrics.qwk_clean:.4f} "
                        f"QWK_attack={metrics.qwk_attack:.4f} "
                        f"ASR={metrics.asr:.4f} ({metrics.total_attack_eligible} eligible) "
                        f"ASR_defended={metrics.asr_defended:.4f} ({metrics.total_defense_eligible} eligible)\n",
                        flush=True,
                    )
                else:
                    logger.info(
                        f"[{config.model_config.name}] [{data_config.name}] "
                        f"QWK_clean={metrics.qwk_clean:.4f} "
                        f"QWK_attack={metrics.qwk_attack:.4f} "
                        f"ASR={metrics.asr:.4f} "
                        f"CAS={metrics.cas:.4f}"
                    )
                    print(
                        f"\n[SUMMARY {data_config.name}] "
                        f"QWK_clean={metrics.qwk_clean:.4f} "
                        f"QWK_attack={metrics.qwk_attack:.4f} "
                        f"ASR={metrics.asr:.4f} ({metrics.total_attack_eligible} eligible)\n",
                        flush=True,
                    )

                # ── Step 6: 保存指标 + 配置摘要 ──
                import json
                summary = metrics.to_dict()
                summary["config"] = {
                    "name": config.name,
                    "attack_method": config.attack_method,
                    "model": config.model_config.name,
                    "model_path": config.model_config.path,
                    "template": "ci",
                    "datasets": [d.name for d in config.data_config],
                    "defenses": [d.__class__.__name__ for d in self.defenses],
                    "nclass": config.nclass,
                    "params": config.params,
                }
                try:
                    with open(logger.metrics_path, "w", encoding="utf-8") as f:
                        json.dump(summary, f, indent=2, ensure_ascii=False)
                    logger.info(f"Metrics saved to {logger.metrics_path}")
                except Exception as e:
                    import traceback
                    print(f"[ERROR] Failed to save metrics: {e}", flush=True)
                    traceback.print_exc()
                    logger.info(f"Failed to save metrics: {e}")

                # ── Step 6b: 后缀银行迁移指标 ──
                if config.attack_method.lower() == "gcg_suffix_bank" and all_results:
                    from eval.bank_metrics import compute_bank_transfer_metrics
                    try:
                        bank_metrics = compute_bank_transfer_metrics(
                            all_results, nclass=config.nclass
                        )
                        bank_summary = bank_metrics.to_dict()
                        bank_metrics_path = logger.metrics_path.replace(
                            "_metrics.json", "_bank_metrics.json"
                        )
                        with open(bank_metrics_path, "w", encoding="utf-8") as f:
                            json.dump(bank_summary, f, indent=2, ensure_ascii=False)
                        logger.info(
                            f"[{data_config.name}] "
                            f"BankASR={bank_metrics.project_asr:.4f} "
                            f"PromoASR={bank_metrics.promotion_asr:.4f} "
                            f"TransferRate={bank_metrics.transfer_rate:.4f} "
                            f"MeanIdx={bank_metrics.mean_success_index:.1f}"
                        )
                        logger.info(f"Bank metrics saved to {bank_metrics_path}")
                        print(
                            f"\n[BANK SUMMARY {data_config.name}] "
                            f"BankSize={bank_metrics.bank_size} "
                            f"TotalNonCorrect={bank_metrics.total_non_correct} "
                            f"Eligible={bank_metrics.eligible_non_correct} "
                            f"ProjectASR={bank_metrics.project_asr:.4f} "
                            f"({bank_metrics.project_asr_success}/{bank_metrics.eligible_non_correct}) "
                            f"PromoASR={bank_metrics.promotion_asr:.4f} "
                            f"TransferRate={bank_metrics.transfer_rate:.4f} "
                            f"MeanSuccessIdx={bank_metrics.mean_success_index:.1f}\n",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"[ERROR] compute_bank_transfer_metrics failed: {e}", flush=True)
                        logger.info(f"compute_bank_transfer_metrics failed: {e}")

            logger.info(
                f"[MODEL] {config.model_config.name}  "
                + f"[DATASET] {data_config.name}  "
                + f"Finished"
            )

        # ── 跨数据集总体 attention 统计 ──
        if self._should_log_attention() and (all_clean_attn_summaries or all_attacked_attn_summaries):
            from utils.attention_utils import (
                print_clean_attention_stats,
                print_attacked_attention_stats,
                format_attention_aggregate_lines,
            )
            if all_clean_attn_summaries:
                clean_only = [s for s, _ in all_clean_attn_summaries]
                print_clean_attention_stats(clean_only, label="[CLEAN AVERAGE OVERALL]")
                for line in format_attention_aggregate_lines(
                    clean_only, "[CLEAN AVERAGE OVERALL]"
                ):
                    logger.info(line)
            if all_attacked_attn_summaries:
                atk_only = [s for s, _ in all_attacked_attn_summaries]
                print_attacked_attention_stats(atk_only, label="[ATTACKED AVERAGE OVERALL]")
                for line in format_attention_aggregate_lines(
                    atk_only, "[ATTACKED AVERAGE OVERALL]"
                ):
                    logger.info(line)

        # ── CSV 导出 ──
        if self._should_log_attention() and all_attn_dfs:
            import pandas as pd
            combined_df = pd.concat(all_attn_dfs, ignore_index=True)
            csv_path = os.path.splitext(logger.result_path)[0] + "_attention.csv"
            combined_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"[ATTENTION] DataFrame saved to {csv_path}  ({len(combined_df)} rows)", flush=True)
            logger.info(f"Attention DataFrame saved to {csv_path}")

            txt_path = os.path.splitext(logger.result_path)[0] + "_attention.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                for line in format_run_metadata_lines(self.run_metadata):
                    f.write(line + "\n")
            print(f"[ATTENTION] Experiment metadata saved to {txt_path}", flush=True)
            logger.info(f"Experiment metadata saved to {txt_path}")

    def _generate(self, messages: list) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        if self.config.debug:
            print(f"{'='*60}", flush=True)
            print(f"[DEBUG] _generate() called", flush=True)
            print(f"[DEBUG] messages = {__import__('json').dumps(messages, indent=2, ensure_ascii=False)}", flush=True)
            gen_args = dict(
                do_sample=False,
                max_new_tokens=self.config.generation_config.max_tokens,
                temperature=self.config.generation_config.temperature,
            )
            print(f"[DEBUG] generate_kwargs = {__import__('json').dumps(gen_args, indent=2)}", flush=True)
            print(f"[DEBUG] device = {self.device}", flush=True)
            print(f"[DEBUG] tokenized prompt length = {inputs.shape[1]} tokens", flush=True)

        print(f"[pipeline] Generating {self.config.generation_config.max_tokens} tokens on {self.device}...", flush=True)
        outputs = self.model.generate(
            inputs,
            do_sample=False,
            max_new_tokens=self.config.generation_config.max_tokens,
            temperature=self.config.generation_config.temperature,
        )
        response_text = self.tokenizer.batch_decode(
            outputs[:, inputs.shape[1]:], skip_special_tokens=True
        )[0]
        print(f"[pipeline] Done. Generated {outputs.shape[1] - inputs.shape[1]} tokens.", flush=True)

        if self.config.debug:
            print(f"[DEBUG] response = {__import__('json').dumps(response_text, indent=2, ensure_ascii=False)}", flush=True)
            print(f"{'='*60}", flush=True)

        return response_text

    def _generate_with_voting(self, messages: list) -> str:
        """对 SmoothLLM 类防御：生成多个扰动版本，多数投票"""
        prompt = messages[0]["content"]
        all_variants = [prompt]
        for d in self.multi_gen_defenses:
            variants = d.generate_variants(prompt)
            all_variants = variants  # 取最后一个 multi_gen 的变体

        responses = []
        for variant in all_variants:
            msgs = [{"role": "user", "content": variant}]
            resp = self._generate(msgs)
            responses.append(resp)

        # 多数投票：从所有响应中提取 grade
        from collections import Counter
        grades = []
        for r in responses:
            g = __import__('utils.data_utils', fromlist=['extract_grade']).extract_grade(r)
            grades.append(g)
        counter = Counter(g for g in grades if g is not None)
        if counter:
            majority_grade = counter.most_common(1)[0][0]
            # 返回投票最多的 grade 对应的完整响应
            # 若没有匹配的，返回第一个非 None 响应
            for i, g in enumerate(grades):
                if g == majority_grade:
                    return responses[i]
        return responses[0] if responses else ""

    def _load_suffix_bank(self) -> list[str]:
        """Load and cache the GCG suffix bank from config.params['bank_path'].

        Supports .jsonl (one dict per line with 'best_string'/'suffix' key),
        .json (list of strings or dict with 'suffixes'/'bank' key),
        and .txt (one suffix per line).
        """
        if getattr(self, '_bank_cache', None) is not None:
            return self._bank_cache
        import json
        from pathlib import Path

        path = Path(self.config.params["bank_path"])
        if not path.exists():
            raise FileNotFoundError(f"Suffix bank not found: {path}")

        suffixes: list[str] = []
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        sfx = self._suffix_from_item(item)
                        if sfx:
                            suffixes.append(sfx)
        elif path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            items = data if isinstance(data, list) else data.get("suffixes", data.get("bank", []))
            for item in items:
                sfx = self._suffix_from_item(item)
                if sfx:
                    suffixes.append(sfx)
        else:
            # Plain text: one suffix per line
            suffixes = [s.rstrip("\n") for s in path.read_text(encoding="utf-8").splitlines()]
            suffixes = [s for s in suffixes if s.strip()]

        deduped = list(dict.fromkeys(suffixes))  # preserve order
        limit = self.config.params.get("bank_limit")
        self._bank_cache = deduped[:limit] if limit else deduped
        print(f"[pipeline] Loaded {len(self._bank_cache)} suffixes from bank: {path}", flush=True)
        return self._bank_cache

    @staticmethod
    def _suffix_from_item(item) -> str | None:
        """Extract suffix string from a bank entry dict."""
        if isinstance(item, str):
            return item
        if not isinstance(item, dict):
            return None
        for key in ("suffix", "best_string", "gcg_suffix"):
            if item.get(key):
                return str(item[key])
        return None

    def _load_bank_source_indices(self) -> set[int]:
        """Load source_index values from bank entries for heldout exclusion."""
        if getattr(self, '_bank_source_indices_cache', None) is not None:
            return self._bank_source_indices_cache
        import json
        from pathlib import Path

        path = Path(self.config.params["bank_path"])
        indices: set[int] = set()
        if not path.exists() or path.suffix not in {".json", ".jsonl"}:
            self._bank_source_indices_cache = indices
            return indices
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if isinstance(item, dict) and isinstance(item.get("source_index"), int):
                            indices.add(item["source_index"])
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            items = data if isinstance(data, list) else data.get("suffixes", data.get("bank", []))
            for item in items:
                if isinstance(item, dict) and isinstance(item.get("source_index"), int):
                    indices.add(item["source_index"])
        self._bank_source_indices_cache = indices
        return indices
