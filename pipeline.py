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

        for data_config in config.data_config:
            data_list = read_student_qa_data_from_jsonl(data_config.path)
            if data_config.max_samples and data_config.max_samples < len(data_list):
                rng = random.Random(data_config.random_seed)
                data_list = rng.sample(data_list, data_config.max_samples)
            all_results = []
            clean_attn_summaries = []
            attacked_attn_summaries = []

            for data_idx, data in enumerate(data_list):
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

                    # ── Step 2: 攻击推理 ──
                    attacked_messages = [{"role": "user", "content": prompt}]
                    attack_suffix = ""
                    if config.attack_method.lower() == "gcg":
                        import time
                        import nanogcg
                        target = config.params["target"]
                        gcg_config = nanogcg.GCGConfig(**config.params["gcg_config"])
                        # Blind prompt for GCG optimization (no solution)
                        blind_prompt = config.grading_template.format(
                            question=data.question,
                            solution="",
                            student_answer=data.student_answer,
                        )
                        end_tag = "</student_answer>"
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
                        end_tag = "</student_answer>"
                        insert_pos = prompt.rfind(end_tag)
                        if insert_pos != -1:
                            attacked_messages[0]["content"] = prompt[:insert_pos] + attack_suffix + end_tag
                        else:
                            attacked_messages[0]["content"] = prompt + attack_suffix

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
                                    defended_attacked_resp = self._generate_with_defense_hooks(
                                        attacked_messages,
                                        attacked_messages[0]["content"],
                                        attack_suffix,
                                    )
                                else:
                                    hook_removers = self._install_defense_hooks(
                                        prompt, attack_suffix
                                    )
                                else:
                                    try:
                                        defended_original_resp = self._generate_with_voting(
                                            [{"role": "user", "content": prompt}]
                                        )
                                        defended_attacked_resp = self._generate_with_voting(
                                            attacked_messages
                                        )
                                    finally:
                                        self._remove_defense_hooks(hook_removers)
                                for d in self.defenses:
                                    defended_original_resp = d.post_process(
                                        defended_original_resp or ""
                                    )
                                    defended_attacked_resp = d.post_process(
                                        defended_attacked_resp or ""
                                    )
                            else:
                                defended_prompt = prompt
                                for d in self.defenses:
                                    defended_prompt = d.pre_process(defended_prompt)

                                if config.attack_method.lower() == "gcg":
                                    defended_insert_pos = defended_prompt.rfind(end_tag)
                                    if defended_insert_pos != -1:
                                        defended_attacked_content = (
                                            defended_prompt[:defended_insert_pos]
                                            + attack_suffix + end_tag
                                        )
                                    else:
                                        defended_attacked_content = defended_prompt + attack_suffix
                                elif config.attack_method.lower() == "roleplay":
                                    end_tag_rp = "</student_answer>"
                                    rp_insert = defended_prompt.rfind(end_tag_rp)
                                    if rp_insert != -1:
                                        defended_attacked_content = (
                                            defended_prompt[:rp_insert]
                                            + attack_suffix + end_tag_rp
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
                                        defended_attacked_resp = self._generate(
                                            [{"role": "user", "content": defended_attacked_content}]
                                        )
                                    finally:
                                        self._remove_defense_hooks(hook_removers)

                                for d in self.defenses:
                                    defended_original_resp = d.post_process(
                                        defended_original_resp or ""
                                    )
                                    defended_attacked_resp = d.post_process(
                                        defended_attacked_resp or ""
                                    )

                        except DefenseRejectException:
                            rejected = True

                    if self._should_log_attention() and self.defenses and not rejected:
                        defended_prompt_for_attn = prompt
                        for d in self.defenses:
                            defended_prompt_for_attn = d.pre_process(defended_prompt_for_attn)
                        if config.attack_method.lower() == "gcg":
                            end_tag_rp = "</student_answer>"
                            defended_insert_pos = defended_prompt_for_attn.rfind(end_tag_rp)
                            if defended_insert_pos != -1:
                                defended_attacked_for_attn = (
                                    defended_prompt_for_attn[:defended_insert_pos]
                                    + attack_suffix + end_tag_rp
                                )
                            else:
                                defended_attacked_for_attn = defended_prompt_for_attn + attack_suffix
                        elif config.attack_method.lower() == "roleplay":
                            end_tag_rp = "</student_answer>"
                            rp_insert = defended_prompt_for_attn.rfind(end_tag_rp)
                            if rp_insert != -1:
                                defended_attacked_for_attn = (
                                    defended_prompt_for_attn[:rp_insert]
                                    + attack_suffix + end_tag_rp
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

                    # ── Step 4: 记录结果 ──
                    result = AttackResult(
                        student_qa_data=data,
                        original_response=original_resp,
                        attacked_response=attacked_resp,
                        meta={
                            "defended_original_response": defended_original_resp,
                            "defended_attacked_response": defended_attacked_resp,
                            "rejected": rejected,
                            "attention": attn_meta or None,
                        }
                    )
                    logger.result(result.as_dict())
                    all_results.append(result.as_dict())

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
                    metrics: EvalMetrics = compute_metrics(flat_results, nclass=config.nclass)
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
                else:
                    logger.info(
                        f"[{config.model_config.name}] [{data_config.name}] "
                        f"QWK_clean={metrics.qwk_clean:.4f} "
                        f"QWK_attack={metrics.qwk_attack:.4f} "
                        f"ASR={metrics.asr:.4f} "
                        f"CAS={metrics.cas:.4f}"
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
