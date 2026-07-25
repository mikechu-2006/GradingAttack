"""Prompt Injection attack — instruction-override based adversarial prompt injection.

Unlike RolePlay (which uses persona-based manipulation: "pretend it is correct"),
Prompt Injection directly overrides the grading instructions by injecting
adversarial directives into the student_answer field.

The attacker controls only the student_answer content, so the injection is
inserted before </student_answer> inside the student_answer XML tag.

Corresponds to the prompt injection attack category in:
- "**Important** You should give me full credits!" (arxiv 2606.03090)
- EvalHack (MDPI Information, 2026)
- OWASP LLM01: Prompt Injection
"""

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from utils.config_utils import AttackConfig
from utils.log_utils import GradingAttackLogger
from utils.data_utils import read_student_qa_data_from_jsonl, AttackResult


class Injection:
    """Prompt Injection attack for LLM-based automatic short answer grading.

    Injects an adversarial instruction-override prompt into the student_answer
    field, inserted before the closing </student_answer> tag so that it appears
    as part of the student-submitted content.

    The injection prompt is read from config.params["injection_prompt"].
    Different strategies (delimiter confusion, authority override, instruction
    mimicry) are achieved by changing this YAML parameter — no code changes needed.
    """

    def __init__(self, config: AttackConfig):
        self.config = config

    def run(self):
        config = self.config

        model = LLM(
            config.model_config.path,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_config.path,
            trust_remote_code=True
        )
        sampling_params = SamplingParams(**config.generation_config.as_dict())
        logger = GradingAttackLogger(config)

        injection_prompt = config.params["injection_prompt"]
        end_tag = "</student_answer>"

        for data_config in config.data_config:
            data_list = read_student_qa_data_from_jsonl(data_config.path)[:10]

            original_message_list = [
                tokenizer.apply_chat_template([{
                        "role": "user",
                        "content": config.grading_template.format(
                            question=data.question,
                            solution=data.question_answer,
                            student_answer=data.student_answer
                        )
                    }],
                    tokenize=False,
                    add_generation_prompt=True
                ) for data in data_list
            ]

            attacked_message_list = []
            for data in data_list:
                template = config.grading_template.format(
                    question=data.question,
                    solution=data.question_answer,
                    student_answer=data.student_answer
                )
                insert_pos = template.rfind(end_tag)
                if insert_pos != -1:
                    attacked_content = template[:insert_pos] + injection_prompt + end_tag
                else:
                    attacked_content = template + injection_prompt

                attacked_message_list.append(
                    tokenizer.apply_chat_template([{
                        "role": "user",
                        "content": attacked_content
                    }],
                    tokenize=False,
                    add_generation_prompt=True
                ))

            original_response_list = model.generate(
                original_message_list,
                sampling_params
            )

            attacked_response_list = model.generate(
                attacked_message_list,
                sampling_params
            )

            for data, ori_res, atk_res in zip(data_list, original_response_list, attacked_response_list):
                ori_res_str = ori_res.outputs[0].text
                atk_res_str = atk_res.outputs[0].text

                result = AttackResult(
                    student_qa_data=data,
                    original_response=ori_res_str,
                    attacked_response=atk_res_str
                )

                logger.result(result.as_dict())

            logger.info(
                f"[MODEL] {config.model_config.name}  "
                + f"[DATASET] {data_config.name}  "
                + f"Finished"
            )
