import nanogcg
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.config_utils import AttackConfig
from utils.log_utils import GradingAttackLogger
from utils.data_utils import read_student_qa_data_from_jsonl, AttackResult


class GCG:
    def __init__(self, config: AttackConfig):
        self.config = config

    def run(self):
        config = self.config

        device = config.params["device"]
        target = config.params["target"]
        gcg_config = nanogcg.GCGConfig(**config.params["gcg_config"])
        model = AutoModelForCausalLM.from_pretrained(
            config.model_config.path,
            trust_remote_code=True,
        ).to(device)

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_config.path,
            trust_remote_code=True,
        )

        logger = GradingAttackLogger(config)

        for data_config in config.data_config:
            data_list = read_student_qa_data_from_jsonl(data_config.path)
            for data in data_list:
                messages = [{
                    "role": "user",
                    "content": config.grading_template.format(
                        question=data.question,
                        solution=data.question_answer,
                        student_answer=data.student_answer
                    )
                }]
                gcg_result = nanogcg.run(model, tokenizer, messages, target, gcg_config)
                best_string = gcg_result.best_string
                best_loss = gcg_result.best_loss

                original_input = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                ).to(device)
                original_output = model.generate(
                    original_input,
                    do_sample=False,
                    max_new_tokens=config.generation_config.max_tokens,
                    temperature=config.generation_config.temperature,
                )
                original_output_str = tokenizer.batch_decode(
                    original_output[:, original_input.shape[1] :], skip_special_tokens=True
                )[0]

                messages[-1]["content"] = messages[-1]["content"] + best_string
                attacked_input = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                ).to(device)
                attacked_output = model.generate(
                    attacked_input,
                    do_sample=False,
                    max_new_tokens=config.generation_config.max_tokens,
                    temperature=config.generation_config.temperature,
                )
                attacked_output_str = tokenizer.batch_decode(
                    attacked_output[:, attacked_input.shape[1] :], skip_special_tokens=True
                )[0]
                
                result = AttackResult(
                    student_qa_data=data,
                    original_response=original_output_str,
                    attacked_response=attacked_output_str,
                    meta=dict(
                        best_string=best_string,
                        best_loss=best_loss
                    )
                )
                logger.result(result.as_dict())

            logger.info(
                f"[MODEL] {config.model_config.name}  "
                + f"[DATASET] {data_config.name}  "
                + f"Finished"
            )
