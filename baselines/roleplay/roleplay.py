from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from utils.config_utils import AttackConfig
from utils.log_utils import GradingAttackLogger
from utils.data_utils import read_student_qa_data_from_jsonl, AttackResult


class RolePlay:
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

            attacked_message_list = [
                tokenizer.apply_chat_template([{
                        "role": "user",
                        "content": config.grading_template.format(
                            question=data.question,
                            solution=data.question_answer,
                            student_answer=data.student_answer
                        ) + config.params["adv_prompt"]
                    }],
                    tokenize=False,
                    add_generation_prompt=True
                ) for data in data_list
            ]

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
