import re
import json

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class StudentQAData:
    question_id: Optional[str] = None
    student_id: Optional[str] = None
    question: Optional[str] = None
    question_answer: Optional[str] = None
    student_answer: Optional[str] = None
    verification: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
    

@dataclass
class AttackResult:
    student_qa_data: Optional[StudentQAData] = None
    original_response: Optional[str] = None
    attacked_response: Optional[str] = None
    meta: Optional[Any] = None
    
    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def read_jsonl(path: str) -> List[Dict[Any, Any]]:
    with open(path, "r", encoding="utf-8") as jsonl_file:
        return [json.loads(line.strip()) for line in jsonl_file if line.strip()]


def read_student_qa_data_from_jsonl(path: str) -> List[StudentQAData]:
    return [StudentQAData(**data) for data in read_jsonl(path)]


def extract_grade(response: str) -> Optional[str]:
    matches = re.findall(r"<answer>(.*?)</answer>", response)
    return matches[-1] if matches else None
