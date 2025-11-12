from pydantic import BaseModel
from typing import Optional, Union, List, TypedDict, Any

class ChatResponseGeneric(BaseModel):
    response: str

class TaskRoute(BaseModel):
    response: bool

class State(TypedDict):
    user_id: str
    user_question: str
    user_questions_validation: bool
    user_answer: Optional[str]