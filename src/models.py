import uuid
from typing import List, Union
from pydantic import BaseModel, Field, model_validator


class MinimalSource(BaseModel):
    """A source location inside the indexed corpus"""
    file_path: str
    first_character_index: int
    last_character_index: int

    @model_validator(mode='after')
    def _check_range(self) -> "MinimalSource":
        """Ensure a valid range of at most 2000 characters"""
        if self.last_character_index < self.first_character_index:
            raise ValueError("last < first")
        if self.last_character_index - self.first_character_index > 2000:
            raise ValueError("source longer than 2000 chars")
        return self


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int
