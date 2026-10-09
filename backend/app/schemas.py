from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    history: list[Message] = Field(default_factory=list, max_length=20)


class Source(BaseModel):
    job_id: int | None
    portal: str | None
    company: str | None
    location: str | None
    country: str | None
    date: str | None
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]