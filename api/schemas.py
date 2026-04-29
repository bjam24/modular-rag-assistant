from pydantic import BaseModel


class AskRequest(BaseModel):
    query: str
    history: str = ""
    retrieval_mode: str = "hybrid"
    generation_mode: str = "balanced"


class TokenUsage(BaseModel):
    input: int
    output: int


class Source(BaseModel):
    source: str | None = None
    score: float | None = None
    text: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency: float
    tokens: TokenUsage
    cost_usd: float