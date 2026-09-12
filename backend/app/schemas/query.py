from typing import Optional

from pydantic import BaseModel


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    image_classification: Optional[dict] = None