from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List
import random

router = APIRouter()

class EmbeddingRequest(BaseModel):
    text: str = Field(..., description="Text to embed")

class EmbeddingResponse(BaseModel):
    vector: List[float]
    dim: int

@router.post("/embedding", response_model=EmbeddingResponse)
async def generate_embedding(payload: EmbeddingRequest):
    """
    Mock embedding generation. Returns a random vector of dimension 128.
    """
    # Mock 128-dimensional vector
    vector = [random.random() for _ in range(128)]
    
    return {
        "vector": vector,
        "dim": 128
    }
