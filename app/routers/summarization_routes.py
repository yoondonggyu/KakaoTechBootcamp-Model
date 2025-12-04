from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Text to summarize")

class SummarizeResponse(BaseModel):
    summary: str
    original_length: int
    summary_length: int

@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(payload: SummarizeRequest):
    """
    Simple rule-based summarization for MVP.
    In a real scenario, this would call an LLM (e.g., Gemini, GPT).
    """
    text = payload.text
    # Simple heuristic: Take the first 2 sentences or first 100 chars
    sentences = text.split('.')
    summary = '. '.join(sentences[:2])
    if len(sentences) > 2:
        summary += "."
    
    if len(summary) > 200:
        summary = summary[:197] + "..."

    return {
        "summary": summary,
        "original_length": len(text),
        "summary_length": len(summary)
    }
