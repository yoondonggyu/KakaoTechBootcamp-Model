from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List

router = APIRouter()

class TaggingRequest(BaseModel):
    text: str = Field(..., description="Text to extract tags from")

class TaggingResponse(BaseModel):
    tags: List[str]

@router.post("/auto-tag", response_model=TaggingResponse)
async def auto_tag_text(payload: TaggingRequest):
    """
    Rule-based auto-tagging for MVP.
    """
    text = payload.text.lower()
    tags = []
    
    keywords = {
        "웨딩홀": ["웨딩홀", "예식장", "홀", "버진로드"],
        "드레스": ["드레스", "메이크업", "스드메", "피팅"],
        "스튜디오": ["스튜디오", "촬영", "사진", "앨범"],
        "견적": ["견적", "비용", "가격", "예산", "할인"],
        "식사": ["뷔페", "식사", "밥", "맛"],
        "주차": ["주차", "교통", "위치"]
    }

    for tag, words in keywords.items():
        if any(word in text for word in words):
            tags.append(tag)
            
    # Default tag if none found
    if not tags:
        tags.append("일반")

    return {"tags": tags}
