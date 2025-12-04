"""
Gemini 2.5 Flash 서비스 - 감정 분석 지원
"""
import asyncio
from typing import AsyncGenerator, Dict, Any
from google import genai
from app.core.config import GEMINI_API_KEY, GEMINI_MODEL


async def analyze_sentiment_with_gemini(text: str) -> Dict[str, Any]:
    """
    Gemini를 사용한 감정 분석
    
    Args:
        text: 분석할 텍스트
    
    Returns:
        감정 분석 결과 (label, confidence, description)
    """
    if not GEMINI_API_KEY:
        return {
            "label": "unknown",
            "confidence": 0.0,
            "description": "GEMINI_API_KEY가 설정되지 않았습니다.",
            "error": True
        }
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        prompt = f"""다음 텍스트의 감정을 분석해주세요. 반드시 아래 JSON 형식으로만 응답해주세요:

텍스트: "{text}"

응답 형식:
{{"label": "positive" 또는 "negative" 또는 "neutral", "confidence": 0.0~1.0 사이의 숫자, "description": "간단한 설명"}}

JSON만 응답하세요:"""
        
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        
        response_text = response.text if hasattr(response, 'text') else str(response)
        
        # JSON 파싱 시도
        import json
        import re
        
        # JSON 추출
        json_match = re.search(r'\{[^}]+\}', response_text)
        if json_match:
            result = json.loads(json_match.group())
            # confidence를 숫자로 변환
            if isinstance(result.get('confidence'), str):
                result['confidence'] = float(result['confidence'])
            return result
        
        # 파싱 실패 시 기본 응답
        return {
            "label": "positive" if any(word in text.lower() for word in ['좋', '감사', '행복', 'good', 'happy', 'love', 'great']) else "negative",
            "confidence": 0.7,
            "description": response_text[:100]
        }
        
    except Exception as e:
        print(f"❌ Gemini Sentiment Analysis 오류: {e}")
        return {
            "label": "unknown",
            "confidence": 0.0,
            "description": str(e),
            "error": True
        }


async def generate_gemini_stream(
    message: str,
    chat_history: list = None,
    model: str = None
) -> AsyncGenerator[str, None]:
    """
    Gemini 2.5 Flash를 사용한 스트리밍 응답 생성
    
    Args:
        message: 사용자 메시지
        chat_history: 이전 대화 기록 (선택적)
        model: 사용할 모델명
    
    Yields:
        str: 스트리밍된 텍스트 청크
    """
    if not GEMINI_API_KEY:
        yield "Error: GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요."
        return
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        contents = message
        if chat_history:
            history_messages = []
            for hist in chat_history:
                content = hist.get("content", "")
                if content:
                    history_messages.append(content)
            if history_messages:
                contents = "\n".join(history_messages) + "\n" + message
        
        response = client.models.generate_content_stream(
            model=model or GEMINI_MODEL,
            contents=contents
        )
        
        for chunk in response:
            if hasattr(chunk, 'text') and chunk.text:
                yield chunk.text
                await asyncio.sleep(0.01)
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"❌ Gemini API 오류: {e}")
        yield error_msg


async def generate_gemini_simple(
    message: str,
    chat_history: list = None,
    model: str = None
) -> str:
    """
    Gemini 2.5 Flash를 사용한 단순 응답 생성 (비스트리밍)
    
    Args:
        message: 사용자 메시지
        chat_history: 이전 대화 기록 (선택적)
        model: 사용할 모델명
    
    Returns:
        str: 완전한 응답 텍스트
    """
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요."
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        contents = message
        if chat_history:
            history_messages = []
            for hist in chat_history:
                content = hist.get("content", "")
                if content:
                    history_messages.append(content)
            if history_messages:
                contents = "\n".join(history_messages) + "\n" + message
        
        response = client.models.generate_content(
            model=model or GEMINI_MODEL,
            contents=contents
        )
        
        return response.text if hasattr(response, 'text') else str(response)
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"❌ Gemini API 오류: {e}")
        return error_msg
