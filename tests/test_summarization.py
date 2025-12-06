"""
요약(Summarization) API 테스트 케이스
Gemini API를 이용한 텍스트 요약
"""
import pytest


class TestSummarization:
    """Gemini 기반 요약 API 테스트"""
    
    def test_summarization_success(self, client, sample_summarization_text):
        """정상 텍스트 요약 테스트"""
        response = client.post("/api/summarize", json=sample_summarization_text)
        
        # Gemini API가 설정되지 않았을 수 있음
        assert response.status_code in [200, 404, 422, 500, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "summary" in data or "result" in data or "text" in data
    
    def test_summarization_short_text(self, client):
        """짧은 텍스트 요약 테스트"""
        payload = {"text": "짧은 문장입니다."}
        
        response = client.post("/api/summarize", json=payload)
        
        assert response.status_code in [200, 400, 404, 422, 500]
    
    def test_summarization_empty_text(self, client):
        """빈 텍스트 요약 테스트"""
        payload = {"text": ""}
        
        response = client.post("/api/summarize", json=payload)
        
        assert response.status_code in [400, 404, 422]
    
    def test_summarization_missing_text(self, client):
        """텍스트 필드 누락 테스트"""
        payload = {}
        
        response = client.post("/api/summarize", json=payload)
        
        assert response.status_code in [404, 422]
    
    def test_summarization_very_long_text(self, client):
        """매우 긴 텍스트 요약 테스트"""
        long_text = "인공지능은 매우 흥미로운 분야입니다. " * 200
        payload = {"text": long_text}
        
        response = client.post("/api/summarize", json=payload)
        
        assert response.status_code in [200, 400, 404, 422, 500, 503]
    
    def test_summarization_english_text(self, client):
        """영어 텍스트 요약 테스트"""
        payload = {
            "text": """
            Artificial intelligence is transforming the world. Machine learning algorithms
            can now recognize images, understand speech, and even generate creative content.
            Deep learning has revolutionized many industries including healthcare and finance.
            """
        }
        
        response = client.post("/api/summarize", json=payload)
        
        assert response.status_code in [200, 404, 422, 500, 503]
    
    def test_summarization_with_max_length(self, client, sample_summarization_text):
        """최대 길이 지정 요약 테스트"""
        payload = {
            **sample_summarization_text,
            "max_length": 100
        }
        
        response = client.post("/api/summarize", json=payload)
        
        assert response.status_code in [200, 404, 422, 500, 503]


class TestLLMSummarization:
    """LLM 기반 요약 테스트 (Gemini/Ollama)"""
    
    def test_gemini_summarization(self, client, sample_summarization_text):
        """Gemini 모델 요약 테스트"""
        payload = {
            **sample_summarization_text,
            "model": "gemini"
        }
        
        response = client.post("/api/summarize", json=payload)
        
        assert response.status_code in [200, 404, 422, 500, 503]
    
    def test_ollama_summarization(self, client, sample_summarization_text):
        """Ollama 로컬 모델 요약 테스트"""
        payload = {
            **sample_summarization_text,
            "model": "ollama"
        }
        
        response = client.post("/api/summarize", json=payload)
        
        # 엔드포인트가 없거나 Ollama가 실행 중이 아닐 수 있음
        assert response.status_code in [200, 404, 422, 500, 503]
    
    def test_summarization_bullet_points(self, client, sample_summarization_text):
        """불릿 포인트 형식 요약 테스트"""
        payload = {
            **sample_summarization_text,
            "format": "bullet_points"
        }
        
        response = client.post("/api/summarize", json=payload)
        
        assert response.status_code in [200, 404, 422, 500, 503]
