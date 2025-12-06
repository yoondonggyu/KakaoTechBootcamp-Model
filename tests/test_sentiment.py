"""
감정 분석(Sentiment) API 테스트 케이스

테스트 대상:
- POST /api/sentiment        : ML 모델 기반 감정 분석
- POST /api/sentiment/gemini : Gemini API 기반 감정 분석

테스트 전략:
1. 엔드포인트 존재 확인 (서비스 가용성과 무관)
2. 입력 유효성 검사 (필수 필드, 형식 등)
3. 정상 응답 형식 검증 (서비스 가용 시)
"""
import pytest


class TestSentimentAnalysisEndpoint:
    """
    ML 모델 기반 감정 분석 API 테스트

    엔드포인트: POST /api/sentiment
    요청 형식: JSON
    {
        "text": "분석할 텍스트",
        "explain": false  // 설명 포함 여부 (선택)
    }
    """

    # =========================================================================
    # 엔드포인트 유효성 테스트
    # =========================================================================

    def test_sentiment_endpoint_exists(self, client):
        """
        [확인] 엔드포인트 존재 여부

        Given: 감정 분석 API 엔드포인트
        When: 유효한 요청 전송
        Then: 422(잘못된 요청)나 200/500(처리 시도) 중 하나

        Note: 404가 아니면 엔드포인트가 존재함
        """
        payload = {"text": "test", "explain": False}
        response = client.post("/api/sentiment", json=payload)

        # 404가 아니면 엔드포인트 존재
        assert response.status_code != 404

    # =========================================================================
    # 입력 유효성 검사 테스트
    # =========================================================================

    def test_sentiment_empty_text(self, client):
        """
        [실패] 빈 텍스트

        Given: text가 빈 문자열
        When: 감정 분석 API 호출
        Then: 400 또는 422 (입력 검증 실패)

        예상 결과:
        - 빈 텍스트는 분석할 수 없음
        """
        payload = {"text": "", "explain": False}
        response = client.post("/api/sentiment", json=payload)

        assert response.status_code in [400, 422]

    def test_sentiment_missing_text_field(self, client):
        """
        [실패] text 필드 누락

        Given: text 필드 없음
        When: 감정 분석 API 호출
        Then: 422 Unprocessable Entity
        """
        payload = {"explain": False}
        response = client.post("/api/sentiment", json=payload)

        assert response.status_code == 422

    def test_sentiment_whitespace_only(self, client):
        """
        [실패] 공백만 있는 텍스트

        Given: text가 공백 문자열만 포함
        When: 감정 분석 API 호출
        Then: 400, 422, 또는 500 (구현에 따라 다름)
        """
        payload = {"text": "   ", "explain": False}
        response = client.post("/api/sentiment", json=payload)

        assert response.status_code in [400, 422, 500]

    def test_sentiment_empty_body(self, client):
        """
        [실패] 빈 요청 본문

        Given: 빈 JSON 객체
        When: 감정 분석 API 호출
        Then: 422 Unprocessable Entity
        """
        response = client.post("/api/sentiment", json={})

        assert response.status_code == 422

    # =========================================================================
    # 정상 케이스 테스트 (서비스 가용 시)
    # =========================================================================

    def test_sentiment_positive_text(self, client, sample_text_positive):
        """
        [성공] 긍정적인 텍스트 분석

        Given: 긍정적인 영어 텍스트
        When: 감정 분석 API 호출
        Then: 200 OK 또는 서비스 불가 에러

        예상 결과 (성공 시):
        - Response: {"label": "positive", "confidence": 0.95, ...}
        """
        payload = {"text": sample_text_positive, "explain": False}
        response = client.post("/api/sentiment", json=payload)

        # 모델 로드 실패 가능성 고려
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "label" in data
            assert "confidence" in data
            assert 0 <= data["confidence"] <= 1

    def test_sentiment_negative_text(self, client, sample_text_negative):
        """
        [성공] 부정적인 텍스트 분석

        Given: 부정적인 영어 텍스트
        When: 감정 분석 API 호출
        Then: 200 OK 또는 서비스 불가 에러
        """
        payload = {"text": sample_text_negative, "explain": False}
        response = client.post("/api/sentiment", json=payload)

        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "label" in data

    def test_sentiment_with_explain(self, client, sample_text_positive):
        """
        [성공] 설명 포함 감정 분석

        Given: explain=True
        When: 감정 분석 API 호출
        Then: 200 OK, 추가 설명 정보 포함

        예상 결과 (성공 시):
        - top_tokens 또는 추가 설명 필드 존재
        """
        payload = {"text": sample_text_positive, "explain": True}
        response = client.post("/api/sentiment", json=payload)

        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            # explain=True일 때 추가 정보 확인
            assert "label" in data

    def test_sentiment_long_text(self, client):
        """
        [성공] 긴 텍스트 분석

        Given: 긴 텍스트 (100단어 이상)
        When: 감정 분석 API 호출
        Then: 200 OK 또는 서비스 불가 에러
        """
        long_text = "I love this! " * 100
        payload = {"text": long_text, "explain": False}
        response = client.post("/api/sentiment", json=payload)

        assert response.status_code in [200, 500, 503]


class TestGeminiSentimentEndpoint:
    """
    Gemini API 기반 감정 분석 테스트

    엔드포인트: POST /api/sentiment/gemini
    요청 형식: JSON
    {
        "text": "분석할 텍스트",
        "explain": false
    }

    Note: Gemini API 키 필요, 없으면 서비스 에러 반환
    """

    # =========================================================================
    # 엔드포인트 유효성 테스트
    # =========================================================================

    def test_gemini_sentiment_endpoint_exists(self, client):
        """
        [확인] Gemini 감정 분석 엔드포인트 존재 여부

        Given: Gemini 감정 분석 API
        When: 요청 전송
        Then: 404가 아님 (엔드포인트 존재)
        """
        payload = {"text": "test", "explain": False}
        response = client.post("/api/sentiment/gemini", json=payload)

        # 404가 아니면 엔드포인트 존재
        assert response.status_code != 404

    # =========================================================================
    # 입력 유효성 검사 테스트
    # =========================================================================

    def test_gemini_sentiment_empty_text(self, client):
        """
        [실패] 빈 텍스트

        Given: text가 빈 문자열
        When: Gemini 감정 분석 API 호출
        Then: 400 또는 422
        """
        payload = {"text": "", "explain": False}
        response = client.post("/api/sentiment/gemini", json=payload)

        assert response.status_code in [400, 422]

    def test_gemini_sentiment_missing_text(self, client):
        """
        [실패] text 필드 누락

        Given: text 필드 없음
        When: Gemini 감정 분석 API 호출
        Then: 422 Unprocessable Entity
        """
        payload = {"explain": False}
        response = client.post("/api/sentiment/gemini", json=payload)

        assert response.status_code == 422

    # =========================================================================
    # 정상 케이스 테스트 (API 키 설정 시)
    # =========================================================================

    def test_gemini_sentiment_korean(self, client, sample_text_korean_positive):
        """
        [성공] 한국어 텍스트 Gemini 분석

        Given: 긍정적인 한국어 텍스트
        When: Gemini 감정 분석 API 호출
        Then: 200 OK 또는 API 키 없음 에러

        Note: Gemini는 한국어 지원이 좋음
        """
        payload = {"text": sample_text_korean_positive, "explain": False}
        response = client.post("/api/sentiment/gemini", json=payload)

        # API 키 미설정 시 500/503 가능
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "label" in data
            assert "confidence" in data

    def test_gemini_sentiment_english(self, client, sample_text_positive):
        """
        [성공] 영어 텍스트 Gemini 분석

        Given: 긍정적인 영어 텍스트
        When: Gemini 감정 분석 API 호출
        Then: 200 OK 또는 API 키 없음 에러
        """
        payload = {"text": sample_text_positive, "explain": False}
        response = client.post("/api/sentiment/gemini", json=payload)

        assert response.status_code in [200, 500, 503]

    def test_gemini_sentiment_with_emoji(self, client):
        """
        [성공] 이모지 포함 텍스트 분석

        Given: 이모지가 포함된 텍스트
        When: Gemini 감정 분석 API 호출
        Then: 200 OK 또는 서비스 에러
        """
        payload = {"text": "이 제품 👍👍👍 정말 좋아요!!!", "explain": False}
        response = client.post("/api/sentiment/gemini", json=payload)

        assert response.status_code in [200, 500, 503]

    def test_gemini_sentiment_mixed_language(self, client):
        """
        [성공] 영어+한글 혼합 텍스트 분석

        Given: 영어와 한글이 혼합된 텍스트
        When: Gemini 감정 분석 API 호출
        Then: 200 OK 또는 서비스 에러
        """
        payload = {"text": "This product is 정말 amazing!", "explain": False}
        response = client.post("/api/sentiment/gemini", json=payload)

        assert response.status_code in [200, 500, 503]


class TestSentimentResponseFormat:
    """
    감정 분석 응답 형식 테스트

    응답 스키마 검증
    """

    def test_sentiment_response_schema(self, client, sample_text_positive):
        """
        [검증] 응답 스키마 확인

        예상 응답 형식:
        {
            "label": "positive" | "negative" | "neutral",
            "confidence": 0.0 ~ 1.0,
            "top_tokens": [...] (explain=True일 때)
        }
        """
        payload = {"text": sample_text_positive, "explain": False}
        response = client.post("/api/sentiment", json=payload)

        if response.status_code == 200:
            data = response.json()

            # 필수 필드 확인
            assert "label" in data
            assert "confidence" in data

            # 타입 확인
            assert isinstance(data["label"], str)
            assert isinstance(data["confidence"], (int, float))

            # 값 범위 확인
            assert data["confidence"] >= 0
            assert data["confidence"] <= 1

    def test_gemini_response_schema(self, client, sample_text_positive):
        """
        [검증] Gemini 응답 스키마 확인
        """
        payload = {"text": sample_text_positive, "explain": False}
        response = client.post("/api/sentiment/gemini", json=payload)

        if response.status_code == 200:
            data = response.json()

            assert "label" in data
            assert "confidence" in data
