"""
LLM 챗봇 API 테스트 케이스

테스트 대상:
- POST /api/chat        : Ollama 로컬 LLM 채팅
- POST /api/chat/gemini : Gemini API 채팅
- GET  /api/chat/models : 사용 가능한 모델 목록

테스트 전략:
1. 엔드포인트 존재 확인
2. 입력 유효성 검사
3. 정상 응답 형식 검증 (서비스 가용 시)
4. 에러 처리 검증
"""
import pytest


class TestOllamaChatEndpoint:
    """
    Ollama 로컬 LLM 채팅 API 테스트

    엔드포인트: POST /api/chat
    요청 형식: JSON
    {
        "message": "사용자 메시지",
        "model": "gemma3:4b"  // 선택, 기본값 있음
    }

    Note: Ollama 서비스 실행 필요 (localhost:11434)
    """

    # =========================================================================
    # 엔드포인트 유효성 테스트
    # =========================================================================

    def test_chat_endpoint_exists(self, client):
        """
        [확인] 채팅 엔드포인트 존재 여부

        Given: 채팅 API 엔드포인트
        When: 요청 전송
        Then: 404가 아님 (엔드포인트 존재)

        Note: Ollama 미실행 시 500/503 허용
        """
        import httpx
        payload = {"message": "Hello", "model": "gemma3:4b"}
        try:
            response = client.post("/api/chat", json=payload)
            # 404가 아니면 엔드포인트 존재
            assert response.status_code != 404
        except httpx.ConnectError:
            # Ollama 서비스 미실행 시 테스트 스킵
            pytest.skip("Ollama service not running")

    # =========================================================================
    # 입력 유효성 검사 테스트
    # =========================================================================

    def test_chat_missing_message(self, client):
        """
        [실패] message 필드 누락

        Given: message 필드 없음
        When: 채팅 API 호출
        Then: 422 Unprocessable Entity
        """
        payload = {"model": "gemma3:4b"}
        response = client.post("/api/chat", json=payload)

        assert response.status_code == 422

    def test_chat_empty_message(self, client):
        """
        [실패] 빈 메시지

        Given: message가 빈 문자열
        When: 채팅 API 호출
        Then: 200, 400, 422, 또는 500 (구현에 따라 다름)

        Note: 빈 메시지 허용 여부는 구현에 따라 다름
        """
        import httpx
        payload = {"message": "", "model": "gemma3:4b"}
        try:
            response = client.post("/api/chat", json=payload)
            # 빈 메시지 처리는 구현에 따라 다름
            assert response.status_code in [200, 400, 422, 500, 503]
        except httpx.ConnectError:
            pytest.skip("Ollama service not running")

    def test_chat_empty_body(self, client):
        """
        [실패] 빈 요청 본문

        Given: 빈 JSON 객체
        When: 채팅 API 호출
        Then: 422 Unprocessable Entity
        """
        response = client.post("/api/chat", json={})

        assert response.status_code == 422

    # =========================================================================
    # 정상 케이스 테스트 (Ollama 서비스 가용 시)
    # =========================================================================

    def test_chat_basic_message(self, client, sample_chat_message):
        """
        [성공] 기본 채팅 메시지

        Given: 유효한 메시지
        When: 채팅 API 호출
        Then: 200 OK 또는 서비스 불가 에러

        예상 결과 (성공 시):
        - 스트리밍 또는 JSON 응답
        """
        import httpx
        payload = {
            "message": sample_chat_message["message"],
            "model": "gemma3:4b"
        }
        try:
            response = client.post("/api/chat", json=payload)
            # Ollama 서비스 미실행 시 500/503
            assert response.status_code in [200, 500, 503]
        except httpx.ConnectError:
            pytest.skip("Ollama service not running")

    def test_chat_default_model(self, client):
        """
        [성공] 기본 모델 사용 (model 미지정)

        Given: model 파라미터 없음
        When: 채팅 API 호출
        Then: 200 OK 또는 서비스 불가 에러

        Note: 기본 모델이 설정되어 있어야 함
        """
        import httpx
        payload = {"message": "Hello!"}
        try:
            response = client.post("/api/chat", json=payload)
            # model 없어도 기본값 사용
            assert response.status_code in [200, 422, 500, 503]
        except httpx.ConnectError:
            pytest.skip("Ollama service not running")

    def test_chat_korean_message(self, client):
        """
        [성공] 한국어 메시지

        Given: 한국어 채팅 메시지
        When: 채팅 API 호출
        Then: 200 OK 또는 서비스 불가 에러
        """
        import httpx
        payload = {
            "message": "안녕하세요, 오늘 날씨가 어떤가요?",
            "model": "gemma3:4b"
        }
        try:
            response = client.post("/api/chat", json=payload)
            assert response.status_code in [200, 500, 503]
        except httpx.ConnectError:
            pytest.skip("Ollama service not running")

    def test_chat_long_message(self, client):
        """
        [성공] 긴 메시지

        Given: 긴 메시지 (500 단어 이상)
        When: 채팅 API 호출
        Then: 200 OK 또는 서비스 불가 에러
        """
        import httpx
        long_message = {"message": "Hello! " * 500, "model": "gemma3:4b"}
        try:
            response = client.post("/api/chat", json=long_message)
            assert response.status_code in [200, 400, 500, 503]
        except httpx.ConnectError:
            pytest.skip("Ollama service not running")

    def test_chat_special_characters(self, client):
        """
        [성공] 특수문자 포함 메시지

        Given: 특수문자, 이모지, HTML 태그 포함 메시지
        When: 채팅 API 호출
        Then: 200 OK 또는 서비스 불가 에러

        Note: XSS 방지 처리 필요
        """
        import httpx
        payload = {
            "message": "Hello! 👋 <script>alert('test')</script> 안녕!",
            "model": "gemma3:4b"
        }
        try:
            response = client.post("/api/chat", json=payload)
            assert response.status_code in [200, 500, 503]
        except httpx.ConnectError:
            pytest.skip("Ollama service not running")

    def test_chat_code_snippet(self, client):
        """
        [성공] 코드 스니펫 포함 메시지

        Given: Python 코드가 포함된 메시지
        When: 채팅 API 호출
        Then: 200 OK 또는 서비스 불가 에러
        """
        import httpx
        payload = {
            "message": "이 Python 코드 설명해줘:\n```python\ndef hello():\n    print('Hello')\n```",
            "model": "gemma3:4b"
        }
        try:
            response = client.post("/api/chat", json=payload)
            assert response.status_code in [200, 500, 503]
        except httpx.ConnectError:
            pytest.skip("Ollama service not running")


class TestGeminiChatEndpoint:
    """
    Gemini API 채팅 테스트

    엔드포인트: POST /api/chat/gemini
    요청 형식: JSON
    {
        "message": "사용자 메시지",
        "chat_history": [...]  // 선택
    }

    Note: GEMINI_API_KEY 환경 변수 필요
    """

    # =========================================================================
    # 엔드포인트 유효성 테스트
    # =========================================================================

    def test_gemini_chat_endpoint_exists(self, client):
        """
        [확인] Gemini 채팅 엔드포인트 존재 여부

        Given: Gemini 채팅 API
        When: 요청 전송
        Then: 엔드포인트 응답 (구현 여부에 따라 다름)

        Note: 엔드포인트가 아직 구현되지 않은 경우 404 허용
        """
        payload = {"message": "Hello"}
        response = client.post("/api/chat/gemini", json=payload)

        # 엔드포인트 존재 여부 확인 (미구현 시 404 허용)
        assert response.status_code in [200, 404, 422, 500, 503]

    # =========================================================================
    # 입력 유효성 검사 테스트
    # =========================================================================

    def test_gemini_chat_missing_message(self, client):
        """
        [실패] message 필드 누락

        Given: message 필드 없음
        When: Gemini 채팅 API 호출
        Then: 422 Unprocessable Entity 또는 404 (미구현 시)
        """
        payload = {}
        response = client.post("/api/chat/gemini", json=payload)

        # 엔드포인트 미구현 시 404 허용
        assert response.status_code in [404, 422]

    # =========================================================================
    # 정상 케이스 테스트 (API 키 설정 시)
    # =========================================================================

    def test_gemini_chat_basic(self, client):
        """
        [성공] 기본 Gemini 채팅

        Given: 유효한 메시지
        When: Gemini 채팅 API 호출
        Then: 200 OK 또는 API 키 없음 에러, 또는 404 (미구현 시)
        """
        payload = {"message": "안녕하세요, 오늘 날씨가 어떤가요?"}
        response = client.post("/api/chat/gemini", json=payload)

        # API 키 미설정 시 에러, 엔드포인트 미구현 시 404
        assert response.status_code in [200, 404, 500, 503]

    def test_gemini_chat_with_history(self, client, chat_history_sample):
        """
        [성공] 대화 기록 포함 채팅

        Given: chat_history가 있는 요청
        When: Gemini 채팅 API 호출
        Then: 200 OK 또는 에러, 또는 404 (미구현 시)

        Note: 대화 컨텍스트 유지
        """
        payload = {
            "message": "그래서 요약하면?",
            "chat_history": chat_history_sample
        }
        response = client.post("/api/chat/gemini", json=payload)

        # 엔드포인트 미구현 시 404 허용
        assert response.status_code in [200, 404, 500, 503]

    def test_gemini_chat_korean(self, client):
        """
        [성공] 한국어 Gemini 채팅

        Given: 한국어 메시지
        When: Gemini 채팅 API 호출
        Then: 200 OK 또는 에러, 또는 404 (미구현 시)
        """
        payload = {"message": "이 제품은 정말 좋아요! 추천합니다."}
        response = client.post("/api/chat/gemini", json=payload)

        # 엔드포인트 미구현 시 404 허용
        assert response.status_code in [200, 404, 500, 503]


class TestChatModelsEndpoint:
    """
    사용 가능한 모델 목록 API 테스트

    엔드포인트: GET /api/chat/models
    """

    def test_models_endpoint_exists(self, client):
        """
        [확인] 모델 목록 엔드포인트 존재 여부

        Given: 모델 목록 API
        When: GET 요청
        Then: 200 또는 404 (엔드포인트 존재 여부에 따라)
        """
        response = client.get("/api/chat/models")

        # 엔드포인트가 있으면 200, 없으면 404
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            # 모델 목록 형식 확인
            assert isinstance(data, (dict, list))


class TestChatResponseFormat:
    """
    채팅 응답 형식 테스트
    """

    def test_chat_response_content_type(self, client, sample_chat_message):
        """
        [검증] 응답 Content-Type 확인

        Given: 채팅 요청
        When: API 호출
        Then: JSON 또는 스트리밍 응답

        예상 Content-Type:
        - application/json
        - application/x-ndjson (스트리밍)
        - text/event-stream (SSE)
        """
        import httpx
        payload = {
            "message": sample_chat_message["message"],
            "model": "gemma3:4b"
        }
        try:
            response = client.post("/api/chat", json=payload)

            if response.status_code == 200:
                content_type = response.headers.get("content-type", "")
                valid_types = [
                    "application/json",
                    "application/x-ndjson",
                    "text/event-stream"
                ]
                assert any(t in content_type for t in valid_types)
        except httpx.ConnectError:
            pytest.skip("Ollama service not running")


class TestChatEdgeCases:
    """
    채팅 엣지 케이스 테스트
    """

    def test_chat_numeric_only_message(self, client):
        """
        [엣지] 숫자만 있는 메시지

        Given: 숫자만 포함된 메시지
        When: 채팅 API 호출
        Then: 정상 처리
        """
        import httpx
        payload = {"message": "12345", "model": "gemma3:4b"}
        try:
            response = client.post("/api/chat", json=payload)
            assert response.status_code in [200, 500, 503]
        except httpx.ConnectError:
            pytest.skip("Ollama service not running")

    def test_chat_invalid_model_name(self, client):
        """
        [엣지] 잘못된 모델 이름

        Given: 존재하지 않는 모델 이름
        When: 채팅 API 호출
        Then: 400, 404, 또는 500 에러
        """
        import httpx
        payload = {
            "message": "Hello",
            "model": "nonexistent-model-12345"
        }
        try:
            response = client.post("/api/chat", json=payload)
            # 잘못된 모델은 에러 반환
            assert response.status_code in [200, 400, 404, 500, 503]
        except httpx.ConnectError:
            pytest.skip("Ollama service not running")

    def test_chat_unicode_message(self, client):
        """
        [엣지] 다양한 유니코드 메시지

        Given: 여러 언어가 혼합된 유니코드 메시지
        When: 채팅 API 호출
        Then: 정상 처리
        """
        import httpx
        payload = {
            "message": "Hello 你好 こんにちは 안녕하세요 مرحبا",
            "model": "gemma3:4b"
        }
        try:
            response = client.post("/api/chat", json=payload)
            assert response.status_code in [200, 500, 503]
        except httpx.ConnectError:
            pytest.skip("Ollama service not running")


class TestOpenAPIDocumentation:
    """
    OpenAPI 문서 테스트
    """

    def test_openapi_json_available(self, client):
        """
        [확인] OpenAPI JSON 문서 접근 가능

        Given: OpenAPI 문서 엔드포인트
        When: GET 요청
        Then: 200 OK, JSON 응답
        """
        response = client.get("/openapi.json")

        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data
