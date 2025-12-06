"""
pytest configuration and fixtures for FASTAPI_Project_model

AI 모델 서빙 서비스 테스트 설정
- 외부 서비스 의존성 Mock 처리
- 테스트용 샘플 데이터 제공
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock

# 상대 임포트를 위한 sys.path 설정
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app


# ============================================================================
# 기본 Fixture
# ============================================================================

@pytest.fixture(scope="module")
def client():
    """
    테스트 클라이언트 Fixture

    Scope: module - 모듈 내 모든 테스트가 같은 클라이언트 공유
    (모델 로딩 시간 절약)
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="function")
def client_per_test():
    """
    테스트별 새 클라이언트

    Scope: function - 각 테스트마다 새로운 클라이언트
    (테스트 격리가 필요할 때 사용)
    """
    with TestClient(app) as test_client:
        yield test_client


# ============================================================================
# 감정 분석 테스트 데이터
# ============================================================================

@pytest.fixture
def sample_text_positive():
    """긍정적인 감정 분석용 샘플 텍스트 (영어)"""
    return "I love this product! It's absolutely amazing and wonderful."


@pytest.fixture
def sample_text_negative():
    """부정적인 감정 분석용 샘플 텍스트 (영어)"""
    return "This is terrible. I hate it and regret buying this."


@pytest.fixture
def sample_text_neutral():
    """중립적인 감정 분석용 샘플 텍스트"""
    return "The product arrived on time. It is as described."


@pytest.fixture
def sample_text_korean_positive():
    """긍정적인 한국어 감정 분석용 샘플"""
    return "이 제품은 정말 좋아요! 강력 추천합니다."


@pytest.fixture
def sample_text_korean_negative():
    """부정적인 한국어 감정 분석용 샘플"""
    return "정말 실망스럽습니다. 다시는 안 삽니다."


# ============================================================================
# 채팅 테스트 데이터
# ============================================================================

@pytest.fixture
def sample_chat_message():
    """기본 채팅 메시지"""
    return {"message": "안녕하세요, 오늘 날씨가 어떤가요?"}


@pytest.fixture
def sample_chat_message_english():
    """영어 채팅 메시지"""
    return {"message": "Hello, how are you today?"}


@pytest.fixture
def chat_history_sample():
    """대화 기록 샘플"""
    return [
        {"role": "user", "content": "안녕하세요"},
        {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"},
        {"role": "user", "content": "오늘 날씨가 어때요?"}
    ]


# ============================================================================
# 요약 테스트 데이터
# ============================================================================

@pytest.fixture
def sample_summarization_text():
    """요약용 샘플 텍스트"""
    return {
        "text": """
        인공지능(AI)은 컴퓨터 시스템이 인간의 지능을 모방하여 학습, 추론, 문제 해결 등의
        작업을 수행할 수 있게 하는 기술입니다. 머신러닝은 AI의 한 분야로, 데이터를 통해
        패턴을 학습하고 예측을 수행합니다. 딥러닝은 머신러닝의 한 유형으로, 인공 신경망을
        사용하여 복잡한 패턴을 학습합니다. 이러한 기술들은 의료, 금융, 교통 등 다양한
        분야에서 활용되고 있습니다.
        """
    }


@pytest.fixture
def sample_short_text():
    """짧은 텍스트 (요약 불필요)"""
    return {"text": "안녕하세요."}


# ============================================================================
# 임베딩 테스트 데이터
# ============================================================================

@pytest.fixture
def sample_embedding_text():
    """임베딩용 샘플 텍스트"""
    return {"text": "This is a sample text for embedding."}


@pytest.fixture
def sample_embedding_korean():
    """한국어 임베딩용 샘플"""
    return {"text": "임베딩 테스트를 위한 한국어 문장입니다."}


# ============================================================================
# 이미지 테스트 데이터
# ============================================================================

@pytest.fixture
def valid_png_image():
    """
    유효한 PNG 이미지 데이터 (1x1 픽셀)

    실제 PNG 파일 헤더와 데이터를 포함
    """
    return b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'


@pytest.fixture
def valid_jpeg_image():
    """유효한 JPEG 이미지 데이터 (최소 헤더)"""
    return b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9'


@pytest.fixture
def invalid_image_data():
    """유효하지 않은 이미지 데이터 (텍스트)"""
    return b"This is not an image file"


# ============================================================================
# 모델 설정 Fixture
# ============================================================================

@pytest.fixture
def ollama_models():
    """사용 가능한 Ollama 모델 목록"""
    return [
        "gemma3:4b",
        "llama3.2:3b",
        "mistral:7b"
    ]


@pytest.fixture
def gemini_model():
    """Gemini 모델 설정"""
    return "gemini-2.5-flash"


@pytest.fixture
def default_model():
    """기본 모델"""
    return "gemma3:4b"


# ============================================================================
# Mock Fixtures (외부 서비스 의존성 제거)
# ============================================================================

@pytest.fixture
def mock_ollama_client():
    """
    Ollama 클라이언트 Mock

    Ollama 서비스가 실행 중이지 않아도 테스트 가능
    """
    with patch('ollama.chat') as mock:
        mock.return_value = {
            "message": {
                "content": "안녕하세요! 오늘 날씨가 좋네요."
            }
        }
        yield mock


@pytest.fixture
def mock_ollama_stream():
    """Ollama 스트리밍 응답 Mock"""
    with patch('ollama.chat') as mock:
        def stream_generator():
            yield {"message": {"content": "안녕"}}
            yield {"message": {"content": "하세요"}}
            yield {"message": {"content": "!"}}
        mock.return_value = stream_generator()
        yield mock


@pytest.fixture
def mock_gemini_client():
    """
    Google Gemini API 클라이언트 Mock

    API 키 없이도 테스트 가능
    """
    with patch('google.generativeai.GenerativeModel') as mock:
        instance = MagicMock()
        instance.generate_content.return_value = MagicMock(
            text="테스트 응답입니다."
        )
        mock.return_value = instance
        yield mock


@pytest.fixture
def mock_sentiment_model():
    """
    감정 분석 모델 Mock

    실제 ML 모델 로딩 없이 테스트
    """
    with patch('app.services.sentiment_service.SentimentService') as mock:
        instance = MagicMock()
        instance.analyze.return_value = {
            "label": "positive",
            "confidence": 0.95
        }
        mock.return_value = instance
        yield mock


@pytest.fixture
def mock_keras_model():
    """
    Keras 이미지 분류 모델 Mock

    실제 모델 파일 없이 테스트
    """
    with patch('tensorflow.keras.models.load_model') as mock:
        model = MagicMock()
        model.predict.return_value = [[0.1, 0.9]]  # Dog 클래스 확률
        mock.return_value = model
        yield mock


# ============================================================================
# 헬퍼 Fixtures
# ============================================================================

@pytest.fixture
def sentiment_payload_positive(sample_text_positive):
    """긍정 텍스트 감정 분석 요청 페이로드"""
    return {"text": sample_text_positive, "explain": False}


@pytest.fixture
def sentiment_payload_negative(sample_text_negative):
    """부정 텍스트 감정 분석 요청 페이로드"""
    return {"text": sample_text_negative, "explain": False}


@pytest.fixture
def chat_payload_basic(sample_chat_message):
    """기본 채팅 요청 페이로드"""
    return {
        "message": sample_chat_message["message"],
        "model": "gemma3:4b"
    }


@pytest.fixture
def chat_payload_with_history(sample_chat_message, chat_history_sample):
    """대화 기록 포함 채팅 요청 페이로드"""
    return {
        "message": sample_chat_message["message"],
        "chat_history": chat_history_sample
    }


# ============================================================================
# 테스트 마커 Fixtures
# ============================================================================

@pytest.fixture
def skip_if_no_ollama():
    """
    Ollama 서비스가 없으면 테스트 스킵

    사용법: 테스트 함수에서 이 fixture를 인자로 받음
    """
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', 11434))
        sock.close()
        if result != 0:
            pytest.skip("Ollama service not running")
    except Exception:
        pytest.skip("Cannot check Ollama service")


@pytest.fixture
def skip_if_no_gemini_key():
    """
    Gemini API 키가 없으면 테스트 스킵
    """
    if not os.environ.get('GEMINI_API_KEY'):
        pytest.skip("GEMINI_API_KEY not set")
