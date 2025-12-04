# 🤖 동물 감정일기 - AI Model API

> Keras 이미지 분류 및 Gemini 감정 분석 모델 서빙 API

## 📋 프로젝트 소개

**동물 감정일기** 커뮤니티의 AI 모델 서빙 서버입니다.  
이미지 분류(강아지/고양이)와 Gemini 기반 감정 분석 기능을 제공합니다.

## 🔗 관련 저장소

| 저장소 | 설명 | 링크 |
|--------|------|------|
| **Frontend** | Vanilla JS 기반 웹 UI | [KakaoTechBootcamp-Frontend](https://github.com/yoondonggyu/KakaoTechBootcamp-Frontend) |
| **Backend** | FastAPI 기반 REST API | [KakaoTechBootcamp-Backend](https://github.com/yoondonggyu/KakaoTechBootcamp-Backend) |
| **Model** | AI 모델 서빙 API | [현재 저장소](https://github.com/yoondonggyu/KakaoTechBootcamp-Model) |

## ✨ 주요 기능

### 🖼️ 이미지 분류
- Keras CNN 모델 기반
- 강아지(Dog) / 고양이(Cat) 분류
- 신뢰도 점수 반환

### 💭 감정 분석
- **기본 모델**: 영어 텍스트 감정 분석
- **Gemini API**: 한글/영어 모두 지원하는 감정 분석
- 긍정(positive) / 부정(negative) / 중립(neutral) 분류

### 💬 채팅 (추가 기능)
- Ollama 로컬 LLM 연동
- 스트리밍 응답 지원

## 🛠 기술 스택

| 분류 | 기술 |
|------|------|
| **Framework** | FastAPI |
| **Language** | Python 3.10+ |
| **ML Framework** | TensorFlow / Keras |
| **LLM** | Google Gemini 2.5 Flash |
| **Server** | Uvicorn |

## 📁 프로젝트 구조

```
FASTAPI_Project_model/
├── app/
│   ├── main.py              # FastAPI 앱 진입점
│   ├── core/
│   │   ├── config.py        # 환경 변수 설정
│   │   └── exceptions.py    # 커스텀 예외
│   ├── routers/
│   │   ├── predict_routes.py     # 이미지 분류 API
│   │   ├── sentiment_routes.py   # 감정 분석 API
│   │   └── chat_routes.py        # 채팅 API
│   ├── services/
│   │   ├── model_service.py      # 이미지 분류 서비스
│   │   ├── sentiment_service.py  # 감정 분석 서비스
│   │   └── gemini_service.py     # Gemini API 서비스
│   └── schemas/
├── models/                  # 학습된 모델 파일
│   └── sentiment/
├── .env.example
└── requirements.txt
```

## 🚀 실행 방법

### 1. 저장소 클론
```bash
git clone https://github.com/yoondonggyu/KakaoTechBootcamp-Model.git
cd KakaoTechBootcamp-Model
```

### 2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정
```bash
cp .env.example .env
# .env 파일을 편집하여 GEMINI_API_KEY 설정
```

### 5. 서버 실행
```bash
uvicorn app.main:app --reload --port 8001
```

### 6. API 문서 확인
```
http://localhost:8001/docs
```

## 📚 API 명세

### 이미지 분류 API

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/predict` | 이미지 분류 (Dog/Cat) |

**Request**: `multipart/form-data` (file)

**Response**:
```json
{
  "class_name": "Dog",
  "confidence_score": 0.9876
}
```

### 감정 분석 API

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/sentiment` | 기본 감정 분석 (영어) |
| POST | `/api/sentiment/gemini` | Gemini 감정 분석 (한글/영어) |

**Request**:
```json
{
  "text": "오늘 정말 행복한 하루였어요!",
  "explain": false
}
```

**Response (Gemini)**:
```json
{
  "label": "positive",
  "confidence": 0.95,
  "description": "긍정적인 감정이 느껴지는 문장입니다."
}
```

### 채팅 API

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/chat` | LLM 채팅 |
| WS | `/api/chat/stream` | 스트리밍 채팅 |

## 🔒 환경 변수

```env
# .env
GEMINI_API_KEY=your-gemini-api-key-here
MODEL_API_BASE_URL=http://localhost:8001
LOG_LEVEL=INFO
```

### Gemini API 키 발급
1. [Google AI Studio](https://makersuite.google.com/app/apikey) 접속
2. API Key 생성
3. `.env` 파일에 `GEMINI_API_KEY` 설정

## 🧠 사용된 모델

### 이미지 분류
- **모델**: Keras CNN (Convolutional Neural Network)
- **학습 데이터**: Dogs vs Cats Dataset
- **출력**: Dog / Cat (이진 분류)

### 감정 분석
- **기본 모델**: DistilBERT 기반 감정 분류기
- **Gemini**: Google Gemini 2.5 Flash

## 👨‍💻 개발자

- **윤동규** - [GitHub](https://github.com/yoondonggyu)

## 📝 라이선스

This project is licensed under the MIT License.
