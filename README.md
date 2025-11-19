# AI Model Serving API

두 가지 AI 모델을 FastAPI로 서빙하는 프로젝트입니다:
1. **이미지 분류 모델**: Keras 기반 강아지/고양이 분류
2. **감성 분석 모델**: Naive Bayes 기반 텍스트 감성 분석 (positive/negative)

## 프로젝트 구조

```
FASTAPI_Project_model/
├── app/
│   ├── main.py                     # FastAPI 앱 진입점
│   ├── routers/
│   │   ├── predict_routes.py       # 이미지 분류 API 라우터
│   │   └── sentiment_routes.py     # 감성 분석 API 라우터
│   ├── services/
│   │   ├── model_service.py        # 이미지 분류 모델 서비스
│   │   └── sentiment_service.py    # 감성 분석 모델 서비스
│   ├── schemas/
│   │   └── prediction.py           # Pydantic 응답 스키마
│   └── core/
│       └── exceptions.py           # 예외 처리
├── assets/
│   ├── keras_model.h5              # Keras 이미지 분류 모델
│   └── labels.txt                  # 클래스 레이블 (Dog, Cat)
├── models/
│   └── sentiment.py                # 감성 분석 모델 구현
└── requirements.txt                # 의존성 패키지
```

## 실행 방법

### 1. 가상환경 활성화
```bash
conda activate env_fastapi
```

### 2. 필요한 패키지 설치
```bash
pip install fastapi uvicorn keras tensorflow pillow numpy
```

### 3. 서버 실행
```bash
cd /Users/yoon-dong-gyu/kakao_bootcamp/FASTAPI_Project_model
uvicorn app.main:app --reload --port 8001
```

서버가 `http://localhost:8001`에서 실행됩니다.

## API 테스트

### 1. 헬스 체크
```bash
curl http://localhost:8001/
```

**응답 예시:**
```json
{
  "message": "AI Model Serving API is running",
  "version": "1.0.0",
  "endpoints": {
    "image_classification": "/api/predict (POST)",
    "sentiment_analysis": "/api/sentiment (POST)",
    "documentation": "/docs"
  }
}
```

### 2. 이미지 분류 예측 (강아지/고양이)

#### curl로 테스트
```bash
curl -X POST "http://localhost:8001/api/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/dog_or_cat_image.jpg"
```

**실제 예시 (프로젝트 내 이미지 사용):**
```bash
# 9week 디렉터리의 고양이 이미지로 테스트
curl -X POST "http://localhost:8001/api/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/Users/yoon-dong-gyu/kakao_bootcamp/9week(20251110~20251115)/cls_cats_and_dogs/cat/cat.1.jpg"

# 강아지 이미지로 테스트
curl -X POST "http://localhost:8001/api/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/Users/yoon-dong-gyu/kakao_bootcamp/9week(20251110~20251115)/cls_cats_and_dogs/dog/dog.1.jpg"
```

#### Postman으로 테스트
1. **Method**: POST
2. **URL**: `http://localhost:8001/api/predict`
3. **Body 탭**: `form-data` 선택
4. **Key**: `file` (Type을 File로 변경)
5. **Value**: 강아지 또는 고양이 이미지 파일 선택
6. **Send** 클릭

**정상 응답 예시:**
```json
{
  "class_name": "Dog",
  "confidence_score": 0.9876543
}
```

또는

```json
{
  "class_name": "Cat",
  "confidence_score": 0.9654321
}
```

### 3. 텍스트 감성 분석

#### curl로 테스트
```bash
# 기본 감성 분석 (토큰 영향도 제외)
curl -X POST "http://localhost:8001/api/sentiment" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I really love this camera, the picture quality is amazing!",
    "explain": false
  }'

# 토큰 영향도 포함
curl -X POST "http://localhost:8001/api/sentiment" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is the worst product ever, totally disappointed",
    "explain": true
  }'
```

#### Postman으로 테스트
1. **Method**: POST
2. **URL**: `http://localhost:8001/api/sentiment`
3. **Body 탭**: `raw` 선택, `JSON` 타입 선택
4. **Body 내용**:
```json
{
  "text": "I absolutely love this product, it works great",
  "explain": true
}
```
5. **Send** 클릭

**정상 응답 예시 (긍정):**
```json
{
  "label": "positive",
  "confidence": 0.9876543,
  "probabilities": {
    "positive": 0.9876543,
    "negative": 0.0123457
  },
  "top_tokens": [
    {"token": "love", "impact": 2.5},
    {"token": "great", "impact": 1.8},
    {"token": "absolutely", "impact": 1.2}
  ]
}
```

**정상 응답 예시 (부정):**
```json
{
  "label": "negative",
  "confidence": 0.9654321,
  "probabilities": {
    "positive": 0.0345679,
    "negative": 0.9654321
  },
  "top_tokens": [
    {"token": "worst", "impact": -3.2},
    {"token": "disappointed", "impact": -2.1}
  ]
}
```

### 4. 예외 처리 테스트

#### 이미지 분류 - 파일 없이 요청
```bash
curl -X POST "http://localhost:8001/api/predict"
```

**응답 (422 Unprocessable Entity):**
```json
{
  "message": "validation_error",
  "data": {
    "details": "..."
  }
}
```

#### 이미지 분류 - 잘못된 파일 형식
```bash
curl -X POST "http://localhost:8001/api/predict" \
  -F "file=@/path/to/document.pdf"
```

**응답 (400 Bad Request):**
```json
{
  "message": "invalid_file_type",
  "data": {
    "allowed": ["jpg", "png", "jpeg"]
  }
}
```

#### 감성 분석 - 빈 텍스트
```bash
curl -X POST "http://localhost:8001/api/sentiment" \
  -H "Content-Type: application/json" \
  -d '{"text": "", "explain": false}'
```

**응답 (400 Bad Request):**
```json
{
  "message": "text_required",
  "data": null
}
```

#### 감성 분석 - 알파벳 없는 텍스트
```bash
curl -X POST "http://localhost:8001/api/sentiment" \
  -H "Content-Type: application/json" \
  -d '{"text": "123 456", "explain": false}'
```

**응답 (400 Bad Request):**
```json
{
  "message": "text must contain alphabetic characters",
  "data": null
}
```

## API 문서

서버 실행 후 다음 URL에서 자동 생성된 API 문서를 확인할 수 있습니다:
- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

## 주요 기능

### 이미지 분류 모델 (Keras)
- ✅ Keras/TensorFlow 모델 로딩 (앱 시작 시)
- ✅ 이미지 파일 업로드 및 전처리
- ✅ 강아지/고양이 분류 예측
- ✅ 파일 형식 검증 (jpg, png, jpeg만 허용)

### 감성 분석 모델 (Naive Bayes)
- ✅ 메모리 기반 경량 모델 (즉시 로딩)
- ✅ 영어 텍스트 감성 분석 (positive/negative)
- ✅ 확률 분포 및 신뢰도 제공
- ✅ 토큰별 영향도 분석 (옵션)
- ✅ 입력 검증 (빈 텍스트, 알파벳 포함 여부)

### 공통 기능
- ✅ 일관된 JSON 응답 포맷
- ✅ Pydantic 스키마 검증
- ✅ 포괄적인 예외 처리
- ✅ 자동 API 문서 생성 (Swagger UI)
- ✅ 앱 시작 시 모델 자동 로딩

## 트러블슈팅

### 모델 로딩 실패
- `assets/keras_model.h5`와 `assets/labels.txt` 파일이 존재하는지 확인
- Keras/TensorFlow가 설치되어 있는지 확인: `pip list | grep -i keras`

### 포트 충돌
- 백엔드 서버가 8000번 포트를 사용하고 있으므로, 모델 서버는 8001번 포트 사용
- 다른 포트로 변경하려면: `uvicorn app.main:app --reload --port 원하는포트번호`

### 이미지 업로드 오류
- 파일 크기가 너무 크지 않은지 확인
- 파일 형식이 jpg, png, jpeg 중 하나인지 확인

