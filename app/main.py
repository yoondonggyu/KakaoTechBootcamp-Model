from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routers import predict_routes, sentiment_routes
from app.core.exceptions import APIError, api_error_handler, RequestValidationError, validation_error_handler, global_exception_handler
from app.services.model_service import load_ai_model
from app.services.sentiment_service import get_sentiment_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load image classification model on startup
    try:
        load_ai_model()
        print("✅ Image classification model loaded successfully")
    except Exception as e:
        print(f"⚠️  WARNING: Failed to load image classification model: {e}")
    
    # Load sentiment analysis model on startup
    try:
        get_sentiment_service()
        print("✅ Sentiment analysis model loaded successfully")
    except Exception as e:
        print(f"⚠️  WARNING: Failed to load sentiment analysis model: {e}")
    
    yield
    # Clean up if needed

app = FastAPI(
    title="AI Model Serving API",
    version="1.0.0",
    description="Keras 이미지 분류 모델과 감성 분석 모델을 서빙하는 FastAPI 애플리케이션",
    lifespan=lifespan
)

# Register Routers
app.include_router(predict_routes.router, prefix="/api", tags=["Image Classification"])
app.include_router(sentiment_routes.router, prefix="/api", tags=["Sentiment Analysis"])

# Register Exception Handlers
app.add_exception_handler(APIError, api_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)
app.add_exception_handler(Exception, global_exception_handler)

@app.get("/", tags=["System"])
async def root():
    return {
        "message": "AI Model Serving API is running",
        "version": "1.0.0",
        "endpoints": {
            "image_classification": "/api/predict (POST)",
            "sentiment_analysis": "/api/sentiment (POST)",
            "documentation": "/docs"
        }
    }
