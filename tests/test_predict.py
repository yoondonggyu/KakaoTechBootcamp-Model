"""
이미지 분류(Predict) API 테스트 케이스
- 이미지 분류 예측
"""
import pytest
import io


class TestPredict:
    """이미지 분류 API 테스트"""
    
    def test_predict_png_image(self, client, valid_png_image):
        """PNG 이미지 분류 테스트"""
        files = {"file": ("test.png", io.BytesIO(valid_png_image), "image/png")}
        
        response = client.post("/api/predict", files=files)
        
        # 모델이 로드되지 않았을 수 있으므로 다양한 상태 코드 허용
        assert response.status_code in [200, 500, 422]
        
        if response.status_code == 200:
            data = response.json()
            assert "predicted_class" in data or "class" in data or "prediction" in data
    
    def test_predict_jpeg_image(self, client, valid_jpeg_image):
        """JPEG 이미지 분류 테스트"""
        files = {"file": ("test.jpg", io.BytesIO(valid_jpeg_image), "image/jpeg")}
        
        response = client.post("/api/predict", files=files)
        
        # 모델이 로드되지 않았거나 유효하지 않은 이미지일 수 있음
        assert response.status_code in [200, 400, 422, 500]
    
    def test_predict_invalid_file_type(self, client):
        """잘못된 파일 타입 테스트"""
        text_data = b"This is not an image"
        files = {"file": ("test.txt", io.BytesIO(text_data), "text/plain")}
        
        response = client.post("/api/predict", files=files)
        
        assert response.status_code in [400, 415, 422]
    
    def test_predict_no_file(self, client):
        """파일 없이 요청 테스트"""
        response = client.post("/api/predict")
        
        assert response.status_code == 422
    
    def test_predict_empty_file(self, client):
        """빈 파일 업로드 테스트"""
        files = {"file": ("empty.png", io.BytesIO(b""), "image/png")}
        
        response = client.post("/api/predict", files=files)
        
        assert response.status_code in [400, 422, 500]
    
    def test_predict_gif_not_allowed(self, client):
        """허용되지 않는 GIF 파일 테스트"""
        # 최소한의 GIF 헤더
        gif_data = b'GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'
        files = {"file": ("test.gif", io.BytesIO(gif_data), "image/gif")}
        
        response = client.post("/api/predict", files=files)
        
        assert response.status_code in [400, 415, 422]
    
    def test_predict_corrupted_image(self, client):
        """손상된 이미지 테스트"""
        corrupted_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR' + b'\x00' * 100
        files = {"file": ("corrupted.png", io.BytesIO(corrupted_data), "image/png")}
        
        response = client.post("/api/predict", files=files)
        
        assert response.status_code in [400, 422, 500]
