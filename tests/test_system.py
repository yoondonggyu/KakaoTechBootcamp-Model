"""
시스템 및 기본 엔드포인트 테스트 케이스
"""
import pytest


class TestSystem:
    """시스템 API 테스트"""
    
    def test_root_endpoint(self, client):
        """루트 엔드포인트 테스트"""
        response = client.get("/")
        
        assert response.status_code == 200
    
    def test_docs_endpoint(self, client):
        """Swagger 문서 엔드포인트 테스트"""
        response = client.get("/docs")
        
        assert response.status_code == 200
    
    def test_redoc_endpoint(self, client):
        """ReDoc 문서 엔드포인트 테스트"""
        response = client.get("/redoc")
        
        assert response.status_code == 200
    
    def test_openapi_json(self, client):
        """OpenAPI JSON 스키마 테스트"""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data
    
    def test_static_files_accessible(self, client):
        """정적 파일 접근 테스트"""
        response = client.get("/static/index.html")
        
        # 파일이 존재하면 200, 없으면 404
        assert response.status_code in [200, 404]
    
    def test_invalid_endpoint(self, client):
        """존재하지 않는 엔드포인트 테스트"""
        response = client.get("/api/nonexistent")
        
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """허용되지 않는 HTTP 메서드 테스트"""
        # GET은 /api/predict에서 허용되지 않음
        response = client.get("/api/predict")
        
        assert response.status_code == 405


class TestCORS:
    """CORS 설정 테스트"""
    
    def test_cors_allowed_origin(self, client):
        """허용된 Origin에서의 요청 테스트"""
        headers = {"Origin": "http://localhost:3000"}
        response = client.get("/", headers=headers)
        
        assert response.status_code == 200
    
    def test_cors_preflight(self, client):
        """CORS Preflight 요청 테스트"""
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type"
        }
        response = client.options("/api/predict", headers=headers)
        
        # Preflight 요청 성공
        assert response.status_code in [200, 204]
