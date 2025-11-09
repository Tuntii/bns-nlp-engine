"""
Tests for FastAPI service endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from bnsnlp.api.service import app


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self, client):
        """Test health check endpoint returns correct status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert data["service"] == "bns-nlp-engine"


class TestPreprocessEndpoint:
    """Tests for /preprocess endpoint."""

    def test_preprocess_basic(self, client, mocker):
        """Test basic preprocessing request."""
        # Mock the TurkishPreprocessor
        mock_result = {
            "text": "merhaba dünya",
            "tokens": ["merhaba", "dünya"],
            "metadata": {"original_length": 15},
        }
        mock_preprocessor = mocker.patch(
            "bnsnlp.preprocess.turkish.TurkishPreprocessor"
        )
        mock_instance = mock_preprocessor.return_value
        mock_instance.process = mocker.AsyncMock(return_value=mock_result)

        response = client.post(
            "/preprocess",
            json={
                "text": "Merhaba DÜNYA!",
                "lowercase": True,
                "remove_punctuation": True,
                "remove_stopwords": False,
                "lemmatize": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "merhaba dünya"
        assert data["tokens"] == ["merhaba", "dünya"]

    def test_preprocess_error_handling(self, client, mocker):
        """Test preprocessing error handling."""
        mock_preprocessor = mocker.patch(
            "bnsnlp.preprocess.turkish.TurkishPreprocessor"
        )
        mock_instance = mock_preprocessor.return_value
        mock_instance.process = mocker.AsyncMock(
            side_effect=Exception("Processing error")
        )

        response = client.post(
            "/preprocess",
            json={"text": "Test text"},
        )

        assert response.status_code == 500
        assert "Preprocessing failed" in response.json()["detail"]


class TestEmbedEndpoint:
    """Tests for /embed endpoint."""

    def test_embed_openai(self, client, mocker):
        """Test embedding with OpenAI provider."""
        mock_result = {
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "model": "text-embedding-3-small",
            "dimensions": 3,
            "metadata": {},
        }
        mock_embedder = mocker.patch("bnsnlp.embed.openai.OpenAIEmbedder")
        mock_instance = mock_embedder.return_value
        mock_instance.embed = mocker.AsyncMock(return_value=mock_result)

        response = client.post(
            "/embed",
            json={
                "texts": ["Merhaba", "Dünya"],
                "provider": "openai",
                "model": "text-embedding-3-small",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["embeddings"]) == 2
        assert data["model"] == "text-embedding-3-small"
        assert data["dimensions"] == 3

    def test_embed_invalid_provider(self, client):
        """Test embedding with invalid provider."""
        response = client.post(
            "/embed",
            json={
                "texts": ["Test"],
                "provider": "invalid_provider",
            },
        )

        assert response.status_code == 400
        assert "Invalid provider" in response.json()["detail"]

    def test_embed_error_handling(self, client, mocker):
        """Test embedding error handling."""
        mock_embedder = mocker.patch("bnsnlp.embed.openai.OpenAIEmbedder")
        mock_instance = mock_embedder.return_value
        mock_instance.embed = mocker.AsyncMock(
            side_effect=Exception("Embedding error")
        )

        response = client.post(
            "/embed",
            json={
                "texts": ["Test"],
                "provider": "openai",
            },
        )

        assert response.status_code == 500
        assert "Embedding generation failed" in response.json()["detail"]


class TestSearchEndpoint:
    """Tests for /search endpoint."""

    def test_search_faiss(self, client, mocker):
        """Test search with FAISS backend."""
        from bnsnlp.embed.base import EmbedResult
        from bnsnlp.search.base import SearchResponse, SearchResult
        
        # Mock embedder
        mock_embed_result = EmbedResult(
            embeddings=[[0.1, 0.2, 0.3]],
            model="text-embedding-3-small",
            dimensions=3,
            metadata={},
        )
        mock_embedder = mocker.patch("bnsnlp.embed.openai.OpenAIEmbedder")
        mock_embedder_instance = mock_embedder.return_value
        mock_embedder_instance.embed = mocker.AsyncMock(return_value=mock_embed_result)

        # Mock searcher
        mock_search_result = SearchResponse(
            results=[
                SearchResult(
                    id="doc_1",
                    score=0.95,
                    text="Merhaba dünya",
                    metadata={},
                )
            ],
            query_time_ms=10.5,
            metadata={},
        )
        mock_searcher = mocker.patch("bnsnlp.search.faiss.FAISSSearch")
        mock_searcher_instance = mock_searcher.return_value
        mock_searcher_instance.search = mocker.AsyncMock(return_value=mock_search_result)

        response = client.post(
            "/search",
            json={
                "query": "Türkçe NLP",
                "top_k": 5,
                "provider": "faiss",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["id"] == "doc_1"
        assert data["results"][0]["score"] == 0.95

    def test_search_with_filters(self, client, mocker):
        """Test search with metadata filters."""
        from bnsnlp.embed.base import EmbedResult
        from bnsnlp.search.base import SearchResponse
        
        mock_embedder = mocker.patch("bnsnlp.embed.openai.OpenAIEmbedder")
        mock_embedder_instance = mock_embedder.return_value
        mock_embedder_instance.embed = mocker.AsyncMock(
            return_value=EmbedResult(embeddings=[[0.1, 0.2]], model="test", dimensions=2, metadata={})
        )

        mock_searcher = mocker.patch("bnsnlp.search.faiss.FAISSSearch")
        mock_searcher_instance = mock_searcher.return_value
        mock_searcher_instance.search = mocker.AsyncMock(
            return_value=SearchResponse(results=[], query_time_ms=5.0, metadata={})
        )

        response = client.post(
            "/search",
            json={
                "query": "test",
                "top_k": 10,
                "provider": "faiss",
                "filters": {"category": "nlp"},
            },
        )

        assert response.status_code == 200
        # Verify filters were passed to search
        mock_searcher_instance.search.assert_called_once()
        call_kwargs = mock_searcher_instance.search.call_args[1]
        assert call_kwargs["filters"] == {"category": "nlp"}

    def test_search_invalid_provider(self, client, mocker):
        """Test search with invalid provider."""
        from bnsnlp.embed.base import EmbedResult
        
        mock_embedder = mocker.patch("bnsnlp.embed.openai.OpenAIEmbedder")
        mock_embedder_instance = mock_embedder.return_value
        mock_embedder_instance.embed = mocker.AsyncMock(
            return_value=EmbedResult(embeddings=[[0.1]], model="test", dimensions=1, metadata={})
        )

        response = client.post(
            "/search",
            json={
                "query": "test",
                "provider": "invalid_provider",
            },
        )

        assert response.status_code == 400
        assert "Invalid provider" in response.json()["detail"]


class TestClassifyEndpoint:
    """Tests for /classify endpoint."""

    def test_classify_basic(self, client, mocker):
        """Test basic classification request."""
        mock_result = {
            "intent": "question",
            "intent_confidence": 0.95,
            "entities": [
                {
                    "text": "yarın",
                    "type": "DATE",
                    "start": 0,
                    "end": 5,
                    "confidence": 0.9,
                }
            ],
            "metadata": {},
        }
        mock_classifier = mocker.patch("bnsnlp.classify.turkish.TurkishClassifier")
        mock_instance = mock_classifier.return_value
        mock_instance.classify = mocker.AsyncMock(return_value=mock_result)

        response = client.post(
            "/classify",
            json={"text": "Yarın saat kaçta toplantı var?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "question"
        assert data["intent_confidence"] == 0.95
        assert len(data["entities"]) == 1
        assert data["entities"][0]["type"] == "DATE"

    def test_classify_error_handling(self, client, mocker):
        """Test classification error handling."""
        mock_classifier = mocker.patch("bnsnlp.classify.turkish.TurkishClassifier")
        mock_instance = mock_classifier.return_value
        mock_instance.classify = mocker.AsyncMock(
            side_effect=Exception("Classification error")
        )

        response = client.post(
            "/classify",
            json={"text": "Test text"},
        )

        assert response.status_code == 500
        assert "Classification failed" in response.json()["detail"]


class TestAPIDocumentation:
    """Tests for API documentation endpoints."""

    def test_openapi_schema(self, client):
        """Test OpenAPI schema is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "bns-nlp-engine API"
        assert "paths" in schema

    def test_docs_endpoint(self, client):
        """Test Swagger UI docs endpoint."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_endpoint(self, client):
        """Test ReDoc endpoint."""
        response = client.get("/redoc")
        assert response.status_code == 200
