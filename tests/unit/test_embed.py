"""
Tests for embedding module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from bnsnlp.embed import BaseEmbedder, EmbedResult
from bnsnlp.core.exceptions import AdapterError


class TestEmbedResult:
    """Tests for EmbedResult model."""

    def test_embed_result_creation(self):
        """Test creating an EmbedResult."""
        result = EmbedResult(
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            model="test-model",
            dimensions=3,
        )

        assert len(result.embeddings) == 2
        assert result.model == "test-model"
        assert result.dimensions == 3
        assert isinstance(result.metadata, dict)

    def test_embed_result_with_metadata(self):
        """Test EmbedResult with custom metadata."""
        metadata = {"batch_size": 2, "provider": "test"}
        result = EmbedResult(
            embeddings=[[0.1, 0.2]], model="test-model", dimensions=2, metadata=metadata
        )

        assert result.metadata["batch_size"] == 2
        assert result.metadata["provider"] == "test"


class TestOpenAIEmbedder:
    """Tests for OpenAIEmbedder."""

    @pytest.fixture
    def mock_openai_module(self):
        """Mock the openai module import."""
        mock_openai = MagicMock()
        mock_async_openai = MagicMock()
        mock_openai.AsyncOpenAI = mock_async_openai

        # Mock client instance
        mock_client = MagicMock()
        mock_async_openai.return_value = mock_client

        # Mock embeddings response
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
        ]

        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with patch.dict("sys.modules", {"openai": mock_openai}):
            yield mock_client

    @pytest.fixture
    def openai_config(self):
        """Create a test configuration for OpenAI embedder."""
        return {
            "api_key": "test-api-key",
            "model": "text-embedding-3-small",
            "batch_size": 16,
        }

    def test_openai_embedder_missing_api_key(self, mock_openai_module):
        """Test that OpenAIEmbedder raises error when API key is missing."""
        from bnsnlp.embed.openai import OpenAIEmbedder

        config = {"model": "text-embedding-3-small"}

        with pytest.raises(AdapterError) as exc_info:
            OpenAIEmbedder(config)

        assert "API key is required" in str(exc_info.value)

    def test_openai_embedder_initialization(self, openai_config, mock_openai_module):
        """Test OpenAIEmbedder initialization."""
        from bnsnlp.embed.openai import OpenAIEmbedder

        embedder = OpenAIEmbedder(openai_config)

        assert embedder.api_key == "test-api-key"
        assert embedder.model == "text-embedding-3-small"
        assert embedder.batch_size == 16
        assert embedder.max_retries == 3
        assert embedder.retry_delay == 1.0

    def test_openai_embedder_custom_config(self, mock_openai_module):
        """Test OpenAIEmbedder with custom configuration."""
        from bnsnlp.embed.openai import OpenAIEmbedder

        config = {
            "api_key": "test-key",
            "model": "custom-model",
            "batch_size": 32,
            "max_retries": 5,
            "retry_delay": 2.0,
        }

        embedder = OpenAIEmbedder(config)

        assert embedder.model == "custom-model"
        assert embedder.batch_size == 32
        assert embedder.max_retries == 5
        assert embedder.retry_delay == 2.0

    @pytest.mark.asyncio
    async def test_embed_single_text(self, openai_config, mock_openai_module):
        """Test embedding a single text."""
        from bnsnlp.embed.openai import OpenAIEmbedder

        # Setup mock to return single embedding
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_openai_module.embeddings.create = AsyncMock(return_value=mock_response)

        embedder = OpenAIEmbedder(openai_config)
        result = await embedder.embed("Hello world")

        assert isinstance(result, EmbedResult)
        assert len(result.embeddings) == 1
        assert result.embeddings[0] == [0.1, 0.2, 0.3]
        assert result.dimensions == 3
        assert result.model == "text-embedding-3-small"

        # Verify API was called correctly
        mock_openai_module.embeddings.create.assert_called_once()
        call_args = mock_openai_module.embeddings.create.call_args
        assert call_args.kwargs["input"] == ["Hello world"]
        assert call_args.kwargs["model"] == "text-embedding-3-small"

    @pytest.mark.asyncio
    async def test_embed_multiple_texts(self, openai_config, mock_openai_module):
        """Test embedding multiple texts."""
        from bnsnlp.embed.openai import OpenAIEmbedder

        embedder = OpenAIEmbedder(openai_config)
        texts = ["Hello world", "Goodbye world"]
        result = await embedder.embed(texts)

        assert isinstance(result, EmbedResult)
        assert len(result.embeddings) == 2
        assert result.embeddings[0] == [0.1, 0.2, 0.3]
        assert result.embeddings[1] == [0.4, 0.5, 0.6]
        assert result.dimensions == 3

        # Verify API was called
        mock_openai_module.embeddings.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_batch_processing(self, mock_openai_module):
        """Test batch processing with multiple API calls."""
        from bnsnlp.embed.openai import OpenAIEmbedder

        config = {
            "api_key": "test-key",
            "model": "test-model",
            "batch_size": 2,  # Small batch size to trigger multiple calls
        }

        # Setup mock to return different embeddings for each call
        call_count = [0]

        async def mock_create(**kwargs):
            call_count[0] += 1
            batch_size = len(kwargs["input"])
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=[float(i), float(i + 1), float(i + 2)])
                for i in range(batch_size)
            ]
            return mock_response

        mock_openai_module.embeddings.create = mock_create

        embedder = OpenAIEmbedder(config)
        texts = ["text1", "text2", "text3", "text4", "text5"]
        result = await embedder.embed(texts)

        # Should have made 3 API calls (2 + 2 + 1)
        assert call_count[0] == 3
        assert len(result.embeddings) == 5
        assert result.metadata["batch_size"] == 2
        assert result.metadata["total_texts"] == 5

    @pytest.mark.asyncio
    async def test_embed_empty_list_error(self, openai_config, mock_openai_module):
        """Test that embedding empty list raises error."""
        from bnsnlp.embed.openai import OpenAIEmbedder

        embedder = OpenAIEmbedder(openai_config)

        with pytest.raises(AdapterError) as exc_info:
            await embedder.embed([])

        assert "No texts provided" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_embed_api_error_with_retry(self, openai_config, mock_openai_module):
        """Test retry logic on API errors."""
        from bnsnlp.embed.openai import OpenAIEmbedder

        # Setup mock to fail twice then succeed
        call_count = [0]

        async def mock_create(**kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("API Error")

            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
            return mock_response

        mock_openai_module.embeddings.create = mock_create

        config = {**openai_config, "max_retries": 3, "retry_delay": 0.01}
        embedder = OpenAIEmbedder(config)

        result = await embedder.embed("test")

        # Should have retried and succeeded
        assert call_count[0] == 3
        assert len(result.embeddings) == 1

    @pytest.mark.asyncio
    async def test_embed_api_error_max_retries(self, openai_config, mock_openai_module):
        """Test that max retries are respected."""
        from bnsnlp.embed.openai import OpenAIEmbedder

        # Setup mock to always fail
        async def mock_create(**kwargs):
            raise Exception("API Error")

        mock_openai_module.embeddings.create = mock_create

        config = {**openai_config, "max_retries": 2, "retry_delay": 0.01}
        embedder = OpenAIEmbedder(config)

        with pytest.raises(AdapterError) as exc_info:
            await embedder.embed("test")

        assert "Failed to generate embeddings after 2 attempts" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_embed_result_metadata(self, openai_config, mock_openai_module):
        """Test that result includes correct metadata."""
        from bnsnlp.embed.openai import OpenAIEmbedder

        embedder = OpenAIEmbedder(openai_config)
        result = await embedder.embed(["text1", "text2"])

        assert result.metadata["batch_size"] == 16
        assert result.metadata["total_texts"] == 2
        assert result.metadata["provider"] == "openai"
        assert result.model == "text-embedding-3-small"


class TestCohereEmbedder:
    """Tests for CohereEmbedder."""

    @pytest.fixture
    def mock_cohere_module(self):
        """Mock the cohere module import."""
        mock_cohere = MagicMock()
        mock_async_client = MagicMock()
        mock_cohere.AsyncClient = mock_async_client

        # Mock client instance
        mock_client = MagicMock()
        mock_async_client.return_value = mock_client

        # Mock embeddings response
        mock_response = MagicMock()
        mock_response.embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]

        mock_client.embed = AsyncMock(return_value=mock_response)

        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            yield mock_client

    @pytest.fixture
    def cohere_config(self):
        """Create a test configuration for Cohere embedder."""
        return {
            "api_key": "test-api-key",
            "model": "embed-multilingual-v3.0",
            "batch_size": 96,
        }

    def test_cohere_embedder_missing_api_key(self, mock_cohere_module):
        """Test that CohereEmbedder raises error when API key is missing."""
        from bnsnlp.embed.cohere import CohereEmbedder

        config = {"model": "embed-multilingual-v3.0"}

        with pytest.raises(AdapterError) as exc_info:
            CohereEmbedder(config)

        assert "API key is required" in str(exc_info.value)

    def test_cohere_embedder_initialization(self, cohere_config, mock_cohere_module):
        """Test CohereEmbedder initialization."""
        from bnsnlp.embed.cohere import CohereEmbedder

        embedder = CohereEmbedder(cohere_config)

        assert embedder.api_key == "test-api-key"
        assert embedder.model == "embed-multilingual-v3.0"
        assert embedder.batch_size == 96
        assert embedder.max_retries == 3
        assert embedder.retry_delay == 1.0
        assert embedder.input_type == "search_document"

    def test_cohere_embedder_custom_config(self, mock_cohere_module):
        """Test CohereEmbedder with custom configuration."""
        from bnsnlp.embed.cohere import CohereEmbedder

        config = {
            "api_key": "test-key",
            "model": "custom-model",
            "batch_size": 50,
            "max_retries": 5,
            "retry_delay": 2.0,
            "input_type": "search_query",
        }

        embedder = CohereEmbedder(config)

        assert embedder.model == "custom-model"
        assert embedder.batch_size == 50
        assert embedder.max_retries == 5
        assert embedder.retry_delay == 2.0
        assert embedder.input_type == "search_query"

    @pytest.mark.asyncio
    async def test_cohere_embed_single_text(self, cohere_config, mock_cohere_module):
        """Test embedding a single text."""
        from bnsnlp.embed.cohere import CohereEmbedder

        # Setup mock to return single embedding
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2, 0.3]]
        mock_cohere_module.embed = AsyncMock(return_value=mock_response)

        embedder = CohereEmbedder(cohere_config)
        result = await embedder.embed("Hello world")

        assert isinstance(result, EmbedResult)
        assert len(result.embeddings) == 1
        assert result.embeddings[0] == [0.1, 0.2, 0.3]
        assert result.dimensions == 3
        assert result.model == "embed-multilingual-v3.0"

        # Verify API was called correctly
        mock_cohere_module.embed.assert_called_once()
        call_args = mock_cohere_module.embed.call_args
        assert call_args.kwargs["texts"] == ["Hello world"]
        assert call_args.kwargs["model"] == "embed-multilingual-v3.0"
        assert call_args.kwargs["input_type"] == "search_document"

    @pytest.mark.asyncio
    async def test_cohere_embed_multiple_texts(self, cohere_config, mock_cohere_module):
        """Test embedding multiple texts."""
        from bnsnlp.embed.cohere import CohereEmbedder

        embedder = CohereEmbedder(cohere_config)
        texts = ["Hello world", "Goodbye world"]
        result = await embedder.embed(texts)

        assert isinstance(result, EmbedResult)
        assert len(result.embeddings) == 2
        assert result.embeddings[0] == [0.1, 0.2, 0.3]
        assert result.embeddings[1] == [0.4, 0.5, 0.6]
        assert result.dimensions == 3

        # Verify API was called
        mock_cohere_module.embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_cohere_embed_batch_processing(self, mock_cohere_module):
        """Test batch processing with multiple API calls."""
        from bnsnlp.embed.cohere import CohereEmbedder

        config = {
            "api_key": "test-key",
            "model": "test-model",
            "batch_size": 2,  # Small batch size to trigger multiple calls
        }

        # Setup mock to return different embeddings for each call
        call_count = [0]

        async def mock_embed(**kwargs):
            call_count[0] += 1
            batch_size = len(kwargs["texts"])
            mock_response = MagicMock()
            mock_response.embeddings = [
                [float(i), float(i + 1), float(i + 2)] for i in range(batch_size)
            ]
            return mock_response

        mock_cohere_module.embed = mock_embed

        embedder = CohereEmbedder(config)
        texts = ["text1", "text2", "text3", "text4", "text5"]
        result = await embedder.embed(texts)

        # Should have made 3 API calls (2 + 2 + 1)
        assert call_count[0] == 3
        assert len(result.embeddings) == 5
        assert result.metadata["batch_size"] == 2
        assert result.metadata["total_texts"] == 5

    @pytest.mark.asyncio
    async def test_cohere_embed_empty_list_error(self, cohere_config, mock_cohere_module):
        """Test that embedding empty list raises error."""
        from bnsnlp.embed.cohere import CohereEmbedder

        embedder = CohereEmbedder(cohere_config)

        with pytest.raises(AdapterError) as exc_info:
            await embedder.embed([])

        assert "No texts provided" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cohere_embed_api_error_with_retry(self, cohere_config, mock_cohere_module):
        """Test retry logic on API errors."""
        from bnsnlp.embed.cohere import CohereEmbedder

        # Setup mock to fail twice then succeed
        call_count = [0]

        async def mock_embed(**kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("API Error")

            mock_response = MagicMock()
            mock_response.embeddings = [[0.1, 0.2, 0.3]]
            return mock_response

        mock_cohere_module.embed = mock_embed

        config = {**cohere_config, "max_retries": 3, "retry_delay": 0.01}
        embedder = CohereEmbedder(config)

        result = await embedder.embed("test")

        # Should have retried and succeeded
        assert call_count[0] == 3
        assert len(result.embeddings) == 1

    @pytest.mark.asyncio
    async def test_cohere_embed_api_error_max_retries(self, cohere_config, mock_cohere_module):
        """Test that max retries are respected."""
        from bnsnlp.embed.cohere import CohereEmbedder

        # Setup mock to always fail
        async def mock_embed(**kwargs):
            raise Exception("API Error")

        mock_cohere_module.embed = mock_embed

        config = {**cohere_config, "max_retries": 2, "retry_delay": 0.01}
        embedder = CohereEmbedder(config)

        with pytest.raises(AdapterError) as exc_info:
            await embedder.embed("test")

        assert "Failed to generate embeddings after 2 attempts" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cohere_embed_result_metadata(self, cohere_config, mock_cohere_module):
        """Test that result includes correct metadata."""
        from bnsnlp.embed.cohere import CohereEmbedder

        embedder = CohereEmbedder(cohere_config)
        result = await embedder.embed(["text1", "text2"])

        assert result.metadata["batch_size"] == 96
        assert result.metadata["total_texts"] == 2
        assert result.metadata["provider"] == "cohere"
        assert result.metadata["input_type"] == "search_document"
        assert result.model == "embed-multilingual-v3.0"


class TestHuggingFaceEmbedder:
    """Tests for HuggingFaceEmbedder."""

    @pytest.fixture
    def mock_sentence_transformers(self):
        """Mock the sentence_transformers module."""
        mock_st = MagicMock()
        mock_model_class = MagicMock()
        mock_st.SentenceTransformer = mock_model_class

        # Mock model instance
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        # Mock encode method
        import numpy as np

        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_model.get_sentence_embedding_dimension.return_value = 3

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            yield mock_model

    @pytest.fixture
    def mock_torch(self):
        """Mock the torch module."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            yield mock_torch

    @pytest.fixture
    def hf_config(self):
        """Create a test configuration for HuggingFace embedder."""
        return {
            "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "use_gpu": True,
            "batch_size": 32,
        }

    def test_hf_embedder_initialization(
        self, hf_config, mock_sentence_transformers, mock_torch
    ):
        """Test HuggingFaceEmbedder initialization."""
        from bnsnlp.embed.huggingface import HuggingFaceEmbedder

        embedder = HuggingFaceEmbedder(hf_config)

        assert (
            embedder.model_name
            == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        assert embedder.use_gpu is True
        assert embedder.batch_size == 32
        assert embedder.normalize_embeddings is True
        assert embedder.device == "cuda"
        assert embedder.dimensions == 3

    def test_hf_embedder_cpu_fallback(self, mock_sentence_transformers):
        """Test that embedder falls back to CPU when GPU is not available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from bnsnlp.embed.huggingface import HuggingFaceEmbedder

            config = {"model": "test-model", "use_gpu": True}
            embedder = HuggingFaceEmbedder(config)

            assert embedder.device == "cpu"

    def test_hf_embedder_custom_device(self, mock_sentence_transformers, mock_torch):
        """Test HuggingFaceEmbedder with custom device."""
        from bnsnlp.embed.huggingface import HuggingFaceEmbedder

        config = {"model": "test-model", "device": "cuda:1"}
        embedder = HuggingFaceEmbedder(config)

        assert embedder.device == "cuda:1"

    def test_hf_embedder_custom_config(self, mock_sentence_transformers, mock_torch):
        """Test HuggingFaceEmbedder with custom configuration."""
        from bnsnlp.embed.huggingface import HuggingFaceEmbedder

        config = {
            "model": "custom-model",
            "use_gpu": False,
            "batch_size": 64,
            "normalize_embeddings": False,
        }

        embedder = HuggingFaceEmbedder(config)

        assert embedder.model_name == "custom-model"
        assert embedder.batch_size == 64
        assert embedder.normalize_embeddings is False
        assert embedder.device == "cpu"

    @pytest.mark.asyncio
    async def test_hf_embed_single_text(
        self, hf_config, mock_sentence_transformers, mock_torch
    ):
        """Test embedding a single text."""
        from bnsnlp.embed.huggingface import HuggingFaceEmbedder

        import numpy as np

        # Setup mock to return single embedding
        mock_sentence_transformers.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        embedder = HuggingFaceEmbedder(hf_config)
        result = await embedder.embed("Hello world")

        assert isinstance(result, EmbedResult)
        assert len(result.embeddings) == 1
        assert result.embeddings[0] == [0.1, 0.2, 0.3]
        assert result.dimensions == 3
        assert (
            result.model
            == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # Verify encode was called correctly
        mock_sentence_transformers.encode.assert_called_once()
        call_args = mock_sentence_transformers.encode.call_args
        assert call_args[0][0] == ["Hello world"]

    @pytest.mark.asyncio
    async def test_hf_embed_multiple_texts(
        self, hf_config, mock_sentence_transformers, mock_torch
    ):
        """Test embedding multiple texts."""
        from bnsnlp.embed.huggingface import HuggingFaceEmbedder

        embedder = HuggingFaceEmbedder(hf_config)
        texts = ["Hello world", "Goodbye world"]
        result = await embedder.embed(texts)

        assert isinstance(result, EmbedResult)
        assert len(result.embeddings) == 2
        assert result.embeddings[0] == [0.1, 0.2, 0.3]
        assert result.embeddings[1] == [0.4, 0.5, 0.6]
        assert result.dimensions == 3

        # Verify encode was called
        mock_sentence_transformers.encode.assert_called_once()

    @pytest.mark.asyncio
    async def test_hf_embed_empty_list_error(
        self, hf_config, mock_sentence_transformers, mock_torch
    ):
        """Test that embedding empty list raises error."""
        from bnsnlp.embed.huggingface import HuggingFaceEmbedder

        embedder = HuggingFaceEmbedder(hf_config)

        with pytest.raises(AdapterError) as exc_info:
            await embedder.embed([])

        assert "No texts provided" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_hf_embed_result_metadata(
        self, hf_config, mock_sentence_transformers, mock_torch
    ):
        """Test that result includes correct metadata."""
        from bnsnlp.embed.huggingface import HuggingFaceEmbedder

        embedder = HuggingFaceEmbedder(hf_config)
        result = await embedder.embed(["text1", "text2"])

        assert result.metadata["batch_size"] == 32
        assert result.metadata["total_texts"] == 2
        assert result.metadata["provider"] == "huggingface"
        assert result.metadata["device"] == "cuda"
        assert result.metadata["gpu_available"] is True
        assert result.metadata["normalize_embeddings"] is True

    @pytest.mark.asyncio
    async def test_hf_embed_gpu_detection(self, mock_sentence_transformers):
        """Test GPU detection in metadata."""
        from bnsnlp.embed.huggingface import HuggingFaceEmbedder

        # Test with GPU available
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            config = {"model": "test-model", "use_gpu": True}
            embedder = HuggingFaceEmbedder(config)
            result = await embedder.embed("test")

            assert result.metadata["gpu_available"] is True
            assert result.metadata["device"] == "cuda"

        # Test with GPU not available
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            config = {"model": "test-model", "use_gpu": True}
            embedder = HuggingFaceEmbedder(config)
            result = await embedder.embed("test")

            assert result.metadata["gpu_available"] is False
            assert result.metadata["device"] == "cpu"

    @pytest.mark.asyncio
    async def test_hf_embed_encoding_error(
        self, hf_config, mock_sentence_transformers, mock_torch
    ):
        """Test error handling during encoding."""
        from bnsnlp.embed.huggingface import HuggingFaceEmbedder

        # Setup mock to raise error
        mock_sentence_transformers.encode.side_effect = Exception("Encoding error")

        embedder = HuggingFaceEmbedder(hf_config)

        with pytest.raises(AdapterError) as exc_info:
            await embedder.embed("test")

        assert "Failed to generate embeddings" in str(exc_info.value)



