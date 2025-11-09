"""
Tests for search module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from bnsnlp.search import BaseSearch, SearchResponse, SearchResult
from bnsnlp.core.exceptions import AdapterError


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_search_result_creation(self):
        """Test creating a SearchResult."""
        result = SearchResult(
            id="doc_123",
            score=0.95,
            text="Merhaba dünya",
            metadata={"source": "test.txt"},
        )

        assert result.id == "doc_123"
        assert result.score == 0.95
        assert result.text == "Merhaba dünya"
        assert result.metadata["source"] == "test.txt"

    def test_search_result_default_metadata(self):
        """Test SearchResult with default metadata."""
        result = SearchResult(
            id="doc_456",
            score=0.85,
            text="Test document",
        )

        assert isinstance(result.metadata, dict)
        assert len(result.metadata) == 0


class TestSearchResponse:
    """Tests for SearchResponse model."""

    def test_search_response_creation(self):
        """Test creating a SearchResponse."""
        results = [
            SearchResult(id="doc_1", score=0.95, text="Text 1"),
            SearchResult(id="doc_2", score=0.85, text="Text 2"),
        ]

        response = SearchResponse(
            results=results,
            query_time_ms=15.5,
            metadata={"total_documents": 1000},
        )

        assert len(response.results) == 2
        assert response.query_time_ms == 15.5
        assert response.metadata["total_documents"] == 1000

    def test_search_response_default_metadata(self):
        """Test SearchResponse with default metadata."""
        response = SearchResponse(
            results=[],
            query_time_ms=10.0,
        )

        assert isinstance(response.metadata, dict)
        assert len(response.metadata) == 0


class TestQdrantSearch:
    """Tests for QdrantSearch adapter."""

    @pytest.fixture
    def mock_qdrant_module(self):
        """Mock the qdrant_client module."""
        # Create mock objects
        mock_client = MagicMock()
        mock_client_class = MagicMock(return_value=mock_client)
        
        # Mock collections response
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections = AsyncMock(return_value=mock_collections)
        mock_client.create_collection = AsyncMock()
        mock_client.upsert = AsyncMock()

        # Mock search response
        mock_search_result = MagicMock()
        mock_search_result.id = "doc_123"
        mock_search_result.score = 0.95
        mock_search_result.payload = {"text": "Merhaba dünya", "source": "test.txt"}
        mock_client.search = AsyncMock(return_value=[mock_search_result])

        # Mock models
        mock_distance = MagicMock()
        mock_distance.COSINE = "COSINE"
        mock_point_struct = MagicMock()
        mock_vector_params = MagicMock()
        mock_filter = MagicMock()

        # Create mock qdrant module
        mock_qdrant = MagicMock()
        mock_qdrant.AsyncQdrantClient = mock_client_class
        mock_qdrant.models = MagicMock()
        mock_qdrant.models.Distance = mock_distance
        mock_qdrant.models.PointStruct = mock_point_struct
        mock_qdrant.models.VectorParams = mock_vector_params
        mock_qdrant.models.Filter = mock_filter

        with patch.dict(
            "sys.modules",
            {
                "qdrant_client": mock_qdrant,
                "qdrant_client.models": mock_qdrant.models,
            },
        ):
            # Reload the module to pick up the mocked imports
            import importlib
            import bnsnlp.search.qdrant as qdrant_module
            importlib.reload(qdrant_module)
            
            yield mock_client

    @pytest.fixture
    def qdrant_config(self):
        """Create a test configuration for Qdrant search."""
        return {
            "url": "http://localhost:6333",
            "collection": "test_collection",
            "timeout": 30,
        }

    def test_qdrant_search_initialization(self, qdrant_config, mock_qdrant_module):
        """Test QdrantSearch initialization."""
        from bnsnlp.search.qdrant import QdrantSearch

        search = QdrantSearch(qdrant_config)

        assert search.url == "http://localhost:6333"
        assert search.collection == "test_collection"
        assert search.timeout == 30
        assert search.api_key is None

    def test_qdrant_search_default_config(self, mock_qdrant_module):
        """Test QdrantSearch with default configuration."""
        from bnsnlp.search.qdrant import QdrantSearch

        search = QdrantSearch({})

        assert search.url == "http://localhost:6333"
        assert search.collection == "bnsnlp"
        assert search.timeout == 30

    def test_qdrant_search_with_api_key(self, mock_qdrant_module):
        """Test QdrantSearch with API key."""
        from bnsnlp.search.qdrant import QdrantSearch

        config = {
            "url": "https://cloud.qdrant.io",
            "collection": "test",
            "api_key": "test-api-key",
        }

        search = QdrantSearch(config)

        assert search.api_key == "test-api-key"

    @pytest.mark.asyncio
    async def test_index_documents(self, qdrant_config, mock_qdrant_module):
        """Test indexing documents."""
        from bnsnlp.search.qdrant import QdrantSearch

        search = QdrantSearch(qdrant_config)

        texts = ["Merhaba dünya", "Test document"]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        ids = ["doc_1", "doc_2"]
        metadata = [{"source": "test1.txt"}, {"source": "test2.txt"}]

        await search.index(texts, embeddings, ids, metadata)

        # Verify collection was created
        mock_qdrant_module.get_collections.assert_called_once()
        mock_qdrant_module.create_collection.assert_called_once()

        # Verify upsert was called
        mock_qdrant_module.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_index_without_metadata(self, qdrant_config, mock_qdrant_module):
        """Test indexing documents without metadata."""
        from bnsnlp.search.qdrant import QdrantSearch

        search = QdrantSearch(qdrant_config)

        texts = ["Text 1", "Text 2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        ids = ["doc_1", "doc_2"]

        await search.index(texts, embeddings, ids)

        # Verify upsert was called
        mock_qdrant_module.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_index_empty_lists_error(self, qdrant_config, mock_qdrant_module):
        """Test that indexing empty lists raises error."""
        from bnsnlp.search.qdrant import QdrantSearch

        search = QdrantSearch(qdrant_config)

        with pytest.raises(AdapterError) as exc_info:
            await search.index([], [], [])

        assert "must be non-empty lists" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_index_mismatched_lengths_error(self, qdrant_config, mock_qdrant_module):
        """Test that mismatched list lengths raise error."""
        from bnsnlp.search.qdrant import QdrantSearch

        search = QdrantSearch(qdrant_config)

        texts = ["Text 1", "Text 2"]
        embeddings = [[0.1, 0.2]]  # Only one embedding
        ids = ["doc_1", "doc_2"]

        with pytest.raises(AdapterError) as exc_info:
            await search.index(texts, embeddings, ids)

        assert "must have the same length" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_index_mismatched_metadata_length_error(
        self, qdrant_config, mock_qdrant_module
    ):
        """Test that mismatched metadata length raises error."""
        from bnsnlp.search.qdrant import QdrantSearch

        search = QdrantSearch(qdrant_config)

        texts = ["Text 1", "Text 2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        ids = ["doc_1", "doc_2"]
        metadata = [{"source": "test.txt"}]  # Only one metadata

        with pytest.raises(AdapterError) as exc_info:
            await search.index(texts, embeddings, ids, metadata)

        assert "metadata must have the same length" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_index_with_retry(self, qdrant_config, mock_qdrant_module):
        """Test indexing with retry logic."""
        from bnsnlp.search.qdrant import QdrantSearch

        # Setup mock to fail twice then succeed
        call_count = [0]

        async def mock_upsert(**kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("Connection error")

        mock_qdrant_module.upsert = mock_upsert

        search = QdrantSearch(qdrant_config)

        texts = ["Text 1"]
        embeddings = [[0.1, 0.2, 0.3]]
        ids = ["doc_1"]

        await search.index(texts, embeddings, ids)

        # Should have retried and succeeded
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_index_max_retries_exceeded(self, qdrant_config, mock_qdrant_module):
        """Test that max retries are respected during indexing."""
        from bnsnlp.search.qdrant import QdrantSearch

        # Setup mock to always fail
        async def mock_upsert(**kwargs):
            raise Exception("Connection error")

        mock_qdrant_module.upsert = mock_upsert

        search = QdrantSearch(qdrant_config)

        texts = ["Text 1"]
        embeddings = [[0.1, 0.2, 0.3]]
        ids = ["doc_1"]

        with pytest.raises(AdapterError) as exc_info:
            await search.index(texts, embeddings, ids)

        assert "Failed to index documents after 3 attempts" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_documents(self, qdrant_config, mock_qdrant_module):
        """Test searching for documents."""
        from bnsnlp.search.qdrant import QdrantSearch

        search = QdrantSearch(qdrant_config)

        query_embedding = [0.1, 0.2, 0.3]
        response = await search.search(query_embedding, top_k=5)

        assert isinstance(response, SearchResponse)
        assert len(response.results) == 1
        assert response.results[0].id == "doc_123"
        assert response.results[0].score == 0.95
        assert response.results[0].text == "Merhaba dünya"
        assert response.results[0].metadata["source"] == "test.txt"
        assert response.query_time_ms > 0
        assert response.metadata["collection"] == "test_collection"
        assert response.metadata["top_k"] == 5

        # Verify search was called correctly
        mock_qdrant_module.search.assert_called_once()
        call_args = mock_qdrant_module.search.call_args
        assert call_args.kwargs["collection_name"] == "test_collection"
        assert call_args.kwargs["query_vector"] == query_embedding
        assert call_args.kwargs["limit"] == 5

    @pytest.mark.asyncio
    async def test_search_with_filters(self, qdrant_config, mock_qdrant_module):
        """Test searching with filters."""
        from bnsnlp.search.qdrant import QdrantSearch

        search = QdrantSearch(qdrant_config)

        query_embedding = [0.1, 0.2, 0.3]
        filters = {"must": [{"key": "source", "match": {"value": "test.txt"}}]}

        response = await search.search(query_embedding, top_k=10, filters=filters)

        assert isinstance(response, SearchResponse)

        # Verify search was called with filters
        mock_qdrant_module.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_empty_embedding_error(self, qdrant_config, mock_qdrant_module):
        """Test that searching with empty embedding raises error."""
        from bnsnlp.search.qdrant import QdrantSearch

        search = QdrantSearch(qdrant_config)

        with pytest.raises(AdapterError) as exc_info:
            await search.search([])

        assert "must be a non-empty list" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_invalid_top_k_error(self, qdrant_config, mock_qdrant_module):
        """Test that invalid top_k raises error."""
        from bnsnlp.search.qdrant import QdrantSearch

        search = QdrantSearch(qdrant_config)

        query_embedding = [0.1, 0.2, 0.3]

        with pytest.raises(AdapterError) as exc_info:
            await search.search(query_embedding, top_k=0)

        assert "must be a positive integer" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_with_retry(self, qdrant_config, mock_qdrant_module):
        """Test search with retry logic."""
        from bnsnlp.search.qdrant import QdrantSearch

        # Setup mock to fail twice then succeed
        call_count = [0]

        async def mock_search(**kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("Connection error")

            mock_result = MagicMock()
            mock_result.id = "doc_1"
            mock_result.score = 0.9
            mock_result.payload = {"text": "Test"}
            return [mock_result]

        mock_qdrant_module.search = mock_search

        search = QdrantSearch(qdrant_config)

        query_embedding = [0.1, 0.2, 0.3]
        response = await search.search(query_embedding)

        # Should have retried and succeeded
        assert call_count[0] == 3
        assert len(response.results) == 1

    @pytest.mark.asyncio
    async def test_search_max_retries_exceeded(self, qdrant_config, mock_qdrant_module):
        """Test that max retries are respected during search."""
        from bnsnlp.search.qdrant import QdrantSearch

        # Setup mock to always fail
        async def mock_search(**kwargs):
            raise Exception("Connection error")

        mock_qdrant_module.search = mock_search

        search = QdrantSearch(qdrant_config)

        query_embedding = [0.1, 0.2, 0.3]

        with pytest.raises(AdapterError) as exc_info:
            await search.search(query_embedding)

        assert "Failed to search after 3 attempts" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_invalid_filter_error(self, qdrant_config, mock_qdrant_module):
        """Test that invalid filter format raises error."""
        from bnsnlp.search.qdrant import QdrantSearch

        # Setup mock Filter to raise error
        def mock_filter(**kwargs):
            raise Exception("Invalid filter format")

        with patch("bnsnlp.search.qdrant.Filter", side_effect=mock_filter):
            search = QdrantSearch(qdrant_config)

            query_embedding = [0.1, 0.2, 0.3]
            filters = {"invalid": "filter"}

            with pytest.raises(AdapterError) as exc_info:
                await search.search(query_embedding, filters=filters)

            assert "Invalid filter format" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_empty_results(self, qdrant_config, mock_qdrant_module):
        """Test search with no results."""
        from bnsnlp.search.qdrant import QdrantSearch

        # Setup mock to return empty results
        mock_qdrant_module.search = AsyncMock(return_value=[])

        search = QdrantSearch(qdrant_config)

        query_embedding = [0.1, 0.2, 0.3]
        response = await search.search(query_embedding)

        assert isinstance(response, SearchResponse)
        assert len(response.results) == 0
        assert response.metadata["num_results"] == 0

    @pytest.mark.asyncio
    async def test_ensure_collection_creates_if_not_exists(
        self, qdrant_config, mock_qdrant_module
    ):
        """Test that collection is created if it doesn't exist."""
        from bnsnlp.search.qdrant import QdrantSearch

        search = QdrantSearch(qdrant_config)

        # Index should trigger collection creation
        texts = ["Text 1"]
        embeddings = [[0.1, 0.2, 0.3]]
        ids = ["doc_1"]

        await search.index(texts, embeddings, ids)

        # Verify collection was checked and created
        mock_qdrant_module.get_collections.assert_called_once()
        mock_qdrant_module.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_collection_skips_if_exists(
        self, qdrant_config, mock_qdrant_module
    ):
        """Test that collection creation is skipped if it exists."""
        from bnsnlp.search.qdrant import QdrantSearch

        # Setup mock to return existing collection
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collections = MagicMock()
        mock_collections.collections = [mock_collection]
        mock_qdrant_module.get_collections = AsyncMock(return_value=mock_collections)

        search = QdrantSearch(qdrant_config)

        texts = ["Text 1"]
        embeddings = [[0.1, 0.2, 0.3]]
        ids = ["doc_1"]

        await search.index(texts, embeddings, ids)

        # Verify collection was checked but not created
        mock_qdrant_module.get_collections.assert_called_once()
        mock_qdrant_module.create_collection.assert_not_called()


class TestPineconeSearch:
    """Tests for PineconeSearch adapter."""

    @pytest.fixture
    def mock_pinecone_module(self):
        """Mock the pinecone module."""
        # Create mock objects
        mock_index = MagicMock()
        mock_index.upsert = MagicMock(return_value=None)

        # Mock query response
        mock_match = MagicMock()
        mock_match.id = "doc_123"
        mock_match.score = 0.95
        mock_match.metadata = {"text": "Merhaba dünya", "source": "test.txt"}

        mock_query_response = MagicMock()
        mock_query_response.matches = [mock_match]
        mock_index.query = MagicMock(return_value=mock_query_response)

        # Mock client
        mock_client = MagicMock()
        
        # Mock list_indexes
        mock_index_info = MagicMock()
        mock_index_info.name = "existing_index"
        mock_client.list_indexes = MagicMock(return_value=[])
        mock_client.create_index = MagicMock(return_value=None)
        mock_client.Index = MagicMock(return_value=mock_index)

        mock_client_class = MagicMock(return_value=mock_client)

        # Mock ServerlessSpec
        mock_serverless_spec_class = MagicMock()
        mock_serverless_spec_instance = MagicMock()
        mock_serverless_spec_class.return_value = mock_serverless_spec_instance

        # Create mock pinecone module
        mock_pinecone = MagicMock()
        mock_pinecone.Pinecone = mock_client_class
        mock_pinecone.ServerlessSpec = mock_serverless_spec_class

        with patch.dict(
            "sys.modules",
            {"pinecone": mock_pinecone},
        ):
            # Reload the module to pick up the mocked imports
            import importlib
            import bnsnlp.search.pinecone as pinecone_module

            importlib.reload(pinecone_module)

            yield mock_client, mock_index

    @pytest.fixture
    def pinecone_config(self):
        """Create a test configuration for Pinecone search."""
        return {
            "api_key": "test-api-key",
            "index_name": "test_index",
            "environment": "us-east-1-aws",
            "dimension": 3,
        }

    def test_pinecone_search_initialization(self, pinecone_config, mock_pinecone_module):
        """Test PineconeSearch initialization."""
        from bnsnlp.search.pinecone import PineconeSearch

        search = PineconeSearch(pinecone_config)

        assert search.api_key == "test-api-key"
        assert search.index_name == "test_index"
        assert search.environment == "us-east-1-aws"
        assert search.dimension == 3

    def test_pinecone_search_missing_api_key(self, mock_pinecone_module):
        """Test that missing API key raises error."""
        from bnsnlp.search.pinecone import PineconeSearch

        with pytest.raises(AdapterError) as exc_info:
            PineconeSearch({})

        assert "API key is required" in str(exc_info.value)

    def test_pinecone_search_default_config(self, mock_pinecone_module):
        """Test PineconeSearch with default configuration."""
        from bnsnlp.search.pinecone import PineconeSearch

        config = {"api_key": "test-key"}
        search = PineconeSearch(config)

        assert search.index_name == "bnsnlp"
        assert search.environment == "us-east-1-aws"
        assert search.metric == "cosine"

    @pytest.mark.asyncio
    async def test_index_documents(self, pinecone_config, mock_pinecone_module):
        """Test indexing documents."""
        from bnsnlp.search.pinecone import PineconeSearch

        mock_client, mock_index = mock_pinecone_module
        search = PineconeSearch(pinecone_config)

        texts = ["Merhaba dünya", "Test document"]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        ids = ["doc_1", "doc_2"]
        metadata = [{"source": "test1.txt"}, {"source": "test2.txt"}]

        await search.index(texts, embeddings, ids, metadata)

        # Verify index was retrieved
        mock_client.Index.assert_called()

        # Verify upsert was called
        mock_index.upsert.assert_called()

    @pytest.mark.asyncio
    async def test_index_without_metadata(self, pinecone_config, mock_pinecone_module):
        """Test indexing documents without metadata."""
        from bnsnlp.search.pinecone import PineconeSearch

        mock_client, mock_index = mock_pinecone_module
        search = PineconeSearch(pinecone_config)

        texts = ["Text 1", "Text 2"]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        ids = ["doc_1", "doc_2"]

        await search.index(texts, embeddings, ids)

        # Verify upsert was called
        mock_index.upsert.assert_called()

    @pytest.mark.asyncio
    async def test_index_empty_lists_error(self, pinecone_config, mock_pinecone_module):
        """Test that indexing empty lists raises error."""
        from bnsnlp.search.pinecone import PineconeSearch

        search = PineconeSearch(pinecone_config)

        with pytest.raises(AdapterError) as exc_info:
            await search.index([], [], [])

        assert "must be non-empty lists" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_index_mismatched_lengths_error(
        self, pinecone_config, mock_pinecone_module
    ):
        """Test that mismatched list lengths raise error."""
        from bnsnlp.search.pinecone import PineconeSearch

        search = PineconeSearch(pinecone_config)

        texts = ["Text 1", "Text 2"]
        embeddings = [[0.1, 0.2, 0.3]]  # Only one embedding
        ids = ["doc_1", "doc_2"]

        with pytest.raises(AdapterError) as exc_info:
            await search.index(texts, embeddings, ids)

        assert "must have the same length" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_documents(self, pinecone_config, mock_pinecone_module):
        """Test searching for documents."""
        from bnsnlp.search.pinecone import PineconeSearch

        mock_client, mock_index = mock_pinecone_module
        search = PineconeSearch(pinecone_config)

        query_embedding = [0.1, 0.2, 0.3]
        response = await search.search(query_embedding, top_k=5)

        assert isinstance(response, SearchResponse)
        assert len(response.results) == 1
        assert response.results[0].id == "doc_123"
        assert response.results[0].score == 0.95
        assert response.results[0].text == "Merhaba dünya"
        assert response.results[0].metadata["source"] == "test.txt"
        assert response.query_time_ms > 0

        # Verify query was called correctly
        mock_index.query.assert_called_once()
        call_kwargs = mock_index.query.call_args.kwargs
        assert call_kwargs["vector"] == query_embedding
        assert call_kwargs["top_k"] == 5
        assert call_kwargs["include_metadata"] is True

    @pytest.mark.asyncio
    async def test_search_with_filters(self, pinecone_config, mock_pinecone_module):
        """Test searching with filters."""
        from bnsnlp.search.pinecone import PineconeSearch

        mock_client, mock_index = mock_pinecone_module
        search = PineconeSearch(pinecone_config)

        query_embedding = [0.1, 0.2, 0.3]
        filters = {"source": "test.txt"}

        response = await search.search(query_embedding, top_k=10, filters=filters)

        assert isinstance(response, SearchResponse)

        # Verify query was called with filters
        call_kwargs = mock_index.query.call_args.kwargs
        assert call_kwargs["filter"] == filters

    @pytest.mark.asyncio
    async def test_search_empty_embedding_error(
        self, pinecone_config, mock_pinecone_module
    ):
        """Test that searching with empty embedding raises error."""
        from bnsnlp.search.pinecone import PineconeSearch

        search = PineconeSearch(pinecone_config)

        with pytest.raises(AdapterError) as exc_info:
            await search.search([])

        assert "must be a non-empty list" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_invalid_top_k_error(
        self, pinecone_config, mock_pinecone_module
    ):
        """Test that invalid top_k raises error."""
        from bnsnlp.search.pinecone import PineconeSearch

        search = PineconeSearch(pinecone_config)

        query_embedding = [0.1, 0.2, 0.3]

        with pytest.raises(AdapterError) as exc_info:
            await search.search(query_embedding, top_k=0)

        assert "must be a positive integer" in str(exc_info.value)


class TestFAISSSearch:
    """Tests for FAISSSearch adapter."""

    @pytest.fixture
    def mock_faiss_module(self, tmp_path):
        """Mock the faiss module."""
        # Create mock FAISS index
        mock_index = MagicMock()
        mock_index.ntotal = 0
        mock_index.add = MagicMock()

        # Mock search results
        mock_index.search = MagicMock(
            return_value=(
                [[0.95, 0.85]],  # distances
                [[0, 1]],  # indices
            )
        )

        # Mock FAISS functions
        mock_faiss = MagicMock()
        mock_faiss.IndexFlatIP = MagicMock(return_value=mock_index)
        mock_faiss.IndexFlatL2 = MagicMock(return_value=mock_index)
        mock_faiss.get_num_gpus = MagicMock(return_value=0)
        mock_faiss.write_index = MagicMock()
        mock_faiss.read_index = MagicMock(return_value=mock_index)

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            # Reload the module to pick up the mocked imports
            import importlib
            import bnsnlp.search.faiss as faiss_module

            importlib.reload(faiss_module)

            yield mock_faiss, mock_index

    @pytest.fixture
    def faiss_config(self, tmp_path):
        """Create a test configuration for FAISS search."""
        return {
            "index_path": str(tmp_path / "test_faiss.index"),
            "dimension": 3,
            "metric": "cosine",
        }

    def test_faiss_search_initialization(self, faiss_config, mock_faiss_module):
        """Test FAISSSearch initialization."""
        from bnsnlp.search.faiss import FAISSSearch

        search = FAISSSearch(faiss_config)

        assert search.index_path == faiss_config["index_path"]
        assert search.dimension == 3
        assert search.metric == "cosine"

    def test_faiss_search_default_config(self, mock_faiss_module):
        """Test FAISSSearch with default configuration."""
        from bnsnlp.search.faiss import FAISSSearch

        search = FAISSSearch({})

        assert search.index_path == "faiss.index"
        assert search.metric == "cosine"
        assert search.use_gpu is False

    @pytest.mark.asyncio
    async def test_index_documents(self, faiss_config, mock_faiss_module):
        """Test indexing documents."""
        from bnsnlp.search.faiss import FAISSSearch

        mock_faiss, mock_index = mock_faiss_module
        search = FAISSSearch(faiss_config)

        texts = ["Merhaba dünya", "Test document"]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        ids = ["doc_1", "doc_2"]
        metadata = [{"source": "test1.txt"}, {"source": "test2.txt"}]

        await search.index(texts, embeddings, ids, metadata)

        # Verify index was created
        mock_faiss.IndexFlatIP.assert_called_once_with(3)

        # Verify vectors were added
        mock_index.add.assert_called_once()

        # Verify documents were stored
        assert len(search.documents) == 2

    @pytest.mark.asyncio
    async def test_index_without_metadata(self, faiss_config, mock_faiss_module):
        """Test indexing documents without metadata."""
        from bnsnlp.search.faiss import FAISSSearch

        mock_faiss, mock_index = mock_faiss_module
        search = FAISSSearch(faiss_config)

        texts = ["Text 1", "Text 2"]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        ids = ["doc_1", "doc_2"]

        await search.index(texts, embeddings, ids)

        # Verify vectors were added
        mock_index.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_index_empty_lists_error(self, faiss_config, mock_faiss_module):
        """Test that indexing empty lists raises error."""
        from bnsnlp.search.faiss import FAISSSearch

        search = FAISSSearch(faiss_config)

        with pytest.raises(AdapterError) as exc_info:
            await search.index([], [], [])

        assert "must be non-empty lists" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_index_mismatched_lengths_error(self, faiss_config, mock_faiss_module):
        """Test that mismatched list lengths raise error."""
        from bnsnlp.search.faiss import FAISSSearch

        search = FAISSSearch(faiss_config)

        texts = ["Text 1", "Text 2"]
        embeddings = [[0.1, 0.2, 0.3]]  # Only one embedding
        ids = ["doc_1", "doc_2"]

        with pytest.raises(AdapterError) as exc_info:
            await search.index(texts, embeddings, ids)

        assert "must have the same length" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_documents(self, faiss_config, mock_faiss_module):
        """Test searching for documents."""
        from bnsnlp.search.faiss import FAISSSearch

        mock_faiss, mock_index = mock_faiss_module
        search = FAISSSearch(faiss_config)

        # First index some documents
        texts = ["Merhaba dünya", "Test document"]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        ids = ["doc_1", "doc_2"]
        metadata = [{"source": "test1.txt"}, {"source": "test2.txt"}]

        await search.index(texts, embeddings, ids, metadata)

        # Update mock to have documents
        mock_index.ntotal = 2

        # Now search
        query_embedding = [0.1, 0.2, 0.3]
        response = await search.search(query_embedding, top_k=2)

        assert isinstance(response, SearchResponse)
        assert len(response.results) == 2
        assert response.results[0].id == "doc_1"
        assert response.results[0].text == "Merhaba dünya"
        assert response.query_time_ms > 0

        # Verify search was called
        mock_index.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_filters(self, faiss_config, mock_faiss_module):
        """Test searching with metadata filters."""
        from bnsnlp.search.faiss import FAISSSearch

        mock_faiss, mock_index = mock_faiss_module
        search = FAISSSearch(faiss_config)

        # Index documents
        texts = ["Text 1", "Text 2"]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        ids = ["doc_1", "doc_2"]
        metadata = [{"source": "test1.txt"}, {"source": "test2.txt"}]

        await search.index(texts, embeddings, ids, metadata)
        mock_index.ntotal = 2

        # Search with filter
        query_embedding = [0.1, 0.2, 0.3]
        filters = {"source": "test1.txt"}

        response = await search.search(query_embedding, top_k=10, filters=filters)

        # Should only return documents matching the filter
        assert len(response.results) == 1
        assert response.results[0].metadata["source"] == "test1.txt"

    @pytest.mark.asyncio
    async def test_search_empty_index(self, faiss_config, mock_faiss_module):
        """Test searching in empty index."""
        from bnsnlp.search.faiss import FAISSSearch

        search = FAISSSearch(faiss_config)

        query_embedding = [0.1, 0.2, 0.3]
        response = await search.search(query_embedding)

        assert isinstance(response, SearchResponse)
        assert len(response.results) == 0
        assert response.metadata["num_results"] == 0

    @pytest.mark.asyncio
    async def test_search_empty_embedding_error(self, faiss_config, mock_faiss_module):
        """Test that searching with empty embedding raises error."""
        from bnsnlp.search.faiss import FAISSSearch

        search = FAISSSearch(faiss_config)

        with pytest.raises(AdapterError) as exc_info:
            await search.search([])

        assert "must be a non-empty list" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_invalid_top_k_error(self, faiss_config, mock_faiss_module):
        """Test that invalid top_k raises error."""
        from bnsnlp.search.faiss import FAISSSearch

        search = FAISSSearch(faiss_config)

        query_embedding = [0.1, 0.2, 0.3]

        with pytest.raises(AdapterError) as exc_info:
            await search.search(query_embedding, top_k=0)

        assert "must be a positive integer" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_index_with_l2_metric(self, tmp_path, mock_faiss_module):
        """Test indexing with L2 distance metric."""
        from bnsnlp.search.faiss import FAISSSearch

        mock_faiss, mock_index = mock_faiss_module
        config = {
            "index_path": str(tmp_path / "test_l2.index"),
            "dimension": 3,
            "metric": "l2",
        }

        search = FAISSSearch(config)

        texts = ["Text 1"]
        embeddings = [[0.1, 0.2, 0.3]]
        ids = ["doc_1"]

        await search.index(texts, embeddings, ids)

        # Verify L2 index was created
        mock_faiss.IndexFlatL2.assert_called_once_with(3)

    @pytest.mark.asyncio
    async def test_unsupported_metric_error(self, tmp_path, mock_faiss_module):
        """Test that unsupported metric raises error."""
        from bnsnlp.search.faiss import FAISSSearch

        config = {
            "index_path": str(tmp_path / "test.index"),
            "dimension": 3,
            "metric": "invalid",
        }

        search = FAISSSearch(config)

        texts = ["Text 1"]
        embeddings = [[0.1, 0.2, 0.3]]
        ids = ["doc_1"]

        with pytest.raises(AdapterError) as exc_info:
            await search.index(texts, embeddings, ids)

        assert "Unsupported metric" in str(exc_info.value)
