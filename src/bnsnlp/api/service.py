"""
FastAPI service for bns-nlp-engine.

This module provides a REST API interface for all NLP operations including
preprocessing, embedding, search, and classification.
"""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from bnsnlp.__version__ import __version__


# Request Models
class PreprocessRequest(BaseModel):
    """Request model for text preprocessing."""

    text: str = Field(..., description="Text to preprocess")
    lowercase: bool = Field(default=True, description="Convert text to lowercase")
    remove_punctuation: bool = Field(default=True, description="Remove punctuation marks")
    remove_stopwords: bool = Field(default=True, description="Remove Turkish stop words")
    lemmatize: bool = Field(default=True, description="Apply lemmatization")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Merhaba DÜNYA! Nasılsın?",
                "lowercase": True,
                "remove_punctuation": True,
                "remove_stopwords": True,
                "lemmatize": True,
            }
        }


class EmbedRequest(BaseModel):
    """Request model for text embedding."""

    texts: List[str] = Field(..., description="List of texts to embed")
    provider: str = Field(default="openai", description="Embedding provider (openai, cohere, huggingface)")
    model: Optional[str] = Field(default=None, description="Model name (provider-specific)")

    class Config:
        json_schema_extra = {
            "example": {
                "texts": ["Merhaba dünya", "Türkçe NLP"],
                "provider": "openai",
                "model": "text-embedding-3-small",
            }
        }


class SearchRequest(BaseModel):
    """Request model for semantic search."""

    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=10, description="Number of results to return", gt=0)
    provider: str = Field(default="faiss", description="Search backend (faiss, qdrant, pinecone)")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata filters")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Türkçe doğal dil işleme",
                "top_k": 10,
                "provider": "faiss",
                "filters": {"category": "nlp"},
            }
        }


class ClassifyRequest(BaseModel):
    """Request model for text classification."""

    text: str = Field(..., description="Text to classify")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Yarın saat 14:00'te toplantı var mı?",
            }
        }

# Create FastAPI application
app = FastAPI(
    title="bns-nlp-engine API",
    description="Turkish NLP Engine REST API - Modular and extensible NLP service",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoints
@app.post("/preprocess", summary="Preprocess Turkish text")
async def preprocess_endpoint(request: PreprocessRequest):
    """
    Preprocess Turkish text with normalization, tokenization, and lemmatization.

    This endpoint applies various preprocessing steps to Turkish text including:
    - Lowercase conversion
    - Punctuation removal
    - Stop word removal
    - Lemmatization

    Args:
        request: PreprocessRequest containing text and preprocessing options

    Returns:
        PreprocessResult with processed text, tokens, and metadata

    Raises:
        HTTPException: If preprocessing fails
    """
    from fastapi import HTTPException

    from bnsnlp.preprocess.turkish import TurkishPreprocessor

    try:
        config = {
            "lowercase": request.lowercase,
            "remove_punctuation": request.remove_punctuation,
            "remove_stopwords": request.remove_stopwords,
            "lemmatize": request.lemmatize,
        }
        preprocessor = TurkishPreprocessor(config)
        result = await preprocessor.process(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")


@app.post("/embed", summary="Generate text embeddings")
async def embed_endpoint(request: EmbedRequest):
    """
    Generate embeddings for text using various providers.

    This endpoint supports multiple embedding providers:
    - OpenAI (text-embedding-3-small, text-embedding-3-large)
    - Cohere (embed-multilingual-v3.0)
    - HuggingFace (local models)

    Args:
        request: EmbedRequest containing texts and provider configuration

    Returns:
        EmbedResult with embeddings, model info, and metadata

    Raises:
        HTTPException: If embedding generation fails
    """
    from fastapi import HTTPException

    from bnsnlp.embed.openai import OpenAIEmbedder
    from bnsnlp.embed.cohere import CohereEmbedder
    from bnsnlp.embed.huggingface import HuggingFaceEmbedder

    try:
        # Select embedder based on provider
        config = {"model": request.model} if request.model else {}

        if request.provider == "openai":
            embedder = OpenAIEmbedder(config)
        elif request.provider == "cohere":
            embedder = CohereEmbedder(config)
        elif request.provider == "huggingface":
            embedder = HuggingFaceEmbedder(config)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid provider: {request.provider}. Must be one of: openai, cohere, huggingface",
            )

        result = await embedder.embed(request.texts)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")


@app.post("/search", summary="Semantic search")
async def search_endpoint(request: SearchRequest):
    """
    Perform semantic search using vector similarity.

    This endpoint:
    1. Embeds the query text using OpenAI (default)
    2. Searches the vector database for similar documents
    3. Returns ranked results with similarity scores

    Supported search backends:
    - FAISS (local index)
    - Qdrant (vector database)
    - Pinecone (cloud vector database)

    Args:
        request: SearchRequest containing query and search parameters

    Returns:
        SearchResponse with ranked results and query time

    Raises:
        HTTPException: If search operation fails
    """
    from fastapi import HTTPException

    from bnsnlp.embed.openai import OpenAIEmbedder
    from bnsnlp.search.faiss import FAISSSearch
    from bnsnlp.search.qdrant import QdrantSearch
    from bnsnlp.search.pinecone import PineconeSearch

    try:
        # First, embed the query
        embedder = OpenAIEmbedder({})
        embed_result = await embedder.embed(request.query)
        query_embedding = embed_result.embeddings[0]

        # Select search backend based on provider
        if request.provider == "faiss":
            searcher = FAISSSearch({})
        elif request.provider == "qdrant":
            searcher = QdrantSearch({})
        elif request.provider == "pinecone":
            searcher = PineconeSearch({})
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid provider: {request.provider}. Must be one of: faiss, qdrant, pinecone",
            )

        # Perform search
        results = await searcher.search(
            query_embedding=query_embedding,
            top_k=request.top_k,
            filters=request.filters,
        )
        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/classify", summary="Classify text and extract entities")
async def classify_endpoint(request: ClassifyRequest):
    """
    Classify intent and extract named entities from Turkish text.

    This endpoint performs:
    - Intent classification (e.g., question, command, statement)
    - Named entity recognition (e.g., PERSON, LOCATION, ORG, DATE)

    Args:
        request: ClassifyRequest containing text to classify

    Returns:
        ClassifyResult with intent, confidence, and extracted entities

    Raises:
        HTTPException: If classification fails
    """
    from fastapi import HTTPException

    from bnsnlp.classify.turkish import TurkishClassifier

    try:
        classifier = TurkishClassifier({})
        result = await classifier.classify(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.get("/health", summary="Health check")
async def health_check():
    """
    Health check endpoint to verify service status.

    Returns:
        Dictionary with service status and version information
    """
    return {
        "status": "healthy",
        "version": __version__,
        "service": "bns-nlp-engine",
    }
