"""
FastAPI Service Example for bns-nlp-engine

This example demonstrates how to deploy bns-nlp-engine as a REST API service.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import logging
from contextlib import asynccontextmanager

# bns-nlp-engine imports
from bnsnlp import Pipeline, Config
from bnsnlp.core.registry import PluginRegistry
from bnsnlp.preprocess import TurkishPreprocessor
from bnsnlp.embed import OpenAIEmbedder, HuggingFaceEmbedder
from bnsnlp.search import FAISSSearch
from bnsnlp.classify import TurkishClassifier
from bnsnlp.core.exceptions import BNSNLPError

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app"""
    # Startup
    logger.info("Initializing bns-nlp-engine...")
    
    # Initialize registry
    registry = PluginRegistry()
    registry.discover_plugins()
    app_state['registry'] = registry
    
    # Initialize config
    config = Config()
    app_state['config'] = config
    
    # Initialize components
    app_state['preprocessor'] = TurkishPreprocessor({
        'lowercase': True,
        'remove_punctuation': True,
        'remove_stopwords': True,
        'lemmatize': True
    })
    
    logger.info("bns-nlp-engine initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down bns-nlp-engine...")
    app_state.clear()


# Create FastAPI app
app = FastAPI(
    title="bns-nlp-engine API",
    description="Turkish NLP Engine REST API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class PreprocessRequest(BaseModel):
    """Preprocessing request"""
    text: str = Field(..., description="Text to preprocess")
    lowercase: bool = Field(True, description="Convert to lowercase")
    remove_punctuation: bool = Field(True, description="Remove punctuation")
    remove_stopwords: bool = Field(True, description="Remove stop words")
    lemmatize: bool = Field(True, description="Apply lemmatization")


class PreprocessResponse(BaseModel):
    """Preprocessing response"""
    text: str = Field(..., description="Processed text")
    tokens: List[str] = Field(..., description="Token list")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class EmbedRequest(BaseModel):
    """Embedding request"""
    texts: List[str] = Field(..., description="Texts to embed")
    provider: str = Field("openai", description="Embedding provider (openai, cohere, huggingface)")
    model: Optional[str] = Field(None, description="Model name")
    batch_size: int = Field(16, description="Batch size")


class EmbedResponse(BaseModel):
    """Embedding response"""
    embeddings: List[List[float]] = Field(..., description="Embedding vectors")
    model: str = Field(..., description="Model used")
    dimensions: int = Field(..., description="Embedding dimensions")


class SearchRequest(BaseModel):
    """Search request"""
    query: str = Field(..., description="Search query")
    top_k: int = Field(10, description="Number of results")
    provider: str = Field("faiss", description="Search provider")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")


class SearchResult(BaseModel):
    """Single search result"""
    id: str
    score: float
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Search response"""
    results: List[SearchResult]
    query_time_ms: float


class ClassifyRequest(BaseModel):
    """Classification request"""
    text: str = Field(..., description="Text to classify")


class Entity(BaseModel):
    """Entity model"""
    text: str
    type: str
    start: int
    end: int
    confidence: float


class ClassifyResponse(BaseModel):
    """Classification response"""
    intent: str
    intent_confidence: float
    entities: List[Entity]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    components: Dict[str, str]


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    code: str
    details: Optional[Dict[str, Any]] = None


# Exception handler
@app.exception_handler(BNSNLPError)
async def bnsnlp_exception_handler(request, exc: BNSNLPError):
    """Handle bns-nlp-engine exceptions"""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=exc.message,
            code=exc.code,
            details=exc.context
        ).dict()
    )


# Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "bns-nlp-engine API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        components={
            "preprocessor": "ready",
            "registry": "ready"
        }
    )


@app.post("/preprocess", response_model=PreprocessResponse, tags=["Preprocessing"])
async def preprocess_text(request: PreprocessRequest):
    """
    Preprocess Turkish text
    
    - **text**: Text to preprocess
    - **lowercase**: Convert to lowercase
    - **remove_punctuation**: Remove punctuation marks
    - **remove_stopwords**: Remove Turkish stop words
    - **lemmatize**: Apply lemmatization
    """
    try:
        # Create preprocessor with custom config
        preprocessor = TurkishPreprocessor({
            'lowercase': request.lowercase,
            'remove_punctuation': request.remove_punctuation,
            'remove_stopwords': request.remove_stopwords,
            'lemmatize': request.lemmatize
        })
        
        # Process
        result = await preprocessor.process(request.text)
        
        return PreprocessResponse(
            text=result.text,
            tokens=result.tokens,
            metadata=result.metadata
        )
    
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preprocess/batch", response_model=List[PreprocessResponse], tags=["Preprocessing"])
async def preprocess_batch(texts: List[str], background_tasks: BackgroundTasks):
    """
    Preprocess multiple texts in batch
    
    - **texts**: List of texts to preprocess
    """
    try:
        preprocessor = app_state['preprocessor']
        
        # Process batch
        results = await preprocessor.process(texts)
        
        return [
            PreprocessResponse(
                text=r.text,
                tokens=r.tokens,
                metadata=r.metadata
            )
            for r in results
        ]
    
    except Exception as e:
        logger.error(f"Batch preprocessing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed", response_model=EmbedResponse, tags=["Embedding"])
async def create_embeddings(request: EmbedRequest):
    """
    Generate embeddings for texts
    
    - **texts**: List of texts to embed
    - **provider**: Embedding provider (openai, cohere, huggingface)
    - **model**: Model name (optional)
    - **batch_size**: Batch size for processing
    """
    try:
        # Create embedder based on provider
        if request.provider == "openai":
            import os
            embedder = OpenAIEmbedder({
                'api_key': os.getenv('OPENAI_API_KEY'),
                'model': request.model or 'text-embedding-3-small',
                'batch_size': request.batch_size
            })
        elif request.provider == "huggingface":
            embedder = HuggingFaceEmbedder({
                'model': request.model or 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                'use_gpu': False,
                'batch_size': request.batch_size
            })
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported provider: {request.provider}"
            )
        
        # Generate embeddings
        result = await embedder.embed(request.texts)
        
        return EmbedResponse(
            embeddings=result.embeddings,
            model=result.model,
            dimensions=result.dimensions
        )
    
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def semantic_search(request: SearchRequest):
    """
    Perform semantic search
    
    - **query**: Search query
    - **top_k**: Number of results to return
    - **provider**: Search provider (faiss, qdrant, pinecone)
    - **filters**: Optional metadata filters
    """
    try:
        # This is a simplified example
        # In production, you would maintain a persistent search index
        raise HTTPException(
            status_code=501,
            detail="Search endpoint requires index setup. See documentation."
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify", response_model=ClassifyResponse, tags=["Classification"])
async def classify_text(request: ClassifyRequest):
    """
    Classify text for intent and entities
    
    - **text**: Text to classify
    """
    try:
        # This is a simplified example
        # In production, you would load actual classification models
        raise HTTPException(
            status_code=501,
            detail="Classification endpoint requires model setup. See documentation."
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline", tags=["Pipeline"])
async def run_pipeline(
    text: str,
    steps: List[str] = ["preprocess", "embed"]
):
    """
    Run a custom pipeline
    
    - **text**: Text to process
    - **steps**: List of pipeline steps
    """
    try:
        registry = app_state['registry']
        config = app_state['config']
        
        # Create pipeline
        pipeline = Pipeline(config, registry)
        
        # Add steps
        for step in steps:
            if step == "preprocess":
                pipeline.add_step('preprocess', 'turkish')
            elif step == "embed":
                pipeline.add_step('embed', 'huggingface')
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown step: {step}"
                )
        
        # Process
        result = await pipeline.process(text)
        
        return {"result": str(result)}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
