# bns-nlp-engine FastAPI Service

REST API service for bns-nlp-engine.

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=your-api-key

# Run server
python main.py

# Or with uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Docker

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop
docker-compose down
```

## API Documentation

Once running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

### Preprocessing

```bash
curl -X POST http://localhost:8000/preprocess \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Merhaba DÜNYA! Bu bir test metnidir.",
    "lowercase": true,
    "remove_punctuation": true,
    "remove_stopwords": true,
    "lemmatize": true
  }'
```

### Batch Preprocessing

```bash
curl -X POST http://localhost:8000/preprocess/batch \
  -H "Content-Type: application/json" \
  -d '["Metin 1", "Metin 2", "Metin 3"]'
```

### Embedding

```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Merhaba dünya", "Python programlama"],
    "provider": "openai",
    "model": "text-embedding-3-small",
    "batch_size": 16
  }'
```

### Pipeline

```bash
curl -X POST "http://localhost:8000/pipeline?text=Merhaba%20dünya&steps=preprocess&steps=embed"
```

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
BNSNLP_LOG_LEVEL=INFO
BNSNLP_COHERE_API_KEY=...
BNSNLP_QDRANT_URL=http://qdrant:6333
```

### Config File

Edit `config.yaml` to customize:

```yaml
preprocess:
  lowercase: true
  remove_punctuation: true
  remove_stopwords: true
  lemmatize: true

embed:
  provider: openai
  model: text-embedding-3-small
  batch_size: 16
```

## Production Deployment

### Using Gunicorn

```bash
gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

### Using Docker

```bash
# Build
docker build -t bnsnlp-api .

# Run
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your-key \
  --name bnsnlp-api \
  bnsnlp-api
```

### Kubernetes

See `k8s/` directory for Kubernetes manifests.

## Monitoring

### Prometheus Metrics

Metrics available at `/metrics` endpoint (if enabled).

### Health Checks

```bash
# Liveness probe
curl http://localhost:8000/health

# Readiness probe
curl http://localhost:8000/health
```

## Performance

### Benchmarking

```bash
# Install Apache Bench
apt-get install apache2-utils

# Benchmark preprocessing
ab -n 1000 -c 10 -p request.json -T application/json \
  http://localhost:8000/preprocess
```

### Load Testing

```bash
# Install locust
pip install locust

# Run load test
locust -f locustfile.py --host http://localhost:8000
```

## Security

### API Key Authentication

Add authentication middleware:

```python
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != "your-secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials
```

### Rate Limiting

Use `slowapi` for rate limiting:

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/preprocess")
@limiter.limit("10/minute")
async def preprocess_text(request: Request, ...):
    ...
```

## Troubleshooting

### Common Issues

**Port already in use:**
```bash
# Find process
lsof -i :8000

# Kill process
kill -9 <PID>
```

**Module not found:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Docker build fails:**
```bash
# Clear cache
docker-compose build --no-cache
```

## Examples

### Python Client

```python
import requests

# Preprocess
response = requests.post(
    "http://localhost:8000/preprocess",
    json={
        "text": "Merhaba dünya",
        "lowercase": True
    }
)
print(response.json())

# Embed
response = requests.post(
    "http://localhost:8000/embed",
    json={
        "texts": ["Merhaba dünya"],
        "provider": "openai"
    }
)
print(response.json())
```

### JavaScript Client

```javascript
// Preprocess
const response = await fetch('http://localhost:8000/preprocess', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: 'Merhaba dünya',
    lowercase: true
  })
});
const data = await response.json();
console.log(data);
```

### cURL Examples

See [API Documentation](#api-documentation) section above.

## License

MIT License - see [LICENSE](../../LICENSE) file.

## Support

- [GitHub Issues](https://github.com/yourusername/bns-nlp-engine/issues)
- [Documentation](https://yourusername.github.io/bns-nlp-engine)
