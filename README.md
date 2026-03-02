# Social Media Knowledge Extraction Tool

A personal tool to automatically process Instagram and Threads posts, extract content using OCR and audio transcription, summarize with LLMs, and store in a searchable knowledge base.

## Features

- 📥 **Content Fetching**: Automatically fetch posts, reels, and threads from Instagram and Threads
- 🖼️ **OCR**: Extract text from images (supports Hindi, English, and Hinglish)
- 🎙️ **Transcription**: Transcribe audio/video content using Whisper
- 🧠 **LLM Summarization**: Summarize and structure content using Gemini Pro (with Ollama fallback)
- 🔍 **Semantic Search**: Search your knowledge base using natural language
- 💾 **Dual Storage**: PostgreSQL for structured data, Qdrant for vector search

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- FFmpeg (for video processing)
- (Optional) Ollama for local LLM fallback

### 1. Clone and Setup

```bash
cd video-summary

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers (for Threads)
playwright install chromium
```

### 2. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit .env with your settings
nano .env
```

**Required settings:**
- `GEMINI_API_KEY`: Your Google Gemini API key

**Optional settings:**
- `INSTAGRAM_USERNAME` / `INSTAGRAM_PASSWORD`: For private posts
- `OLLAMA_MODEL`: Local LLM model name (default: `llama3:8b`)

### 3. Start Services

```bash
# Start PostgreSQL and Qdrant
docker-compose up -d

# Verify services are running
docker-compose ps
```

### 4. Run the Application

```bash
# Start the API server
python -m src.main

# Or with uvicorn directly
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Access the Interface

- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

## Usage

### Process a Post

```bash
curl -X POST http://localhost:8000/api/process \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.instagram.com/p/ABC123/"}'
```

### Search Knowledge Base

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "productivity tips", "limit": 10}'
```

### List All Posts

```bash
curl http://localhost:8000/api/posts?limit=20
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/process` | Submit a URL for processing |
| GET | `/api/posts` | List all processed posts |
| GET | `/api/posts/{id}` | Get specific post details |
| DELETE | `/api/posts/{id}` | Delete a post |
| POST | `/api/search` | Semantic search |
| GET | `/api/stats` | Get statistics |
| GET | `/api/health` | Health check |

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Instagram/    │────▶│   Processors    │────▶│      LLM        │
│    Threads      │     │  (OCR/Whisper)  │     │ (Gemini/Ollama) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                                                         ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │   PostgreSQL    │◀────│   FastAPI       │
                        │  (Structured)   │     │    Backend      │
                        └─────────────────┘     └─────────────────┘
                                                         │
                        ┌─────────────────┐              │
                        │     Qdrant      │◀─────────────┘
                        │ (Vector Search) │
                        └─────────────────┘
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection URL | `postgresql+asyncpg://...` |
| `QDRANT_HOST` | Qdrant server host | `localhost` |
| `QDRANT_PORT` | Qdrant server port | `6333` |
| `GEMINI_API_KEY` | Google Gemini API key | (required) |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama model name | `llama3:8b` |
| `WHISPER_MODEL` | Whisper model size | `base` |
| `OCR_LANGUAGES` | OCR languages | `en,hi` |

### Whisper Model Sizes

| Model | VRAM | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | ~1GB | Fastest | Basic |
| base | ~1GB | Fast | Good |
| small | ~2GB | Medium | Better |
| medium | ~5GB | Slow | Great |
| large | ~10GB | Slowest | Best |

## Development

### Project Structure

```
video-summary/
├── src/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Configuration
│   ├── fetchers/            # Content fetching
│   ├── processors/          # OCR, transcription
│   ├── llm/                 # LLM integration
│   ├── storage/             # Database, vectors
│   └── api/                 # API routes
├── static/                  # Web interface
├── docker-compose.yml       # Services
└── requirements.txt         # Dependencies
```

### Running Tests

```bash
pytest tests/ -v
```

## Troubleshooting

### Instagram login issues
- Use app-specific password if 2FA is enabled
- Clear Instaloader cache: `rm -rf ~/.config/instaloader`

### Whisper memory issues
- Use a smaller model: `WHISPER_MODEL=tiny`

### Qdrant connection issues
- Check if container is running: `docker-compose ps`
- View logs: `docker-compose logs qdrant`

## License

MIT License - Use freely for personal projects.
