# Physical AI & Humanoid Robotics: RAG Chatbot Backend

This repository implements a Retrieval Augmented Generation (RAG) chatbot for the Physical AI & Humanoid Robotics textbook using FastAPI and Qdrant vector database.

## Architecture Overview

- **Backend**: FastAPI server with Python 3.9+
- **Vector Database**: Qdrant for efficient similarity search
- **Embeddings**: Sentence Transformers model (`all-MiniLM-L6-v2`)
- **Content Source**: Docusaurus documentation in the `docs/` directory

## Features

- Efficient semantic search in textbook content
- RESTful API for querying
- Context-aware responses based on textbook content
- Configurable number of relevant chunks to retrieve

## Prerequisites

- Python 3.9+
- Docker (for running Qdrant)
- Textbook content in the `docs/` directory (Markdown files)

## Setup Instructions

1. Clone the repository
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
3. Make sure Docker is running on your system

4. Start the services using the batch script:
   ```bash
   ./start_server.bat  # On Windows
   # or manually:
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant  # In a separate terminal
   python ingest_backend.py
   python backend.py
   ```

## API Endpoints

### GET /
Health check endpoint to verify the service is running.

### GET /api/health
Detailed health check.

### POST /api/query
Main query endpoint for interacting with the RAG system.

Request body:
```json
{
  "question": "Your question about the textbook content",
  "top_k": 3
}
```

Response:
```json
{
  "answer": "The answer based on textbook content",
  "sources": [
    {
      "text": "Relevant text snippet...",
      "source": "Source document",
      "relevance_score": "0.8765"
    }
  ],
  "question": "Your original question"
}
```

## Configuration

Environment variables can be set in a `.env` file:

```env
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=humanoid_robotics
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## How It Works

1. The ingestion script (`ingest_backend.py`) processes all Markdown files in the `docs/` directory
2. Content is split into manageable chunks while preserving context
3. Each chunk is converted to embeddings using the Sentence Transformer model
4. Embeddings are stored in Qdrant with associated metadata
5. When a query arrives, it's converted to an embedding
6. Qdrant performs similarity search to find the most relevant chunks
7. The query endpoint combines relevant content and returns a contextual answer

## Customization

- To use a different embedding model, change the `EMBEDDING_MODEL` environment variable
- Adjust chunk size in `ingest_backend.py` if needed
- Modify the answer generation logic in `backend.py` to use a different LLM

## Troubleshooting

- If Qdrant fails to start, ensure Docker is running and the ports aren't in use
- If ingestion fails, check that the `docs/` directory contains Markdown files
- If queries return no results, verify the collection was created and populated properly

## License

[Add your license information here]