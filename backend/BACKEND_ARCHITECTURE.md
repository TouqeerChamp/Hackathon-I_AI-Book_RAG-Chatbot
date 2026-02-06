# Physical AI & Humanoid Robotics: Integrated RAG Chatbot

This project implements an integrated RAG (Retrieval-Augmented Generation) chatbot for the Physical AI & Humanoid Robotics textbook, using a FastAPI backend with Qdrant vector database.

## Overview

This implementation provides:
- A FastAPI backend for handling queries
- Qdrant vector database for efficient similarity search
- Semantic search capabilities using embeddings
- Integration with Google's Gemini for advanced response generation
- Content sourced from the textbook documentation in the `docs/` directory

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │───▶│   FastAPI        │───▶│   Qdrant        │
│   (Docusaurus)  │    │   Backend        │    │   Vector DB     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                       ┌──────────────────┐
                       │  Google Gemini   │
                       │  (Optional LLM)  │
                       └──────────────────┘
```

## Features

- **Semantic Search**: Finds relevant textbook content based on meaning rather than keywords
- **Context-Aware Responses**: Generates answers based on retrieved context
- **Flexible Configuration**: Easy to customize embedding models and settings
- **Robust Error Handling**: Comprehensive error management
- **Health Checks**: Built-in endpoints for monitoring

## Prerequisites

- Python 3.9+
- Node.js (for the Docusaurus frontend)
- Docker (for running Qdrant)
- Textbook content in the `docs/` directory (Markdown files)

## Setup Instructions

### 1. Backend Setup

1. Clone the repository
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables in `.env`:
   ```
   GEMINI_API_KEY=your_api_key_here
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   QDRANT_COLLECTION=humanoid_robotics
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   ```

### 2. Running the Services

1. Start Qdrant (in a separate terminal):
   ```bash
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```

2. Ingest the textbook content into Qdrant:
   ```bash
   python ingest_backend.py
   ```

3. Start the FastAPI server:
   ```bash
   python backend_with_llm.py
   ```
   
   Or use the simplified version without LLM integration:
   ```bash
   python backend.py
   ```

### 3. Frontend Integration

The Docusaurus frontend can be run separately:
```bash
npm run start
```

## API Endpoints

### GET `/`
Health check endpoint.

### GET `/api/health`
Detailed health status.

### POST `/api/query`
Main query endpoint with the following request body:
```json
{
  "question": "Your question about the textbook content",
  "top_k": 3
}
```

Response:
```json
{
  "answer": "The generated answer based on textbook content",
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

## Configuration Options

- `QDRANT_HOST`: Host for Qdrant database (default: localhost)
- `QDRANT_PORT`: Port for Qdrant database (default: 6333)
- `QDRANT_COLLECTION`: Name of the Qdrant collection (default: humanoid_robotics)
- `EMBEDDING_MODEL`: Sentence transformer model to use (default: all-MiniLM-L6-v2)

## Implementation Details

### Embedding Generation
- Uses Sentence Transformers for generating text embeddings
- All-MiniLM-L6-v2 model provides good balance of performance and accuracy
- Text chunks are limited to 512 tokens with 50-token overlap

### Retrieval Process
1. Incoming question is converted to an embedding
2. Qdrant performs similarity search against stored textbook embeddings
3. Top-k most relevant chunks are retrieved based on cosine similarity
4. Retrieved context is used to generate a response

### Response Generation
- If Google Gemini is configured, uses the LLM for advanced response generation
- Otherwise, uses a keyword-based approach to synthesize information from relevant chunks
- Responses cite sources to maintain transparency

## Extending the System

### Adding New Content
- Place new Markdown files in the `docs/` directory
- Re-run the ingestion script: `python ingest_backend.py`

### Changing Embedding Models
- Update the `EMBEDDING_MODEL` environment variable
- Consider model compatibility with the sentence-transformers library

### Customizing Response Generation
- Modify the `generate_answer` function in backend files
- Integrate with different LLM providers as needed

## Troubleshooting

### Common Issues
- **Qdrant Connection**: Ensure Docker is running and ports are available
- **Ingestion Fails**: Check that `docs/` directory contains Markdown files
- **Empty Responses**: Verify collection has been populated with content

### Performance Tuning
- Adjust `top_k` parameter to balance response quality and speed
- Consider using more powerful embedding models for better relevance
- Monitor resource usage during high traffic periods

## Security Considerations

- API keys should be stored in environment variables, not in source code
- Implement rate limiting in production environments
- Validate and sanitize all user inputs
- Use HTTPS in production deployments

## License

[Add your license information here]