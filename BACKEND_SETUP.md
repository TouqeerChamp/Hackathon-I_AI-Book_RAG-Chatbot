# Backend Setup Guide for Physical AI & Humanoid Robotics RAG System

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git
- Docker (for Qdrant)

## Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/TouqeerChamp/Hackathon-I_AI-Book_RAG-Chatbot.git
   cd Hackathon-I_AI-Book_RAG-Chatbot
   ```

2. Navigate to the backend directory:
   ```bash
   cd backend
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

4. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\\Scripts\\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Environment Variables

Create a `.env` file in the `backend` directory with the following variables:

```env
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
HF_API_TOKEN=your_huggingface_api_token
COLLECTION_NAME=your_collection_name
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
GENERATION_MODEL_NAME=mistralai/Mistral-7B-v0.1
```

## Running the Backend

1. Make sure your virtual environment is activated

2. Start the backend server:
   ```bash
   python backend.py
   ```

   Or if using uvicorn:
   ```bash
   uvicorn backend:app --reload --port 8000
   ```

3. The API will be available at `http://localhost:8000`

## Qdrant Setup

### Using Docker (Recommended)

1. Install Docker Desktop

2. Run Qdrant using Docker:
   ```bash
   docker-compose up -d
   ```

   Or if you don't have docker-compose, run:
   ```bash
   docker run -d --name qdrant-container -p 6333:6333 qdrant/qdrant
   ```

3. Verify Qdrant is running at `http://localhost:6333`

### Manual Qdrant Setup

1. Install Qdrant using pip:
   ```bash
   pip install qdrant-client
   ```

2. Run Qdrant server:
   ```bash
   qdrant
   ```

## Data Ingestion

1. Place your textbook content in the `docs/` directory as markdown files

2. Run the ingestion script:
   ```bash
   python ingest_backend.py
   ```

3. The script will:
   - Read all markdown files in the `docs/` directory
   - Chunk the content
   - Generate embeddings using the Hugging Face model
   - Store the embeddings in Qdrant

## Troubleshooting

### Common Issues:
- If you get SSL certificate errors, try adding `--trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org` to your pip install command
- If you get "port already in use" errors, try a different port or kill the process using that port
- If you get "module not found" errors, make sure your virtual environment is activated

### Qdrant issues:
- If Qdrant container fails to start, verify Docker is running and you have proper permissions

### API Access:
- The API will be available at: https://hackathon-i-ai-book-rag-chatbotfina.vercel.app
- API documentation available at: https://hackathon-i-ai-book-rag-chatbotfina.vercel.app/docs

## Usage

1. Once the system is running, add your textbook content to the `docs/` directory as markdown files

2. Run the ingestion script to index your content

3. Use the API endpoints to query your content

## API Endpoints

- `GET /` - Health check endpoint
- `POST /api/query` - Query endpoint for RAG system

## Security

- Never commit your `.env` file to version control
- Use strong API keys and rotate them regularly
- Limit access to your Qdrant instance

## Production Deployment

For production deployment, consider:
- Using environment variables for configuration
- Setting up proper logging
- Implementing rate limiting
- Adding authentication if needed

