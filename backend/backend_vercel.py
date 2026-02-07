import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cloud Clients
hf_client = InferenceClient(token=os.getenv("HF_API_TOKEN"))
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

class QueryRequest(BaseModel):
    query: str

@app.get("")
def home():
    return {"message": "Vercel Backend is Live!"}

@app.post("/api/query")
async def process_query(request: QueryRequest):
    try:
        # Step 1: Get Embeddings from Cloud
        embeddings = hf_client.feature_extraction(
            request.query,
            model="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vector format setting
        vector = embeddings[0] if isinstance(embeddings[0], list) else (embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings)

        # Step 2: Search in Qdrant with Version Safety
        # Attempt to use the newer query_points method first (for v1.10+)
        try:
            search_result = qdrant_client.query_points(
                collection_name="humanoid_robotics",
                query=vector,
                limit=3
            ).points
        except AttributeError:
            # Fall back to the older search method (for v1.9 and below)
            search_result = qdrant_client.search(
                collection_name="humanoid_robotics",
                query_vector=vector,
                limit=3
            )

        # Extract context safely
        context = "\n".join([res.payload.get("text", "") for res in search_result if res.payload])

        # Step 3: Get Answer from AI Model
        prompt = f"Context: {context}\n\nQuestion: {request.query}\n\nAnswer:"
        response = hf_client.text_generation(
            model="gpt2",
            prompt=prompt,
            max_new_tokens=250,
            temperature=0.7,
            return_full_text=False
        )
        
        return {"answer": response}

    except Exception as e:
        print(f"Error logic: {str(e)}")
        return {"answer": f"Backend Error: {str(e)}", "sources": []}
