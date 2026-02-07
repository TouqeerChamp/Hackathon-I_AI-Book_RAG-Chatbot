import os

from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from huggingface_hub import InferenceClient

from qdrant_client import QdrantClient

import qdrant_client as qc

from packaging import version

from dotenv import load_dotenv




load_dotenv()

# Create the FastAPI app
app = FastAPI()



app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_methods=["*"],

    allow_headers=["*"],

)


# Initialize clients but defer error handling to the request level
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Initialize clients globally but handle missing env vars gracefully
hf_client = InferenceClient(token=HF_API_TOKEN) if HF_API_TOKEN else None
qdrant_client_inst = None

if QDRANT_URL and QDRANT_API_KEY:
    qdrant_client_inst = QdrantClient(

        url=QDRANT_URL,

        api_key=QDRANT_API_KEY

    )


class QueryRequest(BaseModel):

    query: str


@app.get("")
def home():
    return {"message": "Vercel Backend is Live!"}


@app.post("/api/query")

async def process_query(request: QueryRequest):

    try:
        # Check if required clients are initialized
        if not hf_client:
            return {"answer": "Backend Error: HF client not initialized. Missing HF_API_TOKEN.", "sources": []}
        
        if not qdrant_client_inst:
            return {"answer": "Backend Error: Qdrant client not initialized. Missing QDRANT_URL or QDRANT_API_KEY.", "sources": []}

        # Step 1: Embeddings

        embeddings = hf_client.feature_extraction(

            request.query,

            model="sentence-transformers/all-MiniLM-L6-v2"

        )

        vector = embeddings[0] if isinstance(embeddings[0], list) else embeddings.tolist()



        # Step 2: Qdrant Search (Version Safe)

        v = version.parse(qc.__version__)

        if v >= version.parse("1.10.0"):

            search_result = qdrant_client_inst.query_points(

                collection_name="humanoid_robotics",

                query=vector,

                limit=3

            ).points

        else:

            search_result = qdrant_client_inst.search(

                collection_name="humanoid_robotics",

                query_vector=vector,

                limit=3

            )



        context = "\n".join([res.payload.get("text", "") for res in search_result if res.payload])



        # Step 3: Powerful Free AI Model (Mistral-7B)

        prompt = f"Context: {context}\n\nQuestion: {request.query}\n\nAnswer concisely based on context:"



        # text_generation zyada stable hai free tier par

        response = hf_client.text_generation(

            prompt=prompt,

            model="mistralai/Mistral-7B-v0.1",

            max_new_tokens=300,

            temperature=0.7,

            return_full_text=False

        )



        return {"answer": response}



    except Exception as e:

        return {"answer": f"Backend Error: {str(e)}", "sources": []}