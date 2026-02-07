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

app = FastAPI()



app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_methods=["*"],

    allow_headers=["*"],

)



hf_client = InferenceClient(token=os.getenv("HF_API_TOKEN"))

qdrant_client_inst = QdrantClient(

    url=os.getenv("QDRANT_URL"),

    api_key=os.getenv("QDRANT_API_KEY")

)



class QueryRequest(BaseModel):

    query: str


@app.get("/")
def home():
    return {"message": "Vercel Backend is Live!"}


@app.post("/api/query")

async def process_query(request: QueryRequest):

    try:

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