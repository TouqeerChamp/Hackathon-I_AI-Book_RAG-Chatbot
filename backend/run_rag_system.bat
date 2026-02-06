@echo off
setlocal

echo Installing Python dependencies...
cd backend
pip install -r requirements.txt
cd ..

echo Stopping any existing Qdrant container...
docker stop qdrant_rag 2>nul

echo Removing any stopped Qdrant container...
docker rm qdrant_rag 2>nul

echo Starting Qdrant (make sure Docker is running)...
docker run -d --name qdrant_rag -p 6333:6333 -p 6334:6334 qdrant/qdrant

echo Waiting for Qdrant to start...
timeout /t 15

echo Ingesting textbook content into Qdrant...
cd backend
python ingest_backend.py
cd ..

echo Starting FastAPI backend server...
cd backend
python backend_with_llm.py
cd ..