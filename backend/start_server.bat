@echo off
setlocal

echo Installing Python dependencies...
pip install -r requirements.txt

echo Starting Qdrant (make sure Docker is running)...
start /b docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

echo Waiting for Qdrant to start...
timeout /t 10

echo Ingesting documents into Qdrant...
python ingest_backend.py

echo Starting FastAPI server...
python backend.py