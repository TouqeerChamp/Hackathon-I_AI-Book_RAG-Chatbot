@echo off
setlocal

echo Checking if Docker is installed and running...
docker --version >nul 2>&1
if errorlevel 1 (
    echo Docker is not installed. Please install Docker Desktop to run Qdrant.
    echo Visit: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

echo Docker is installed. Checking if Docker daemon is running...
docker ps >nul 2>&1
if errorlevel 1 (
    echo Docker daemon is not running. Please start Docker Desktop application.
    pause
    exit /b 1
)

echo Installing Python dependencies...
pip install -r requirements.txt

echo Starting Qdrant in Docker...
start /b docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

echo Waiting for Qdrant to start (15 seconds)...
timeout /t 15 /nobreak >nul

echo Ingesting textbook content into Qdrant...
python ingest_backend.py

echo Starting FastAPI backend server...
python backend_with_llm.py