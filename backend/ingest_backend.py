import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from dotenv import dotenv_values
import logging
from typing import List, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

# Load environment variables
config = dotenv_values(".env")
if not config:
    config = os.environ

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
COLLECTION_NAME = config['COLLECTION_NAME']
QDRANT_URL = config['QDRANT_URL']
QDRANT_API_KEY = config['QDRANT_API_KEY']
HF_API_TOKEN = config['HF_API_TOKEN']
EMBEDDING_MODEL_NAME = config['EMBEDDING_MODEL_NAME']
DOCS_PATH = Path("./docs")  # Path to your textbook content

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(
    model_name=config["EMBEDDING_MODEL_NAME"]
)

# Initialize Qdrant client for cloud
client = QdrantClient(
    url=config["QDRANT_URL"],
    api_key=config["QDRANT_API_KEY"],
    https=True  # Ensuring HTTPS for cloud connection
)

# Initialize Qdrant vector store
qdrant = Qdrant(
    client=client,
    collection_name=config["COLLECTION_NAME"],
    embeddings=embeddings,
)


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Find the nearest sentence boundary if possible
        if end < len(text):
            # Look for period, exclamation mark, or question mark
            for punct in '.!?':
                last_punct = text.rfind(punct, start, end)
                if last_punct != -1 and last_punct > start + chunk_size // 2:
                    end = last_punct + 1
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position forward, considering overlap
        start = end - overlap if end < len(text) else len(text)

        # If no progress is being made, advance by chunk_size anyway
        if start == end:
            start = end

    return chunks


def read_docs(docs_path: Path) -> List[Tuple[str, str]]:
    """
    Read all markdown files from the docs directory
    Returns list of tuples (content, source_filename)
    """
    docs_data = []

    for file_path in docs_path.glob("*.md"):
        logger.info(f"Reading file: {file_path.name}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

                # Skip if content is too short (probably just frontmatter)
                if len(content.strip()) < 50:
                    logger.warning(f"Skipping {file_path.name} due to short length")
                    continue

                docs_data.append((content, file_path.name))
        except Exception as e:
            logger.error(f"Error reading file {file_path.name}: {str(e)}")

    return docs_data


def create_collection_if_not_exists():
    """
    Create the Qdrant collection if it doesn't exist
    """
    try:
        # Get existing collections
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]

        # Determine embedding dimension from the HuggingFace model
        sample_embedding = embeddings.embed_query("sample text")
        embedding_size = len(sample_embedding)

        if config["COLLECTION_NAME"] not in collection_names:
            logger.info(f"Creating collection: {config['COLLECTION_NAME']}")
            client.create_collection(
                collection_name=config["COLLECTION_NAME"],
                vectors_config=models.VectorParams(
                    size=embedding_size,
                    distance=models.Distance.COSINE
                ),
            )
            logger.info(f"Collection {config['COLLECTION_NAME']} created successfully")
        else:
            logger.info(f"Collection {config['COLLECTION_NAME']} already exists")

    except Exception as e:
        logger.error(f"Error creating collection: {str(e)}")
        raise


async def ingest_documents():
    """
    Main ingestion function to process all documents and store embeddings in Qdrant
    """
    logger.info("Starting ingestion process...")

    # Create collection if it doesn't exist
    create_collection_if_not_exists()

    # Read documents
    docs_data = read_docs(DOCS_PATH)

    if not docs_data:
        logger.error("No documents found to ingest. Please check the docs directory.")
        return

    # Process documents using LangChain's Qdrant integration
    for content, source in docs_data:
        logger.info(f"Processing document: {source}")

        # Split the content into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(content)

        # Add documents to Qdrant using the LangChain integration
        texts_with_metadata = []
        metadatas = []
        for chunk_idx, chunk in enumerate(chunks):
            texts_with_metadata.append(chunk)
            metadatas.append({
                "source": source,
                "chunk_index": chunk_idx
            })

        # Add to Qdrant
        qdrant.add_texts(
            texts=texts_with_metadata,
            metadatas=metadatas
        )

        logger.info(f"Added {len(chunks)} chunks from {source} to collection {config['COLLECTION_NAME']}")

    logger.info(f"Ingestion complete! Documents uploaded to collection {config['COLLECTION_NAME']}")


if __name__ == "__main__":
    asyncio.run(ingest_documents())