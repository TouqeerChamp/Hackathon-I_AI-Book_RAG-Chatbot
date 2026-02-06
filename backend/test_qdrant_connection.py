import os
from qdrant_client import QdrantClient
from dotenv import dotenv_values

# Load environment variables
config = dotenv_values(".env")
if not config:
    config = os.environ

def test_qdrant_connection():
    """
    Test function to check if we can connect to Qdrant Cloud
    """
    print("Testing Qdrant Cloud connection...")

    try:
        # Initialize Qdrant client for cloud
        client = QdrantClient(
            url=config["QDRANT_URL"],
            api_key=config["QDRANT_API_KEY"],
            https=True  # Ensuring HTTPS for cloud connection
        )

        # Try to get collections to verify connection
        collections = client.get_collections()

        print("[SUCCESS] Successfully connected to Qdrant Cloud!")
        print(f"Available collections: {[col.name for col in collections.collections]}")

        # Check if our collection exists and has content
        collection_name = config['COLLECTION_NAME']
        collection_names = [col.name for col in collections.collections]

        if collection_name in collection_names:
            print(f"Collection '{collection_name}' exists.")

            # Get count of vectors in the collection
            count = client.count(collection_name=collection_name)
            print(f"Collection '{collection_name}' has {count.count} vectors.")
        else:
            print(f"Collection '{collection_name}' does not exist. Run the ingestion script first.")

        return True
    except Exception as e:
        print(f"[ERROR] Error connecting to Qdrant Cloud: {str(e)}")
        print("Check your QDRANT_URL and QDRANT_API_KEY in the .env file.")
        return False

if __name__ == "__main__":
    test_qdrant_connection()