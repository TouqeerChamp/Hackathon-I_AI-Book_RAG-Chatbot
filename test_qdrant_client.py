from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import dotenv_values

# Load environment variables
config = dotenv_values(".env")
if not config:
    config = os.environ

# Test script to validate QdrantClient functionality for Qdrant Cloud
def test_qdrant_client():
    client = QdrantClient(
        url=config["QDRANT_URL"],
        api_key=config["QDRANT_API_KEY"],
        https=True  # Ensuring HTTPS for cloud connection
    )

    print("Testing Qdrant Cloud client...")

    # Check if the search method exists
    if hasattr(client, 'search'):
        print("OK: search method exists")
    else:
        print("ERROR: search method does not exist")

    # Check available attributes
    print("Available methods/attributes:", [attr for attr in dir(client) if not attr.startswith('_')])

    # Try to get collections
    try:
        collections = client.get_collections()
        print("Collections:", collections)
    except Exception as e:
        print("Error getting collections:", e)

    # Try a simple search if the method exists
    if hasattr(client, 'search'):
        try:
            # Just check if the collection exists
            collection_name = config['COLLECTION_NAME']
            records_count = client.count(collection_name=collection_name)
            print(f"Records count in {collection_name}:", records_count)
        except Exception as e:
            print("Error getting collection count:", e)

    print("Test complete.")

if __name__ == "__main__":
    test_qdrant_client()