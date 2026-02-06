import requests
import json
import time

def test_rag_api():
    # Wait a bit for the server to fully start and complete ingestion
    print("Waiting for server to be ready...")
    time.sleep(10)  # Wait 10 seconds
    
    # Test health endpoint
    try:
        health_response = requests.get('http://localhost:3003/health')
        print(f"Health check status: {health_response.status_code}")
        print(f"Health check response: {health_response.json()}")
    except Exception as e:
        print(f"Error connecting to health endpoint: {e}")
        return False
    
    # Test chat endpoint with a sample question
    try:
        sample_question = {
            "query": "What is humanoid robotics?"
        }
        chat_response = requests.post(
            'http://localhost:3003/api/chat',
            headers={'Content-Type': 'application/json'},
            data=json.dumps(sample_question)
        )
        
        print(f"Chat endpoint status: {chat_response.status_code}")
        if chat_response.status_code == 200:
            response_data = chat_response.json()
            print(f"Chat response: {response_data}")
            return True
        else:
            print(f"Chat endpoint returned status code: {chat_response.status_code}")
            print(f"Chat response: {chat_response.text}")
            return False
            
    except Exception as e:
        print(f"Error connecting to chat endpoint: {e}")
        return False

if __name__ == "__main__":
    success = test_rag_api()
    if success:
        print("\nRAG API is working correctly!")
    else:
        print("\nThere were issues with the RAG API.")