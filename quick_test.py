#!/usr/bin/env python3
"""
Quick API test using only requests library
No external dependencies required
"""

import requests
import json
import time
import base64

API_BASE = "http://localhost:8000"

def test_endpoint(name, method, path, data=None, headers=None):
    """Test an API endpoint"""
    try:
        url = f"{API_BASE}{path}"
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)
        else:
            print(f"❌ {name}: Unknown method {method}")
            return False
            
        if response.status_code < 300:
            print(f"✅ {name}: {response.status_code}")
            if response.text:
                try:
                    data = response.json()
                    print(f"   Response: {json.dumps(data, indent=2)[:200]}...")
                except:
                    print(f"   Response: {response.text[:200]}...")
            return True
        else:
            print(f"❌ {name}: {response.status_code}")
            print(f"   Error: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ {name}: {str(e)}")
        return False


def main():
    print("\n🚀 Quick Claude-OpenAI API Test\n")
    
    # 1. Test root endpoint
    test_endpoint("Root Endpoint", "GET", "/")
    
    # 2. Test health check
    test_endpoint("Health Check", "GET", "/health")
    
    # 3. Test models listing
    test_endpoint("List Models", "GET", "/v1/models")
    
    # 4. Test basic completion
    test_endpoint(
        "Basic Completion",
        "POST",
        "/v1/chat/completions",
        {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Say hello in 3 words"}],
            "max_tokens": 20
        }
    )
    
    # 5. Test with system message
    test_endpoint(
        "System Message",
        "POST",
        "/v1/chat/completions",
        {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What are you?"}
            ],
            "max_tokens": 50
        }
    )
    
    # 6. Test JSON mode
    test_endpoint(
        "JSON Response Mode",
        "POST",
        "/v1/chat/completions",
        {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "List 2 colors as JSON with name and hex"}],
            "response_format": {"type": "json_object"},
            "max_tokens": 100
        }
    )
    
    # 7. Test function calling
    test_endpoint(
        "Function Calling",
        "POST",
        "/v1/chat/completions",
        {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            }],
            "max_tokens": 150
        }
    )
    
    # 8. Test stop sequences
    test_endpoint(
        "Stop Sequences",
        "POST",
        "/v1/chat/completions",
        {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Count: 1, 2, 3, 4, 5, 6, 7, 8"}],
            "stop": ["5"],
            "max_tokens": 50
        }
    )
    
    # 9. Test multiple completions
    test_endpoint(
        "Multiple Completions (n=2)",
        "POST",
        "/v1/chat/completions",
        {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Give me a random number"}],
            "n": 2,
            "temperature": 1.5,
            "max_tokens": 20
        }
    )
    
    # 10. Test conversation with session
    print("\n--- Testing Conversation History ---")
    
    # Create session
    session_resp = requests.post(f"{API_BASE}/v1/sessions")
    if session_resp.status_code == 200:
        session_id = session_resp.json()["session_id"]
        print(f"✅ Created session: {session_id[:12]}...")
        
        # First message
        test_endpoint(
            "First Message with Session",
            "POST",
            "/v1/chat/completions",
            {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "My favorite color is blue. Remember this."}],
                "session_id": session_id,
                "max_tokens": 50
            }
        )
        
        # Second message using history
        test_endpoint(
            "Second Message with History",
            "POST",
            "/v1/chat/completions",
            {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "What's my favorite color?"}],
                "session_id": session_id,
                "include_history": True,
                "max_tokens": 50
            }
        )
        
        # Get session info
        test_endpoint("Get Session Info", "GET", f"/v1/sessions/{session_id}")
        
        # List sessions
        test_endpoint("List Sessions", "GET", "/v1/sessions")
    
    # 11. Test streaming (basic check)
    print("\n--- Testing Streaming ---")
    try:
        response = requests.post(
            f"{API_BASE}/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Count to 3"}],
                "stream": True,
                "max_tokens": 50
            },
            stream=True
        )
        
        if response.status_code == 200:
            print("✅ Streaming Response")
            chunks = 0
            for line in response.iter_lines():
                if line:
                    chunks += 1
                    if chunks <= 3:  # Show first 3 chunks
                        print(f"   Chunk {chunks}: {line.decode('utf-8')[:100]}...")
            print(f"   Total chunks: {chunks}")
        else:
            print(f"❌ Streaming: {response.status_code}")
    except Exception as e:
        print(f"❌ Streaming: {str(e)}")
    
    # 12. Test error handling
    print("\n--- Testing Error Handling ---")
    test_endpoint(
        "Invalid Request (empty messages)",
        "POST",
        "/v1/chat/completions",
        {"model": "gpt-4", "messages": []}
    )
    
    print("\n✨ Quick test completed!")


if __name__ == "__main__":
    main()