import requests
import json
import sseclient
from typing import Iterator

API_BASE_URL = "http://localhost:8000"


def test_chat_completion():
    url = f"{API_BASE_URL}/v1/chat/completions"
    
    payload = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print("Non-streaming response:")
        print(json.dumps(data, indent=2))
        print(f"\nAssistant's response: {data['choices'][0]['message']['content']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def test_streaming_chat_completion():
    url = f"{API_BASE_URL}/v1/chat/completions"
    
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Count from 1 to 5 slowly."}
        ],
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 150
    }
    
    response = requests.post(url, json=payload, stream=True)
    
    if response.status_code == 200:
        print("\nStreaming response:")
        client = sseclient.SSEClient(response)
        
        full_content = ""
        for event in client.events():
            if event.data != "[DONE]":
                try:
                    chunk = json.loads(event.data)
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            content = delta["content"]
                            print(content, end="", flush=True)
                            full_content += content
                except json.JSONDecodeError:
                    pass
        
        print(f"\n\nFull response: {full_content}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def test_list_models():
    url = f"{API_BASE_URL}/v1/models"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        print("\nAvailable models:")
        print(json.dumps(data, indent=2))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def test_with_openai_client():
    from openai import OpenAI
    
    client = OpenAI(
        api_key="dummy-key",
        base_url=API_BASE_URL + "/v1"
    )
    
    print("\nTesting with OpenAI Python client:")
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2 + 2?"}
        ],
        temperature=0.7,
        max_tokens=50
    )
    
    print(f"Response: {response.choices[0].message.content}")
    print(f"Usage: {response.usage}")
    
    print("\nTesting streaming with OpenAI client:")
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Tell me a very short joke."}
        ],
        stream=True,
        temperature=0.7,
        max_tokens=100
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


if __name__ == "__main__":
    print("Testing Claude OpenAI API")
    print("=" * 50)
    
    try:
        test_list_models()
        print("\n" + "=" * 50)
        
        test_chat_completion()
        print("\n" + "=" * 50)
        
        test_streaming_chat_completion()
        print("\n" + "=" * 50)
        
        try:
            test_with_openai_client()
        except ImportError:
            print("\nOpenAI client not installed. Install with: pip install openai")
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running on port 8000.")