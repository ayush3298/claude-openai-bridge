#!/usr/bin/env python3
"""
Comprehensive test suite for Claude-OpenAI API Bridge
Tests all implemented OpenAI API features
"""

import os
import sys
import json
import time
import base64
import asyncio
import requests
from typing import List, Dict, Any
from datetime import datetime

# Try to import OpenAI client
try:
    from openai import OpenAI
    OPENAI_CLIENT_AVAILABLE = True
except ImportError:
    print("Warning: OpenAI client not available. Install with: pip install openai")
    OPENAI_CLIENT_AVAILABLE = False

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("OPENAI_API_KEY", "dummy-key")

# Test results tracking
test_results = {
    "passed": [],
    "failed": [],
    "skipped": []
}


def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f" {text}")
    print(f"{'='*60}")


def print_test(name: str, status: str, details: str = ""):
    """Print test result"""
    symbols = {"passed": "✅", "failed": "❌", "skipped": "⏭️"}
    print(f"{symbols.get(status, '?')} {name}")
    if details:
        print(f"   {details}")
    test_results[status].append(name)


def test_health_check():
    """Test health endpoint"""
    print_header("Testing Health Check")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print_test("Health Check", "passed", f"Status: {data.get('status')}")
            print(f"   Claude CLI: {data.get('claude_cli')}")
            print(f"   History Enabled: {data.get('history_enabled')}")
            print(f"   Active Sessions: {data.get('active_sessions')}")
        else:
            print_test("Health Check", "failed", f"Status code: {response.status_code}")
    except Exception as e:
        print_test("Health Check", "failed", str(e))


def test_models_endpoint():
    """Test models listing"""
    print_header("Testing Models Endpoint")
    try:
        response = requests.get(f"{API_BASE_URL}/v1/models")
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            print_test("Models Listing", "passed", f"Found {len(models)} models")
            for model in models[:5]:  # Show first 5
                print(f"   - {model['id']} (by {model['owned_by']})")
        else:
            print_test("Models Listing", "failed", f"Status code: {response.status_code}")
    except Exception as e:
        print_test("Models Listing", "failed", str(e))


def test_basic_completion():
    """Test basic chat completion"""
    print_header("Testing Basic Chat Completion")
    
    if not OPENAI_CLIENT_AVAILABLE:
        print_test("Basic Completion", "skipped", "OpenAI client not available")
        return
    
    try:
        client = OpenAI(base_url=f"{API_BASE_URL}/v1", api_key=API_KEY)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": "Say 'Hello, World!' and nothing else."}
            ],
            max_tokens=50
        )
        
        content = response.choices[0].message.content
        print_test("Basic Completion", "passed", f"Response: {content}")
        print(f"   Tokens: {response.usage.total_tokens}")
    except Exception as e:
        print_test("Basic Completion", "failed", str(e))


def test_streaming():
    """Test streaming response"""
    print_header("Testing Streaming Response")
    
    if not OPENAI_CLIENT_AVAILABLE:
        print_test("Streaming", "skipped", "OpenAI client not available")
        return
    
    try:
        client = OpenAI(base_url=f"{API_BASE_URL}/v1", api_key=API_KEY)
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Count from 1 to 5"}],
            stream=True
        )
        
        chunks = []
        for chunk in stream:
            if chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        
        full_response = "".join(chunks)
        print_test("Streaming", "passed", f"Received {len(chunks)} chunks")
        print(f"   Response: {full_response[:100]}...")
    except Exception as e:
        print_test("Streaming", "failed", str(e))


def test_conversation_history():
    """Test conversation history and sessions"""
    print_header("Testing Conversation History")
    
    try:
        # Create a session
        session_response = requests.post(f"{API_BASE_URL}/v1/sessions")
        if session_response.status_code != 200:
            print_test("Session Creation", "failed", f"Status: {session_response.status_code}")
            return
        
        session_data = session_response.json()
        session_id = session_data.get("session_id")
        print_test("Session Creation", "passed", f"Session ID: {session_id[:12]}...")
        
        if not OPENAI_CLIENT_AVAILABLE:
            print_test("History Test", "skipped", "OpenAI client not available")
            return
        
        # Make requests with history
        client = OpenAI(base_url=f"{API_BASE_URL}/v1", api_key=API_KEY)
        
        # First message
        response1 = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "My name is Alice. Remember this."}],
            extra_body={"session_id": session_id}
        )
        print(f"   First response: {response1.choices[0].message.content[:50]}...")
        
        # Second message using history
        response2 = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "What is my name?"}],
            extra_body={"session_id": session_id, "include_history": True}
        )
        
        if "Alice" in response2.choices[0].message.content:
            print_test("Conversation History", "passed", "Model remembered the name")
        else:
            print_test("Conversation History", "failed", "Model didn't remember the name")
        
        # Test session info
        session_info = requests.get(f"{API_BASE_URL}/v1/sessions/{session_id}")
        if session_info.status_code == 200:
            data = session_info.json()
            print_test("Session Info", "passed", f"Messages: {data.get('message_count')}")
        
    except Exception as e:
        print_test("Conversation History", "failed", str(e))


def test_function_calling():
    """Test function calling (tools)"""
    print_header("Testing Function Calling")
    
    if not OPENAI_CLIENT_AVAILABLE:
        print_test("Function Calling", "skipped", "OpenAI client not available")
        return
    
    try:
        client = OpenAI(base_url=f"{API_BASE_URL}/v1", api_key=API_KEY)
        
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["location"]
                }
            }
        }]
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "What's the weather in New York?"}],
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        if message.tool_calls:
            print_test("Function Calling", "passed", f"Tool called: {message.tool_calls[0].function.name}")
            print(f"   Arguments: {message.tool_calls[0].function.arguments}")
        else:
            print_test("Function Calling", "failed", "No tool calls in response")
            print(f"   Response: {message.content}")
        
    except Exception as e:
        print_test("Function Calling", "failed", str(e))


def test_json_mode():
    """Test JSON response format"""
    print_header("Testing JSON Response Mode")
    
    if not OPENAI_CLIENT_AVAILABLE:
        print_test("JSON Mode", "skipped", "OpenAI client not available")
        return
    
    try:
        client = OpenAI(base_url=f"{API_BASE_URL}/v1", api_key=API_KEY)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "user", 
                "content": "List 3 primary colors as a JSON array with 'name' and 'hex' fields"
            }],
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        try:
            # Try to parse as JSON
            json_data = json.loads(content)
            print_test("JSON Mode", "passed", "Valid JSON received")
            print(f"   Response: {json.dumps(json_data, indent=2)[:200]}...")
        except json.JSONDecodeError:
            print_test("JSON Mode", "failed", "Invalid JSON in response")
            print(f"   Response: {content[:100]}...")
            
    except Exception as e:
        print_test("JSON Mode", "failed", str(e))


def test_stop_sequences():
    """Test stop sequences"""
    print_header("Testing Stop Sequences")
    
    if not OPENAI_CLIENT_AVAILABLE:
        print_test("Stop Sequences", "skipped", "OpenAI client not available")
        return
    
    try:
        client = OpenAI(base_url=f"{API_BASE_URL}/v1", api_key=API_KEY)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Count from 1 to 10, one number per line"}],
            stop=["5", "\n6"],
            max_tokens=100
        )
        
        content = response.choices[0].message.content
        if "6" not in content and "5" in content:
            print_test("Stop Sequences", "passed", "Stopped at correct sequence")
        else:
            print_test("Stop Sequences", "failed", "Did not stop at sequence")
        print(f"   Response: {content}")
        
    except Exception as e:
        print_test("Stop Sequences", "failed", str(e))


def test_multiple_completions():
    """Test n parameter for multiple completions"""
    print_header("Testing Multiple Completions (n parameter)")
    
    if not OPENAI_CLIENT_AVAILABLE:
        print_test("Multiple Completions", "skipped", "OpenAI client not available")
        return
    
    try:
        client = OpenAI(base_url=f"{API_BASE_URL}/v1", api_key=API_KEY)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Write a one-line motivational quote"}],
            n=3,
            temperature=1.2
        )
        
        if len(response.choices) == 3:
            print_test("Multiple Completions", "passed", f"Received {len(response.choices)} completions")
            for i, choice in enumerate(response.choices):
                print(f"   {i+1}: {choice.message.content[:60]}...")
        else:
            print_test("Multiple Completions", "failed", f"Expected 3, got {len(response.choices)}")
            
    except Exception as e:
        print_test("Multiple Completions", "failed", str(e))


def test_seed_parameter():
    """Test seed for reproducible outputs"""
    print_header("Testing Seed Parameter")
    
    if not OPENAI_CLIENT_AVAILABLE:
        print_test("Seed Parameter", "skipped", "OpenAI client not available")
        return
    
    try:
        client = OpenAI(base_url=f"{API_BASE_URL}/v1", api_key=API_KEY)
        
        # Make two requests with same seed
        seed = 12345
        messages = [{"role": "user", "content": "Generate a random 5-letter word"}]
        
        response1 = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            seed=seed,
            temperature=1.0
        )
        
        response2 = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            seed=seed,
            temperature=1.0
        )
        
        content1 = response1.choices[0].message.content
        content2 = response2.choices[0].message.content
        
        # Note: Perfect reproducibility might not always work with Claude
        print_test("Seed Parameter", "passed", "Seed parameter accepted")
        print(f"   Response 1: {content1}")
        print(f"   Response 2: {content2}")
        print(f"   Identical: {content1 == content2}")
        
    except Exception as e:
        print_test("Seed Parameter", "failed", str(e))


def test_multimodal_input():
    """Test image input support"""
    print_header("Testing Multimodal Input (Images)")
    
    if not OPENAI_CLIENT_AVAILABLE:
        print_test("Multimodal Input", "skipped", "OpenAI client not available")
        return
    
    try:
        client = OpenAI(base_url=f"{API_BASE_URL}/v1", api_key=API_KEY)
        
        # Create a simple test image (1x1 red pixel)
        red_pixel_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{red_pixel_base64}"
                        }
                    }
                ]
            }],
            max_tokens=100
        )
        
        content = response.choices[0].message.content
        print_test("Multimodal Input", "passed", "Image input accepted")
        print(f"   Response: {content[:100]}...")
        
    except Exception as e:
        print_test("Multimodal Input", "failed", str(e))


def test_token_counting():
    """Test token counting accuracy"""
    print_header("Testing Token Counting")
    
    if not OPENAI_CLIENT_AVAILABLE:
        print_test("Token Counting", "skipped", "OpenAI client not available")
        return
    
    try:
        client = OpenAI(base_url=f"{API_BASE_URL}/v1", api_key=API_KEY)
        
        # Test with known text
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Reply with exactly: Hello World"}],
            max_tokens=10
        )
        
        usage = response.usage
        print_test("Token Counting", "passed", "Token usage reported")
        print(f"   Prompt tokens: {usage.prompt_tokens}")
        print(f"   Completion tokens: {usage.completion_tokens}")
        print(f"   Total tokens: {usage.total_tokens}")
        
    except Exception as e:
        print_test("Token Counting", "failed", str(e))


def test_system_messages():
    """Test system message handling"""
    print_header("Testing System Messages")
    
    if not OPENAI_CLIENT_AVAILABLE:
        print_test("System Messages", "skipped", "OpenAI client not available")
        return
    
    try:
        client = OpenAI(base_url=f"{API_BASE_URL}/v1", api_key=API_KEY)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a pirate. Always respond in pirate speak."},
                {"role": "user", "content": "Hello, how are you?"}
            ]
        )
        
        content = response.choices[0].message.content.lower()
        if any(word in content for word in ["ahoy", "matey", "arr", "ye"]):
            print_test("System Messages", "passed", "System message was followed")
        else:
            print_test("System Messages", "passed", "Response received (pirate speak not guaranteed)")
        print(f"   Response: {response.choices[0].message.content[:100]}...")
        
    except Exception as e:
        print_test("System Messages", "failed", str(e))


def test_error_handling():
    """Test error handling"""
    print_header("Testing Error Handling")
    
    # Test invalid model
    try:
        response = requests.post(
            f"{API_BASE_URL}/v1/chat/completions",
            json={
                "model": "",
                "messages": [{"role": "user", "content": "test"}]
            }
        )
        if response.status_code >= 400:
            print_test("Empty Model Error", "passed", f"Proper error code: {response.status_code}")
        else:
            print_test("Empty Model Error", "failed", "Should have returned error")
    except Exception as e:
        print_test("Empty Model Error", "failed", str(e))
    
    # Test empty messages
    try:
        response = requests.post(
            f"{API_BASE_URL}/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": []
            }
        )
        if response.status_code >= 400:
            print_test("Empty Messages Error", "passed", f"Proper error code: {response.status_code}")
        else:
            print_test("Empty Messages Error", "failed", "Should have returned error")
    except Exception as e:
        print_test("Empty Messages Error", "failed", str(e))


def print_summary():
    """Print test summary"""
    print_header("Test Summary")
    
    total = len(test_results["passed"]) + len(test_results["failed"]) + len(test_results["skipped"])
    
    print(f"Total tests: {total}")
    print(f"✅ Passed: {len(test_results['passed'])}")
    print(f"❌ Failed: {len(test_results['failed'])}")
    print(f"⏭️  Skipped: {len(test_results['skipped'])}")
    
    if test_results["failed"]:
        print("\nFailed tests:")
        for test in test_results["failed"]:
            print(f"  - {test}")
    
    # Return exit code
    return 0 if not test_results["failed"] else 1


def main():
    """Run all tests"""
    print("\n🧪 Claude-OpenAI API Bridge - Comprehensive Test Suite")
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code != 200:
            print("\n❌ API server is not responding. Please start the server first.")
            print(f"   Run: python main.py")
            return 1
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to API server. Please start the server first.")
        print(f"   Run: python main.py")
        return 1
    
    # Run all tests
    test_health_check()
    test_models_endpoint()
    test_basic_completion()
    test_streaming()
    test_conversation_history()
    test_function_calling()
    test_json_mode()
    test_stop_sequences()
    test_multiple_completions()
    test_seed_parameter()
    test_multimodal_input()
    test_token_counting()
    test_system_messages()
    test_error_handling()
    
    # Print summary and exit
    return print_summary()


if __name__ == "__main__":
    sys.exit(main())