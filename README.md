# Claude OpenAI API

An OpenAI-compatible API server that uses Claude via command-line subprocess. This allows you to use Claude with any application that supports the OpenAI API format.

## Features

### Core API Features
- **OpenAI-compatible** chat completions endpoint (`/v1/chat/completions`)
- **Streaming and non-streaming** responses
- **Model mapping** (OpenAI models automatically mapped to Claude equivalents)
- **Full compatibility** with OpenAI Python client library
- **FastAPI-based** with automatic API documentation

### Advanced Features
- **Function calling (tools)** - Complete OpenAI-style function definitions
- **Conversation history** - Session-based conversations with persistent memory
- **JSON response mode** - Force responses in valid JSON format
- **Stop sequences** - Custom stop words/phrases
- **Multiple completions** - Generate up to 10 completions per request (n parameter)
- **Multimodal input** - Support for images in base64 format
- **Proper token counting** - Accurate usage tracking with tiktoken
- **Seed parameter** - For reproducible outputs
- **Comprehensive error handling** - Robust retry logic and proper HTTP status codes

## Prerequisites

- Claude CLI must be installed and available in your PATH
- You can verify by running: `claude --version`

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file (optional, for port configuration):
```bash
cp .env.example .env
```

## Running the Server

```bash
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --reload --port 8000
```

The server will start on `http://localhost:8000`

## API Endpoints

### Core Endpoints
- `GET /` - API information and feature list
- `GET /health` - Health check and system status
- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Create chat completion (OpenAI-compatible)

### Session Management Endpoints
- `POST /v1/sessions` - Create a new conversation session
- `GET /v1/sessions` - List all conversation sessions
- `GET /v1/sessions/{session_id}` - Get session details and history
- `DELETE /v1/sessions/{session_id}` - Delete a session
- `POST /v1/sessions/{session_id}/clear` - Clear session history

## Usage Examples

### Using curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### Using OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy-key",  # API key is not used but required by client
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(response.choices[0].message.content)
```

### Streaming Example

```python
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Count to 5"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

## Model Mapping

The following OpenAI models are mapped to Claude models:

- `gpt-4`, `gpt-4-turbo`, `gpt-4-turbo-preview` → `claude-3-5-sonnet-20241022`
- `gpt-3.5-turbo`, `gpt-3.5-turbo-16k` → `claude-3-5-haiku-20241022`

## Testing

Run the test script to verify the API is working:

```bash
python test_api.py
```

This will test:
- Listing models
- Non-streaming chat completions
- Streaming chat completions
- OpenAI client compatibility (if `openai` package is installed)

## API Documentation

FastAPI automatically generates interactive API documentation:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Environment Variables

- `PORT` - Server port (default: 8000)

## Testing

### Quick Test (No Dependencies)
```bash
# Start the server in one terminal
python main.py

# Run quick test in another terminal
python quick_test.py
```

### Comprehensive Test Suite
```bash
# Install OpenAI client for full testing
pip install openai

# Run comprehensive tests
python test_all_features.py
```

The test suite validates:
- ✅ Basic completions and streaming
- ✅ Function calling (tools)
- ✅ Conversation history and sessions
- ✅ JSON response mode
- ✅ Stop sequences
- ✅ Multiple completions (n parameter)
- ✅ Multimodal input (images)
- ✅ Token counting accuracy
- ✅ Error handling
- ✅ All OpenAI API features

### Manual Testing
You can also test individual features using curl or the OpenAI client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# Test function calling
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }
    }]
)
```

## Error Handling

The API includes proper error handling for:
- Claude CLI not found
- Invalid requests
- Subprocess errors
- Streaming errors
- Rate limiting
- Authentication issues

All errors follow the OpenAI error format for compatibility.

## How It Works

The API translates OpenAI-format requests into Claude CLI commands using subprocess. Messages are formatted and passed to the `claude` command, and the output is streamed back in OpenAI-compatible format.