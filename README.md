# Claude OpenAI API Bridge

An OpenAI-compatible API server that uses Claude via command-line subprocess. This allows you to use Claude with any application that supports the OpenAI API format, with built-in security restrictions for safe operation.

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
- **Response API** - OpenAI Response API format support (`/v1/responses`)
- **JSON response mode** - Force responses in valid JSON format
- **Stop sequences** - Custom stop words/phrases
- **Multiple completions** - Generate up to 10 completions per request (n parameter)
- **Multimodal input** - Support for images in base64 format
- **Proper token counting** - Accurate usage tracking with tiktoken
- **Seed parameter** - For reproducible outputs
- **Comprehensive error handling** - Robust retry logic and proper HTTP status codes

### Security Features
- **Web-only mode** - Restrict Claude to only web tools (WebSearch, WebFetch, WebView)
- **Tool access control** - Configure allowed/disallowed tools
- **File system protection** - Block all file system access by default
- **Configurable restrictions** - Fine-tune security settings via environment variables

## Prerequisites

- Python 3.11+
- Docker and Docker Compose (for containerized deployment)
- Claude CLI (for local installation)

## Quick Start with Docker

```bash
# Clone the repository
git clone https://github.com/ayush3298/claude-openai-bridge.git
cd claude-openai-bridge

# Run with Docker (recommended)
make run

# Or using docker-compose directly
docker-compose up -d
```

The API will be available at `http://localhost:8000`

## Installation Options

### Option 1: Docker (Recommended)

```bash
# Basic usage (web-only mode for security)
docker-compose up -d

# Development mode with hot reload
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Production mode with Nginx reverse proxy
docker-compose --profile production up -d

# With Redis for distributed sessions
docker-compose --profile redis up -d
```

### Option 2: Local Installation

1. Install Claude CLI and verify:
```bash
claude --version
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment (optional):
```bash
cp .env.example .env
# Edit .env with your settings
```

4. Run the server:
```bash
python main.py
```

## API Endpoints

### Core Endpoints
- `GET /` - API information and feature list
- `GET /health` - Health check and system status
- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Create chat completion (OpenAI-compatible)
- `POST /v1/responses` - Create response (OpenAI Response API format)
- `GET /v1/security` - View current security configuration

### Session Management Endpoints
- `POST /v1/sessions` - Create a new conversation session
- `GET /v1/sessions` - List all conversation sessions
- `GET /v1/sessions/{session_id}` - Get session details and history
- `DELETE /v1/sessions/{session_id}` - Delete a session
- `POST /v1/sessions/{session_id}/clear` - Clear session history

### Response Management Endpoints
- `GET /v1/responses/{response_id}` - Retrieve stored response
- `GET /v1/responses` - List all stored responses
- `DELETE /v1/responses/{response_id}` - Delete stored response

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

## Configuration

### Environment Variables

```bash
# Server Configuration
PORT=8000                    # API server port
HOST=0.0.0.0                # API server host
DEBUG=false                  # Enable debug mode

# Claude CLI Configuration
CLAUDE_COMMAND=claude        # Path to Claude CLI
REQUEST_TIMEOUT=60          # Request timeout in seconds
MAX_RETRIES=3               # Maximum retry attempts

# History Management
ENABLE_HISTORY=true         # Enable conversation history
HISTORY_STORAGE=memory      # Storage type: memory, file
HISTORY_DIR=./conversations # Directory for file storage
MAX_HISTORY_MESSAGES=100    # Max messages per session
HISTORY_TTL_HOURS=24        # History expiration time

# Security Configuration
RESTRICTED_MODE=true        # Enable restricted mode
ALLOWED_TOOLS=WebSearch,WebFetch,WebView  # Allowed tools
DISALLOWED_TOOLS=Bash,Edit,Write,Read,... # Blocked tools
```

See `.env.example` for a complete configuration template.

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

## Docker Commands

```bash
# Build and run
make build          # Build Docker image
make run            # Run in default mode
make run-dev        # Run in development mode
make run-prod       # Run with Nginx proxy
make run-redis      # Run with Redis sessions

# Management
make stop           # Stop all containers
make clean          # Remove containers and volumes
make logs           # View container logs
make shell          # Open shell in container
make health         # Check API health

# Development
make test           # Run tests
make lint           # Run linting
make format         # Format code with black
```

## Security Considerations

By default, the API runs in **web-only mode** which:
- ✅ Allows: WebSearch, WebFetch, WebView
- ❌ Blocks: Bash, Edit, Write, Read, LS, Grep, Glob, etc.
- ❌ Blocks: All file system access
- ❌ Blocks: Command execution on host

This ensures Claude cannot access or modify your host system.

## How It Works

The API translates OpenAI-format requests into Claude CLI commands using subprocess. Messages are formatted and passed to the `claude` command with security restrictions, and the output is streamed back in OpenAI-compatible format.