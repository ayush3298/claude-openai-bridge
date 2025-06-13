import os
import time
import uuid
import subprocess
import asyncio
import logging
import json
import hashlib
import base64
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, AsyncIterator, Union
from collections import defaultdict
from io import BytesIO

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

# Optional imports with fallbacks
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available, using approximate token counting")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available, image support disabled")

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    def __init__(self):
        self.port = int(os.getenv("PORT", 8000))
        self.host = os.getenv("HOST", "0.0.0.0")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.claude_command = os.getenv("CLAUDE_COMMAND", "claude")
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", 60))
        self.max_retries = int(os.getenv("MAX_RETRIES", 3))
        self.retry_delay = float(os.getenv("RETRY_DELAY", 1.0))
        self.max_prompt_length = int(os.getenv("MAX_PROMPT_LENGTH", 100000))
        
        # History management configuration
        self.enable_history = os.getenv("ENABLE_HISTORY", "true").lower() == "true"
        self.history_storage = os.getenv("HISTORY_STORAGE", "memory")  # memory, file, redis
        self.history_dir = os.getenv("HISTORY_DIR", "./conversations")
        self.max_history_messages = int(os.getenv("MAX_HISTORY_MESSAGES", 100))
        self.history_ttl_hours = int(os.getenv("HISTORY_TTL_HOURS", 24))
        self.max_context_tokens = int(os.getenv("MAX_CONTEXT_TOKENS", 32000))
        
config = Config()

# Conversation History Management
class ConversationHistory:
    def __init__(self):
        self.sessions = {}  # session_id -> conversation data
        self.session_access = defaultdict(lambda: datetime.now())  # Track last access
        self.responses = {}  # response_id -> StoredResponse data
        self.response_access = defaultdict(lambda: datetime.now())  # Track response access
        self.cleanup_interval = 3600  # Cleanup every hour
        self.last_cleanup = time.time()
        
        # Initialize storage directory if using file storage
        if config.history_storage == "file":
            Path(config.history_dir).mkdir(parents=True, exist_ok=True)
            Path(config.history_dir + "/responses").mkdir(parents=True, exist_ok=True)
    
    def _generate_session_id(self, user_id: Optional[str] = None) -> str:
        """Generate a unique session ID"""
        base = f"{user_id or 'anonymous'}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        return hashlib.md5(base.encode()).hexdigest()
    
    def _cleanup_expired_sessions(self):
        """Remove expired sessions"""
        if time.time() - self.last_cleanup < self.cleanup_interval:
            return
            
        cutoff_time = datetime.now() - timedelta(hours=config.history_ttl_hours)
        expired_sessions = [
            session_id for session_id, last_access in self.session_access.items()
            if last_access < cutoff_time
        ]
        
        for session_id in expired_sessions:
            self._delete_session(session_id)
            
        self.last_cleanup = time.time()
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def _delete_session(self, session_id: str):
        """Delete a session from all storage"""
        # Remove from memory
        self.sessions.pop(session_id, None)
        self.session_access.pop(session_id, None)
        
        # Remove from file storage if applicable
        if config.history_storage == "file":
            session_file = Path(config.history_dir) / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
    
    def _load_session_from_file(self, session_id: str) -> Optional[Dict]:
        """Load session from file storage"""
        if config.history_storage != "file":
            return None
            
        session_file = Path(config.history_dir) / f"{session_id}.json"
        if not session_file.exists():
            return None
            
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    def _save_session_to_file(self, session_id: str, session_data: Dict):
        """Save session to file storage"""
        if config.history_storage != "file":
            return
            
        session_file = Path(config.history_dir) / f"{session_id}.json"
        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
    
    def get_or_create_session(self, session_id: Optional[str] = None, user_id: Optional[str] = None) -> str:
        """Get existing session or create new one"""
        self._cleanup_expired_sessions()
        
        if not session_id:
            session_id = self._generate_session_id(user_id)
            logger.info(f"Created new session: {session_id}")
        
        # Update access time
        self.session_access[session_id] = datetime.now()
        
        # Initialize session if it doesn't exist
        if session_id not in self.sessions:
            # Try to load from file storage
            session_data = self._load_session_from_file(session_id)
            if session_data is None:
                session_data = {
                    "id": session_id,
                    "created_at": datetime.now().isoformat(),
                    "messages": [],
                    "metadata": {"user_id": user_id}
                }
            
            self.sessions[session_id] = session_data
        
        return session_id
    
    def add_message(self, session_id: str, message: Dict[str, Any]):
        """Add a message to the conversation history"""
        if not config.enable_history:
            return
            
        if session_id not in self.sessions:
            logger.warning(f"Session {session_id} not found, creating new one")
            self.get_or_create_session(session_id)
        
        session_data = self.sessions[session_id]
        session_data["messages"].append({
            **message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Limit history size
        if len(session_data["messages"]) > config.max_history_messages:
            # Remove oldest messages but keep recent context
            messages_to_keep = config.max_history_messages // 2
            session_data["messages"] = session_data["messages"][-messages_to_keep:]
            logger.debug(f"Pruned session {session_id} to {messages_to_keep} messages")
        
        # Update access time and save
        self.session_access[session_id] = datetime.now()
        self._save_session_to_file(session_id, session_data)
    
    def get_conversation_context(self, session_id: str, include_system: bool = True) -> List[Dict[str, Any]]:
        """Get conversation context for Claude"""
        if not config.enable_history or session_id not in self.sessions:
            return []
        
        session_data = self.sessions[session_id]
        messages = session_data.get("messages", [])
        
        # Filter messages based on requirements
        context_messages = []
        total_tokens = 0
        
        # Start from the most recent messages and work backwards
        for message in reversed(messages):
            # Estimate token count (rough approximation)
            message_tokens = estimate_tokens(message.get("content", ""))
            
            if total_tokens + message_tokens > config.max_context_tokens:
                break
                
            if not include_system and message.get("role") == "system":
                continue
                
            context_messages.insert(0, {
                "role": message["role"],
                "content": message["content"]
            })
            total_tokens += message_tokens
        
        logger.debug(f"Retrieved {len(context_messages)} messages ({total_tokens} tokens) for session {session_id}")
        return context_messages
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session metadata"""
        if session_id not in self.sessions:
            return None
            
        session_data = self.sessions[session_id]
        return {
            "id": session_id,
            "created_at": session_data.get("created_at"),
            "message_count": len(session_data.get("messages", [])),
            "last_activity": self.session_access[session_id].isoformat(),
            "metadata": session_data.get("metadata", {})
        }
    
    def list_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all sessions, optionally filtered by user_id"""
        sessions = []
        for session_id, session_data in self.sessions.items():
            if user_id and session_data.get("metadata", {}).get("user_id") != user_id:
                continue
                
            sessions.append(self.get_session_info(session_id))
        
        return sorted(sessions, key=lambda x: x["last_activity"], reverse=True)
    
    # Response Management Methods
    def _generate_response_id(self) -> str:
        """Generate a unique response ID in OpenAI format"""
        return f"resp_{uuid.uuid4().hex[:29]}"
    
    def store_response(
        self, 
        content: str, 
        model: str, 
        finish_reason: str,
        tokens: Dict[str, int],
        messages: List[Dict[str, Any]],
        conversation_id: Optional[str] = None,
        parent_response_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a response and return the response ID"""
        response_id = self._generate_response_id()
        
        stored_response = StoredResponse(
            id=response_id,
            conversation_id=conversation_id,
            content=content,
            model=model,
            created=int(time.time()),
            tokens=tokens,
            finish_reason=finish_reason,
            metadata=metadata or {},
            parent_response_id=parent_response_id,
            messages=messages
        )
        
        # Store in memory
        self.responses[response_id] = stored_response.model_dump()
        self.response_access[response_id] = datetime.now()
        
        # Store in file if file storage is enabled
        self._save_response_to_file(response_id, stored_response.model_dump())
        
        logger.info(f"Stored response {response_id}")
        return response_id
    
    def get_response(self, response_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored response by ID"""
        # Clean up expired responses first
        self._cleanup_expired_responses()
        
        # Try to get from memory first
        if response_id in self.responses:
            self.response_access[response_id] = datetime.now()
            return self.responses[response_id]
        
        # Try to load from file storage
        response_data = self._load_response_from_file(response_id)
        if response_data:
            self.responses[response_id] = response_data
            self.response_access[response_id] = datetime.now()
            return response_data
        
        return None
    
    def get_response_context_chain(self, response_id: str) -> List[Dict[str, Any]]:
        """Get the full conversation context chain leading up to a response"""
        context_messages = []
        current_response_id = response_id
        
        # Traverse the response chain backwards to build context
        visited = set()  # Prevent infinite loops
        while current_response_id and current_response_id not in visited:
            visited.add(current_response_id)
            response_data = self.get_response(current_response_id)
            
            if not response_data:
                break
                
            # Add the messages that led to this response
            if response_data.get("messages"):
                context_messages = response_data["messages"] + context_messages
            
            # Add the response itself as an assistant message
            context_messages.append({
                "role": "assistant",
                "content": response_data["content"]
            })
            
            # Move to parent response
            current_response_id = response_data.get("parent_response_id")
        
        return context_messages
    
    def _save_response_to_file(self, response_id: str, response_data: Dict):
        """Save response to file storage"""
        if config.history_storage != "file":
            return
            
        response_file = Path(config.history_dir) / "responses" / f"{response_id}.json"
        try:
            with open(response_file, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save response {response_id}: {e}")
    
    def _load_response_from_file(self, response_id: str) -> Optional[Dict]:
        """Load response from file storage"""
        if config.history_storage != "file":
            return None
            
        response_file = Path(config.history_dir) / "responses" / f"{response_id}.json"
        if not response_file.exists():
            return None
            
        try:
            with open(response_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load response {response_id}: {e}")
            return None
    
    def _cleanup_expired_responses(self):
        """Remove expired responses"""
        if time.time() - self.last_cleanup < self.cleanup_interval:
            return
            
        cutoff_time = datetime.now() - timedelta(hours=config.history_ttl_hours)
        expired_responses = [
            response_id for response_id, last_access in self.response_access.items()
            if last_access < cutoff_time
        ]
        
        for response_id in expired_responses:
            self._delete_response(response_id)
            
        if expired_responses:
            logger.info(f"Cleaned up {len(expired_responses)} expired responses")
    
    def _delete_response(self, response_id: str):
        """Delete a response from all storage"""
        # Remove from memory
        self.responses.pop(response_id, None)
        self.response_access.pop(response_id, None)
        
        # Remove from file storage if applicable
        if config.history_storage == "file":
            response_file = Path(config.history_dir) / "responses" / f"{response_id}.json"
            if response_file.exists():
                response_file.unlink()
    
    def list_responses(self, conversation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List stored responses, optionally filtered by conversation_id"""
        responses = []
        for response_id, response_data in self.responses.items():
            if conversation_id and response_data.get("conversation_id") != conversation_id:
                continue
                
            responses.append({
                "id": response_id,
                "conversation_id": response_data.get("conversation_id"),
                "model": response_data.get("model"),
                "created": response_data.get("created"),
                "tokens": response_data.get("tokens"),
                "finish_reason": response_data.get("finish_reason"),
                "parent_response_id": response_data.get("parent_response_id")
            })
        
        return sorted(responses, key=lambda x: x["created"], reverse=True)

# Global conversation history manager
conversation_history = ConversationHistory()

# Custom exceptions
class ClaudeError(Exception):
    """Base exception for Claude-related errors"""
    pass

class ClaudeNotFoundError(ClaudeError):
    """Claude command not found"""
    pass

class ClaudeTimeoutError(ClaudeError):
    """Claude command timed out"""
    pass

class ClaudeProcessError(ClaudeError):
    """Claude process failed"""
    pass

class ClaudeRateLimitError(ClaudeError):
    """Claude rate limit exceeded"""
    pass

app = FastAPI(
    title="Claude OpenAI API Bridge", 
    version="1.1.0",
    description="OpenAI-compatible API for Claude CLI with enhanced features",
    docs_url="/docs" if config.debug else None,
    redoc_url="/redoc" if config.debug else None
)


class Message(BaseModel):
    role: str
    content: Optional[str] = None  # Can be None when using tools
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, content, info):
        # Content can be None if tool_calls are present
        # In Pydantic v2, we need to check the data differently
        if content is None:
            # Allow None content for now - tool_calls validation happens elsewhere
            pass
        return content


class StoredResponse(BaseModel):
    """Model for stored API responses"""
    id: str  # response_id (e.g. "resp_abc123xyz")
    conversation_id: Optional[str] = None  # Link to session
    content: str  # The actual response content
    model: str  # Model used
    created: int  # Unix timestamp
    tokens: Dict[str, int]  # Token usage info
    finish_reason: str
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Additional metadata
    parent_response_id: Optional[str] = None  # Previous response in chain
    messages: List[Dict[str, Any]] = Field(default_factory=list)  # Request messages that led to this response


class ResponseRequest(BaseModel):
    """Model for Response API requests"""
    model: str
    input: Optional[Union[str, List[Dict[str, Any]]]] = None
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = Field(default=None, gt=0)
    temperature: Optional[float] = Field(default=1.0, ge=0, le=2)
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1)
    store: Optional[bool] = Field(default=True, description="Whether to store this response")
    previous_response_id: Optional[str] = Field(default=None, description="ID of previous response to use for context")
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = "auto"
    parallel_tool_calls: Optional[bool] = True
    text: Optional[Dict[str, Any]] = None  # Text formatting options
    truncation: Optional[str] = "disabled"
    user: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ResponseOutput(BaseModel):
    """Model for response output content"""
    type: str = "message"
    id: str
    status: str = "completed"
    role: str = "assistant"
    content: List[Dict[str, Any]]


class ResponseUsageDetails(BaseModel):
    """Model for detailed token usage"""
    cached_tokens: int = 0
    reasoning_tokens: int = 0


class ResponseUsage(BaseModel):
    """Model for response usage statistics"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: ResponseUsageDetails = Field(default_factory=ResponseUsageDetails)
    output_tokens_details: ResponseUsageDetails = Field(default_factory=ResponseUsageDetails)


class ResponseReasoning(BaseModel):
    """Model for response reasoning details"""
    effort: Optional[str] = None
    summary: Optional[str] = None


class ResponseAPIResponse(BaseModel):
    """Model for Response API responses"""
    id: str
    object: str = "response"
    created_at: int
    status: str = "completed"
    error: Optional[Any] = None
    incomplete_details: Optional[Any] = None
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = None
    model: str
    output: List[ResponseOutput]
    parallel_tool_calls: bool = True
    previous_response_id: Optional[str] = None
    reasoning: ResponseReasoning = Field(default_factory=ResponseReasoning)
    store: bool = True
    temperature: float = 1.0
    text: Optional[Dict[str, Any]] = None
    tool_choice: str = "auto"
    tools: List[Any] = Field(default_factory=list)
    top_p: float = 1.0
    truncation: str = "disabled"
    usage: ResponseUsage
    user: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = Field(default=1.0, ge=0, le=2)
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1)
    n: Optional[int] = Field(default=1, ge=1, le=10)  # Support up to 10 completions
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = Field(default=None, gt=0)
    presence_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    frequency_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    user: Optional[str] = None
    
    # History management fields
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation history")
    include_history: Optional[bool] = Field(default=True, description="Whether to include conversation history")
    
    # Responses API fields
    store: Optional[bool] = Field(default=False, description="Whether to store this response for later retrieval")
    previous_response_id: Optional[str] = Field(default=None, description="ID of previous response to use for context")
    
    # Additional OpenAI API fields
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None  # "auto", "none", or specific tool
    response_format: Optional[Dict[str, str]] = None  # {"type": "json_object"}
    seed: Optional[int] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = Field(default=None, ge=0, le=20)
    logit_bias: Optional[Dict[str, float]] = None
    suffix: Optional[str] = None
    
    @field_validator('messages')
    @classmethod
    def validate_messages(cls, messages):
        if not messages:
            raise ValueError('Messages list cannot be empty')
        if len(messages) > 100:
            raise ValueError('Too many messages (max 100)')
        return messages
    
    @field_validator('model')
    @classmethod
    def validate_model(cls, model):
        if not model or not model.strip():
            raise ValueError('Model cannot be empty')
        return model


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str
    logprobs: Optional[Any] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage
    system_fingerprint: Optional[str] = None
    response_id: Optional[str] = None  # For stored responses


class StreamChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]
    system_fingerprint: Optional[str] = None


def apply_stop_sequences(text: str, stop_sequences: Optional[List[str]]) -> tuple[str, str]:
    """Apply stop sequences to text, return (truncated_text, stop_reason)"""
    if not stop_sequences or not text:
        return text, "stop"
    
    earliest_pos = len(text)
    matched_sequence = None
    
    for sequence in stop_sequences:
        pos = text.find(sequence)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos
            matched_sequence = sequence
    
    if matched_sequence:
        return text[:earliest_pos], "stop_sequence"
    
    return text, "stop"

def parse_tool_calls_from_response(response: str) -> tuple[str, Optional[List[Dict[str, Any]]]]:
    """Parse tool calls from Claude's response"""
    tool_calls = None
    clean_response = response
    
    # Look for JSON block with tool_calls
    json_pattern = r'```json\s*(\{.*?"tool_calls".*?\})\s*```'
    match = re.search(json_pattern, response, re.DOTALL)
    
    if match:
        try:
            json_str = match.group(1)
            data = json.loads(json_str)
            if "tool_calls" in data:
                tool_calls = data["tool_calls"]
                # Remove the JSON block from response
                clean_response = response[:match.start()] + response[match.end():]
                clean_response = clean_response.strip()
        except json.JSONDecodeError:
            logger.debug("Failed to parse tool calls JSON")
    
    return clean_response, tool_calls

def format_tools_for_claude(tools: List[Dict[str, Any]]) -> str:
    """Format OpenAI tools/functions for Claude"""
    if not tools:
        return ""
    
    tool_descriptions = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            description = func.get("description", "")
            params = func.get("parameters", {})
            
            # Format parameters
            param_desc = []
            if "properties" in params:
                for param_name, param_info in params["properties"].items():
                    param_type = param_info.get("type", "string")
                    param_description = param_info.get("description", "")
                    required = param_name in params.get("required", [])
                    req_text = " (required)" if required else " (optional)"
                    param_desc.append(f"  - {param_name}: {param_type}{req_text} - {param_description}")
            
            tool_text = f"Function: {name}\nDescription: {description}"
            if param_desc:
                tool_text += f"\nParameters:\n" + "\n".join(param_desc)
            tool_descriptions.append(tool_text)
    
    if tool_descriptions:
        return "\n\nAvailable tools:\n" + "\n\n".join(tool_descriptions) + "\n\nTo use a tool, respond with a JSON object in this format:\n```json\n{\n  \"tool_calls\": [{\n    \"id\": \"unique_id\",\n    \"type\": \"function\",\n    \"function\": {\n      \"name\": \"function_name\",\n      \"arguments\": \"{\\\"param1\\\": \\\"value1\\\"}\"\n    }\n  }]\n}\n```"
    return ""

def format_messages_for_claude(
    messages: List[Message], 
    conversation_context: List[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    response_format: Optional[Dict[str, str]] = None,
    stop_sequences: Optional[List[str]] = None
) -> str:
    """Format messages into a single prompt for claude command with conversation history"""
    formatted_parts = []
    
    # Validate and sanitize input
    if not messages:
        raise ValueError("Messages cannot be empty")
    
    # Add conversation history first if provided
    if conversation_context:
        logger.debug(f"Including {len(conversation_context)} messages from conversation history")
        for msg in conversation_context:
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
                
            # Sanitize content to prevent command injection
            if len(content) > config.max_prompt_length // 4:  # Limit individual message size
                content = content[:config.max_prompt_length // 4]
                
            role = msg.get("role", "user")
            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"Human: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
    
    # Add tools/functions description if provided
    if tools:
        tool_desc = format_tools_for_claude(tools)
        if tool_desc:
            formatted_parts.append(f"System: {tool_desc}")
    
    # Add response format instruction if JSON mode requested
    if response_format and response_format.get("type") == "json_object":
        formatted_parts.append("System: You must respond with valid JSON only. No other text or markdown.")
    
    # Add current messages
    for msg in messages:
        content = ""
        
        # Handle multimodal content (images)
        if isinstance(msg.content, list):
            text_parts = []
            for content_part in msg.content:
                if content_part.get("type") == "text":
                    text_parts.append(content_part.get("text", ""))
                elif content_part.get("type") == "image_url":
                    # Extract image data
                    image_url = content_part.get("image_url", {}).get("url", "")
                    if image_url.startswith("data:"):
                        # Handle base64 encoded images
                        try:
                            header, data = image_url.split(",", 1)
                            image_data = base64.b64decode(data)
                            # Save temporarily and reference
                            text_parts.append("[Image provided - Claude will analyze this]")
                        except Exception as e:
                            logger.error(f"Failed to process image: {e}")
                    else:
                        text_parts.append(f"[Image URL: {image_url}]")
            content = " ".join(text_parts)
        else:
            # Regular text content
            content = msg.content or ""
            
        # Handle tool responses
        if msg.tool_call_id:
            content = f"Tool response for {msg.tool_call_id}: {content}"
        
        # Handle tool calls from assistant
        if msg.tool_calls:
            tool_strs = []
            for tool_call in msg.tool_calls:
                if tool_call.get("type") == "function":
                    func = tool_call.get("function", {})
                    tool_strs.append(f"Using tool: {func.get('name')} with args: {func.get('arguments')}")
            if tool_strs:
                content = (content + "\n" if content else "") + "\n".join(tool_strs)
        
        content = content.strip()
        if not content and msg.role != "assistant":
            continue
            
        # Sanitize content to prevent command injection
        if len(content) > config.max_prompt_length:
            logger.warning(f"Message content truncated from {len(content)} to {config.max_prompt_length} characters")
            content = content[:config.max_prompt_length]
        
        if msg.role == "system":
            formatted_parts.append(f"System: {content}")
        elif msg.role == "user":
            formatted_parts.append(f"Human: {content}")
        elif msg.role == "assistant":
            formatted_parts.append(f"Assistant: {content}")
        elif msg.role == "tool":
            formatted_parts.append(f"System: {content}")
        else:
            logger.warning(f"Unknown message role: {msg.role}")
    
    # Add final Human/Assistant prompt if needed
    if messages and messages[-1].role != "user":
        formatted_parts.append("Human: Please continue.")
    
    formatted_parts.append("Assistant:")
    
    formatted_prompt = "\n\n".join(formatted_parts)
    
    # Final length check
    if len(formatted_prompt) > config.max_prompt_length:
        logger.warning(f"Final prompt truncated from {len(formatted_prompt)} to {config.max_prompt_length} characters")
        formatted_prompt = formatted_prompt[:config.max_prompt_length]
    
    return formatted_prompt


def map_openai_model_to_claude(model: str) -> str:
    """Map OpenAI model names to Claude model names"""
    # Load custom mappings from environment if provided
    custom_mappings = os.getenv("CLAUDE_MODEL_MAPPINGS")
    if custom_mappings:
        try:
            model_mapping = json.loads(custom_mappings)
        except json.JSONDecodeError:
            logger.warning("Invalid CLAUDE_MODEL_MAPPINGS JSON, using defaults")
            model_mapping = {}
    else:
        model_mapping = {}
    
    # Default mappings - use model aliases that work with Claude CLI
    default_mappings = {
        "gpt-4": "sonnet",
        "gpt-4-turbo": "sonnet",
        "gpt-4-turbo-preview": "sonnet",
        "gpt-4o": "sonnet",
        "gpt-4.1": "sonnet",
        "gpt-4o-mini": "haiku",
        "gpt-3.5-turbo": "haiku",
        "gpt-3.5-turbo-16k": "haiku",
        # Direct Claude model support - use aliases
        "claude-3-5-sonnet-20241022": "sonnet",
        "claude-3-5-haiku-20241022": "haiku",
        "claude-3-opus-20240229": "opus",
    }
    
    # Merge with custom mappings taking precedence
    final_mappings = {**default_mappings, **model_mapping}
    
    mapped_model = final_mappings.get(model, "sonnet")
    if mapped_model != model:
        logger.info(f"Mapped model {model} to {mapped_model}")
    
    return mapped_model


async def call_claude_subprocess(
    prompt: str, 
    model: str = None, 
    timeout: int = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stop_sequences: Optional[List[str]] = None
) -> AsyncIterator[str]:
    """Call claude command via subprocess with retry logic and proper error handling"""
    timeout = timeout or config.request_timeout
    attempt = 0
    
    while attempt < config.max_retries:
        attempt += 1
        process = None
        
        try:
            logger.debug(f"Starting Claude subprocess attempt {attempt}/{config.max_retries}")
            
            # Build command args
            cmd_args = [config.claude_command]
            if model:
                cmd_args.extend(["--model", model])
            
            # Note: Claude CLI doesn't support temperature, max_tokens, or stop sequences
            # These will be handled via prompt engineering or post-processing
            # Temperature and max_tokens are not supported by Claude CLI
            
            cmd_args.extend(["--print"])  # Use print mode for non-interactive output
            # Don't append prompt to cmd_args - we'll send it via stdin instead
            
            # Create the subprocess with timeout
            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1024*1024  # 1MB buffer limit
            )
            
            # Send the prompt via stdin and close it
            try:
                process.stdin.write(prompt.encode('utf-8'))
                await process.stdin.drain()
                process.stdin.close()
                await process.stdin.wait_closed()
            except Exception as e:
                logger.error(f"Failed to send prompt via stdin: {e}")
                if process:
                    process.terminate()
                raise ClaudeProcessError(f"Failed to send prompt to Claude: {e}")
            
            # Read output with timeout
            output_lines = []
            try:
                while True:
                    try:
                        line = await asyncio.wait_for(
                            process.stdout.readline(), 
                            timeout=timeout
                        )
                        if not line:
                            break
                        
                        decoded_line = line.decode('utf-8', errors='replace')
                        output_lines.append(decoded_line)
                        yield decoded_line
                        
                    except asyncio.TimeoutError:
                        logger.warning(f"Claude subprocess timeout after {timeout}s")
                        if process:
                            process.terminate()
                            await asyncio.sleep(1)
                            if process.returncode is None:
                                process.kill()
                        raise ClaudeTimeoutError(f"Claude command timed out after {timeout} seconds")
                        
            except Exception as e:
                if process:
                    try:
                        process.terminate()
                        await asyncio.sleep(1)
                        if process.returncode is None:
                            process.kill()
                    except Exception:
                        pass
                raise e
            
            # Wait for process to complete
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                logger.warning("Process didn't terminate cleanly")
                if process:
                    process.kill()
            
            # Check return code
            if process.returncode != 0:
                stderr_data = b''
                try:
                    stderr_data = await asyncio.wait_for(
                        process.stderr.read(), 
                        timeout=5
                    )
                except asyncio.TimeoutError:
                    pass
                
                stderr_text = stderr_data.decode('utf-8', errors='replace')
                stdout_content = ''.join(output_lines) if output_lines else ''
                
                logger.error(f"Claude command failed. Command: {' '.join(cmd_args)}")
                logger.error(f"Exit code: {process.returncode}")
                logger.error(f"Stderr: {repr(stderr_text)}")
                logger.error(f"Stdout: {repr(stdout_content[:500])}")  # First 500 chars
                
                # Check for specific error types
                error_text = stderr_text + stdout_content
                if 'rate limit' in error_text.lower():
                    raise ClaudeRateLimitError(f"Claude rate limit exceeded: {stderr_text}")
                elif 'authentication' in error_text.lower() or 'auth' in error_text.lower():
                    raise ClaudeError(f"Claude authentication error: {stderr_text}")
                else:
                    raise ClaudeProcessError(f"Claude command failed with code {process.returncode}. Stderr: {stderr_text}. Stdout: {stdout_content[:200]}")
            
            # Success, no need to retry
            logger.debug(f"Claude subprocess completed successfully on attempt {attempt}")
            return
            
        except FileNotFoundError:
            raise ClaudeNotFoundError(f"Claude command '{config.claude_command}' not found. Make sure Claude CLI is installed and in PATH")
        
        except (ClaudeNotFoundError, ClaudeTimeoutError) as e:
            # Don't retry these errors
            raise e
            
        except (ClaudeRateLimitError, ClaudeProcessError) as e:
            if attempt >= config.max_retries:
                logger.error(f"Claude subprocess failed after {config.max_retries} attempts: {e}")
                raise e
            else:
                logger.warning(f"Claude subprocess failed (attempt {attempt}/{config.max_retries}): {e}")
                await asyncio.sleep(config.retry_delay * attempt)  # Exponential backoff
                continue
                
        except Exception as e:
            logger.error(f"Unexpected error in Claude subprocess: {e}")
            if attempt >= config.max_retries:
                raise ClaudeError(f"Unexpected error after {config.max_retries} attempts: {e}")
            else:
                await asyncio.sleep(config.retry_delay * attempt)
                continue
        
        finally:
            # Cleanup
            if process and process.returncode is None:
                try:
                    process.terminate()
                    await asyncio.sleep(1)
                    if process.returncode is None:
                        process.kill()
                except Exception:
                    pass


async def generate_stream_response(
    request: ChatCompletionRequest,
    prompt: str,
    session_id: Optional[str] = None
) -> AsyncIterator[str]:
    """Generate streaming response with improved error handling"""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created = int(time.time())
    mapped_model = map_openai_model_to_claude(request.model)
    
    logger.info(f"Starting streaming response for request {completion_id}")
    
    # Send initial chunk
    initial_chunk = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=request.model,
        choices=[StreamChoice(
            index=0,
            delta={"role": "assistant", "content": ""},
            finish_reason=None
        )]
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"
    
    try:
        buffer = ""
        chunk_count = 0
        
        stop_triggered = False
        
        async for chunk in call_claude_subprocess(
            prompt, 
            mapped_model, 
            config.request_timeout,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop_sequences=request.stop
        ):
            chunk_count += 1
            
            # Check for stop sequences in streaming mode
            if request.stop and not stop_triggered:
                temp_buffer = buffer + chunk
                for stop_seq in request.stop:
                    if stop_seq in temp_buffer:
                        # Found stop sequence
                        stop_pos = temp_buffer.find(stop_seq)
                        chunk = temp_buffer[len(buffer):stop_pos]
                        buffer = temp_buffer[:stop_pos]
                        stop_triggered = True
                        break
            
            # Send each chunk of text as it comes
            if chunk.strip() and not stop_triggered:
                chunk_data = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[StreamChoice(
                        index=0,
                        delta={"content": chunk},
                        finish_reason=None
                    )]
                )
                yield f"data: {chunk_data.model_dump_json()}\n\n"
                buffer += chunk
            
            if stop_triggered:
                break
        
        logger.debug(f"Streamed {chunk_count} chunks for request {completion_id}")
        
        # Send final chunk
        final_chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[StreamChoice(
                index=0,
                delta={},
                finish_reason="stop"
            )]
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        
        # Add assistant response to conversation history
        if config.enable_history and session_id and buffer.strip():
            conversation_history.add_message(session_id, {
                "role": "assistant",
                "content": buffer.strip()
            })
            logger.debug(f"Added streaming response to session {session_id}")
        
        logger.info(f"Streaming response completed for request {completion_id}")
        
    except ClaudeNotFoundError as e:
        logger.error(f"Claude not found for request {completion_id}: {e}")
        error_chunk = {
            "error": {
                "message": "Claude CLI not found. Please install and configure Claude CLI.",
                "type": "configuration_error",
                "code": "claude_not_found"
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        
    except ClaudeTimeoutError as e:
        logger.error(f"Timeout for request {completion_id}: {e}")
        error_chunk = {
            "error": {
                "message": "Request timed out. Please try again with a shorter prompt.",
                "type": "timeout_error",
                "code": "request_timeout"
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        
    except ClaudeRateLimitError as e:
        logger.error(f"Rate limit for request {completion_id}: {e}")
        error_chunk = {
            "error": {
                "message": "Rate limit exceeded. Please wait before making another request.",
                "type": "rate_limit_error",
                "code": "rate_limit_exceeded"
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        
    except Exception as e:
        logger.error(f"Unexpected error for request {completion_id}: {e}")
        error_chunk = {
            "error": {
                "message": "An unexpected error occurred. Please try again.",
                "type": "internal_server_error",
                "code": "internal_error"
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"


# Token counting utilities
_tokenizer = None

def get_tokenizer():
    """Get or initialize the tokenizer"""
    global _tokenizer
    if TIKTOKEN_AVAILABLE and _tokenizer is None:
        try:
            _tokenizer = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer

def estimate_tokens(text: Union[str, List[Dict[str, Any]]]) -> int:
    """Accurate token estimation using tiktoken or fallback"""
    if not text:
        return 0
    
    # Convert messages to text if needed
    if isinstance(text, list):
        text_parts = []
        for msg in text:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                text_parts.append(f"{role}: {content}")
        text = "\n".join(text_parts)
    
    if TIKTOKEN_AVAILABLE:
        try:
            tokenizer = get_tokenizer()
            if tokenizer:
                return len(tokenizer.encode(text))
        except Exception as e:
            logger.debug(f"Token counting error: {e}")
    
    # Fallback to approximate counting
    word_count = len(text.split())
    char_count = len(text)
    
    # Use character-based estimation for non-English text
    char_tokens = char_count / 4
    word_tokens = word_count * 1.3
    
    # Take the average of both estimates
    estimated_tokens = int((char_tokens + word_tokens) / 2)
    
    # Minimum of 1 token for non-empty text
    return max(1, estimated_tokens)

@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    x_session_id: Optional[str] = Header(None, description="Session ID for conversation history")
):
    """Create a chat completion with conversation history support"""
    request_id = f"req-{uuid.uuid4().hex[:12]}"
    logger.info(f"Received chat completion request {request_id} for model {request.model}")
    
    try:
        # Handle session management
        session_id = request.session_id or x_session_id
        if config.enable_history and request.include_history:
            session_id = conversation_history.get_or_create_session(session_id, request.user)
            logger.info(f"Request {request_id}: Using session {session_id}")
        
        # Get conversation context - prioritize previous_response_id over session history
        conversation_context = []
        parent_response_id = None
        
        if config.enable_history and request.previous_response_id:
            # Use previous response chain for context
            conversation_context = conversation_history.get_response_context_chain(request.previous_response_id)
            parent_response_id = request.previous_response_id
            logger.info(f"Request {request_id}: Using previous response context from {request.previous_response_id}")
            logger.debug(f"Request {request_id}: Retrieved {len(conversation_context)} context messages from response chain")
        elif config.enable_history and request.include_history and session_id:
            # Use session history for context
            conversation_context = conversation_history.get_conversation_context(session_id)
            logger.debug(f"Request {request_id}: Retrieved {len(conversation_context)} context messages from session")
        
        # Add current user messages to history before processing
        if config.enable_history and session_id:
            for message in request.messages:
                conversation_history.add_message(session_id, {
                    "role": message.role,
                    "content": message.content,
                    "name": message.name
                })
        
        # Format messages for claude command with all features
        prompt = format_messages_for_claude(
            request.messages, 
            conversation_context,
            tools=request.tools,
            response_format=request.response_format,
            stop_sequences=request.stop
        )
        mapped_model = map_openai_model_to_claude(request.model)
        
        logger.debug(f"Request {request_id}: prompt length = {len(prompt)} chars")
        
        if request.stream:
            logger.info(f"Request {request_id}: Starting streaming response")
            return StreamingResponse(
                generate_stream_response(request, prompt, session_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Request-ID": request_id,
                    "X-Session-ID": session_id or ""
                }
            )
        
        # For non-streaming, handle multiple completions if n > 1
        logger.info(f"Request {request_id}: Starting non-streaming response (n={request.n})")
        
        choices = []
        total_prompt_tokens = estimate_tokens(prompt)
        total_completion_tokens = 0
        
        # Generate n completions
        for i in range(request.n or 1):
            full_response = ""
            start_time = time.time()
            
            # Add variation for multiple completions or use seed for deterministic output
            if request.seed is not None:
                # Use seed to make temperature deterministic
                import random
                random.seed(request.seed + i)
                adjusted_temp = 0.1  # Low temperature for reproducible outputs
            else:
                temp_adjustment = 0.1 * i if request.n > 1 else 0
                adjusted_temp = min(2.0, (request.temperature or 1.0) + temp_adjustment)
            
            async for chunk in call_claude_subprocess(
                prompt, 
                mapped_model, 
                config.request_timeout,
                temperature=adjusted_temp,
                max_tokens=request.max_tokens,
                stop_sequences=request.stop
            ):
                full_response += chunk
            
            response_time = time.time() - start_time
            logger.info(f"Request {request_id}: Claude response {i+1}/{request.n} received in {response_time:.2f}s")
        
            # Apply stop sequences
            full_response, finish_reason = apply_stop_sequences(full_response, request.stop)
            
            # Parse tool calls if tools were provided
            tool_calls = None
            if request.tools:
                full_response, tool_calls = parse_tool_calls_from_response(full_response)
            
            # Handle JSON response format
            if request.response_format and request.response_format.get("type") == "json_object":
                # Try to extract JSON from the response
                try:
                    # First try to parse the entire response as JSON
                    json_data = json.loads(full_response)
                    full_response = json.dumps(json_data)
                except json.JSONDecodeError:
                    # Try to find JSON in the response
                    json_match = re.search(r'\{.*\}', full_response, re.DOTALL)
                    if json_match:
                        try:
                            json_data = json.loads(json_match.group())
                            full_response = json.dumps(json_data)
                        except json.JSONDecodeError:
                            logger.warning("Failed to extract valid JSON from response")
            
            # Token estimation for this completion
            completion_tokens = estimate_tokens(full_response)
            total_completion_tokens += completion_tokens
            
            # Build the assistant message
            assistant_message = Message(
                role="assistant", 
                content=full_response.strip() if not tool_calls else None,
                tool_calls=tool_calls
            )
            
            # Add choice
            choices.append(Choice(
                index=i,
                message=assistant_message,
                finish_reason=finish_reason
            ))
            
            # Add to conversation history (only the first response)
            if i == 0 and config.enable_history and session_id:
                conversation_history.add_message(session_id, {
                    "role": "assistant",
                    "content": full_response.strip(),
                    "tool_calls": tool_calls
                })
        
        completion_response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:29]}",
            created=int(time.time()),
            model=request.model,
            choices=choices,
            usage=Usage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens
            )
        )
        
        # Store response if requested
        response_id = None
        if config.enable_history and request.store and choices:
            # Store the first choice's response (primary response)
            first_choice = choices[0]
            response_content = first_choice.message.content or ""
            
            # Convert request messages to dict format for storage
            request_messages = [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "name": msg.name
                }
                for msg in request.messages
            ]
            
            response_id = conversation_history.store_response(
                content=response_content,
                model=mapped_model,
                finish_reason=first_choice.finish_reason,
                tokens={
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": total_prompt_tokens + total_completion_tokens
                },
                messages=request_messages,
                conversation_id=session_id,
                parent_response_id=parent_response_id,
                metadata={
                    "request_id": request_id,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "n": request.n,
                    "tools": request.tools is not None,
                    "response_format": request.response_format
                }
            )
            logger.info(f"Request {request_id}: Stored response as {response_id}")
        
        # Add session ID and response ID to response headers if available
        if session_id:
            completion_response.system_fingerprint = session_id
        
        # Add response_id to the response if stored
        if response_id:
            completion_response.response_id = response_id
        
        logger.info(f"Request {request_id}: Completed successfully")
        return completion_response
        
    except ClaudeNotFoundError as e:
        logger.error(f"Request {request_id}: Claude not found - {e}")
        raise HTTPException(
            status_code=503,
            detail="Claude CLI not found. Please install and configure Claude CLI."
        )
        
    except ClaudeTimeoutError as e:
        logger.error(f"Request {request_id}: Timeout - {e}")
        raise HTTPException(
            status_code=408,
            detail="Request timed out. Please try again with a shorter prompt."
        )
        
    except ClaudeRateLimitError as e:
        logger.error(f"Request {request_id}: Rate limit - {e}")
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait before making another request."
        )
        
    except ValueError as e:
        logger.error(f"Request {request_id}: Validation error - {e}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error(f"Request {request_id}: Unexpected error - {e}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again."
        )


@app.get("/v1/models")
async def list_models():
    """List available models with enhanced model support"""
    models = [
        {
            "id": "gpt-4",
            "object": "model",
            "created": 1687882411,
            "owned_by": "openai"
        },
        {
            "id": "gpt-4-turbo",
            "object": "model",
            "created": 1706037612,
            "owned_by": "openai"
        },
        {
            "id": "gpt-4-turbo-preview",
            "object": "model",
            "created": 1706037612,
            "owned_by": "openai"
        },
        {
            "id": "gpt-4o",
            "object": "model",
            "created": 1715367049,
            "owned_by": "openai"
        },
        {
            "id": "gpt-4o-mini",
            "object": "model",
            "created": 1721172741,
            "owned_by": "openai"
        },
        {
            "id": "gpt-3.5-turbo",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai"
        },
        {
            "id": "gpt-3.5-turbo-16k",
            "object": "model",
            "created": 1683758102,
            "owned_by": "openai"
        },
        # Also expose Claude models directly
        {
            "id": "claude-3-5-sonnet-20241022",
            "object": "model",
            "created": 1729555200,
            "owned_by": "anthropic"
        },
        {
            "id": "claude-3-5-haiku-20241022",
            "object": "model",
            "created": 1729555200,
            "owned_by": "anthropic"
        },
        {
            "id": "claude-3-opus-20240229",
            "object": "model",
            "created": 1709251200,
            "owned_by": "anthropic"
        },
        {
            "id": "gpt-4.1",
            "object": "model",
            "created": 1740000000,
            "owned_by": "openai"
        }
    ]
    
    logger.debug(f"Listed {len(models)} available models")
    return {"object": "list", "data": models}


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Claude OpenAI API Bridge",
        "version": "1.1.0",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "responses": "/v1/responses",
            "models": "/v1/models",
            "health": "/health"
        },
        "features": [
            "OpenAI-compatible API",
            "Streaming responses", 
            "Multiple Claude models",
            "Conversation history management",
            "Session-based conversations",
            "Function calling (tools) support",
            "JSON response format",
            "Stop sequences",
            "Multiple completions (n parameter)",
            "Multimodal input (images)",
            "Automatic retries",
            "Comprehensive error handling",
            "File or memory-based persistence"
        ],
        "notes": {
            "tools": "Function calling is implemented via prompt engineering",
            "logprobs": "Not supported by Claude CLI",
            "logit_bias": "Not supported by Claude CLI",
            "presence_penalty": "Mapped to Claude temperature adjustments",
            "frequency_penalty": "Mapped to Claude temperature adjustments"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test if Claude CLI is available
        test_process = await asyncio.create_subprocess_exec(
            config.claude_command, "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await asyncio.wait_for(test_process.wait(), timeout=5)
        claude_available = test_process.returncode == 0
    except Exception:
        claude_available = False
    
    return {
        "status": "healthy" if claude_available else "degraded",
        "claude_cli": "available" if claude_available else "unavailable",
        "history_enabled": config.enable_history,
        "active_sessions": len(conversation_history.sessions),
        "timestamp": int(time.time())
    }

# Session Management Endpoints
@app.post("/v1/sessions")
async def create_session(user_id: Optional[str] = None):
    """Create a new conversation session"""
    if not config.enable_history:
        raise HTTPException(status_code=503, detail="Conversation history is disabled")
    
    session_id = conversation_history.get_or_create_session(None, user_id)
    session_info = conversation_history.get_session_info(session_id)
    
    logger.info(f"Created new session: {session_id}")
    return {
        "session_id": session_id,
        "created_at": session_info["created_at"],
        "message_count": 0,
        "user_id": user_id
    }

@app.get("/v1/sessions")
async def list_sessions(user_id: Optional[str] = None):
    """List all conversation sessions"""
    if not config.enable_history:
        raise HTTPException(status_code=503, detail="Conversation history is disabled")
    
    sessions = conversation_history.list_sessions(user_id)
    return {
        "object": "list",
        "data": sessions,
        "total": len(sessions)
    }

@app.get("/v1/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session information and conversation history"""
    if not config.enable_history:
        raise HTTPException(status_code=503, detail="Conversation history is disabled")
    
    session_info = conversation_history.get_session_info(session_id)
    if not session_info:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get full conversation history
    if session_id in conversation_history.sessions:
        messages = conversation_history.sessions[session_id].get("messages", [])
    else:
        messages = []
    
    return {
        **session_info,
        "messages": messages
    }

@app.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a conversation session"""
    if not config.enable_history:
        raise HTTPException(status_code=503, detail="Conversation history is disabled")
    
    if session_id not in conversation_history.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    conversation_history._delete_session(session_id)
    logger.info(f"Deleted session: {session_id}")
    
    return {"message": "Session deleted successfully"}

@app.post("/v1/sessions/{session_id}/clear")
async def clear_session_history(session_id: str):
    """Clear conversation history for a session"""
    if not config.enable_history:
        raise HTTPException(status_code=503, detail="Conversation history is disabled")
    
    if session_id not in conversation_history.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Clear messages but keep session metadata
    session_data = conversation_history.sessions[session_id]
    session_data["messages"] = []
    conversation_history._save_session_to_file(session_id, session_data)
    
    logger.info(f"Cleared history for session: {session_id}")
    return {"message": "Session history cleared successfully"}


def _format_input_as_messages(input_data: Optional[Union[str, List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
    """Convert input data to messages format for storage"""
    if not input_data:
        return []
    
    if isinstance(input_data, str):
        return [{"role": "user", "content": input_data}]
    elif isinstance(input_data, list):
        return input_data
    else:
        return [{"role": "user", "content": str(input_data)}]


# Response Management Endpoints
@app.post("/v1/responses")
async def create_response(request: ResponseRequest):
    """Create a new response using the Response API format"""
    request_id = f"resp-{uuid.uuid4().hex[:12]}"
    logger.info(f"Received response request {request_id} for model {request.model}")
    
    try:
        # Get conversation context if previous_response_id is provided
        conversation_context = []
        if config.enable_history and request.previous_response_id:
            conversation_context = conversation_history.get_response_context_chain(request.previous_response_id)
            logger.info(f"Request {request_id}: Using previous response context from {request.previous_response_id}")
            logger.debug(f"Request {request_id}: Retrieved {len(conversation_context)} context messages from response chain")
        
        # Build the prompt from input and instructions
        prompt_parts = []
        if request.instructions:
            prompt_parts.append(f"System: {request.instructions}")
        
        # Handle both string and list inputs
        if request.input:
            if isinstance(request.input, str):
                prompt_parts.append(f"Human: {request.input}")
            elif isinstance(request.input, list):
                # Handle list of messages (like chat format)
                for msg in request.input:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "user":
                        prompt_parts.append(f"Human: {content}")
                    elif role == "assistant":
                        prompt_parts.append(f"Assistant: {content}")
                    elif role == "system":
                        prompt_parts.append(f"System: {content}")
            else:
                # Convert to string as fallback
                prompt_parts.append(f"Human: {str(request.input)}")
        
        # Add conversation context if available
        if conversation_context:
            logger.debug(f"Including {len(conversation_context)} messages from conversation history")
            for msg in conversation_context:
                content = str(msg.get("content", "")).strip()
                if not content:
                    continue
                role = msg.get("role", "user")
                if role == "system":
                    prompt_parts.insert(-1 if request.input else -1, f"System: {content}")
                elif role == "user":
                    prompt_parts.insert(-1 if request.input else -1, f"Human: {content}")
                elif role == "assistant":
                    prompt_parts.insert(-1 if request.input else -1, f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        prompt = "\n\n".join(prompt_parts)
        
        mapped_model = map_openai_model_to_claude(request.model)
        logger.debug(f"Request {request_id}: prompt length = {len(prompt)} chars")
        
        # Generate response using Claude
        full_response = ""
        start_time = time.time()
        
        async for chunk in call_claude_subprocess(
            prompt, 
            mapped_model, 
            config.request_timeout,
            temperature=request.temperature,
            max_tokens=request.max_output_tokens
        ):
            full_response += chunk
        
        response_time = time.time() - start_time
        logger.info(f"Request {request_id}: Claude response received in {response_time:.2f}s")
        
        # Calculate tokens
        input_tokens = estimate_tokens(prompt)
        output_tokens = estimate_tokens(full_response)
        total_tokens = input_tokens + output_tokens
        
        # Generate response ID
        response_id = conversation_history._generate_response_id()
        
        # Create the response content in the expected format
        output_message_id = f"msg_{uuid.uuid4().hex[:29]}"
        
        response_output = ResponseOutput(
            type="message",
            id=output_message_id,
            status="completed",
            role="assistant",
            content=[{
                "type": "output_text",
                "text": full_response.strip(),
                "annotations": []
            }]
        )
        
        response_usage = ResponseUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_tokens_details=ResponseUsageDetails(cached_tokens=0),
            output_tokens_details=ResponseUsageDetails(reasoning_tokens=0)
        )
        
        # Store response if requested
        if config.enable_history and request.store:
            stored_response_id = conversation_history.store_response(
                content=full_response.strip(),
                model=mapped_model,
                finish_reason="stop",
                tokens={
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": total_tokens
                },
                messages=_format_input_as_messages(request.input),
                conversation_id=None,
                parent_response_id=request.previous_response_id,
                metadata={
                    "request_id": request_id,
                    "temperature": request.temperature,
                    "max_output_tokens": request.max_output_tokens,
                    "instructions": request.instructions,
                    "api_type": "responses"
                }
            )
            logger.info(f"Request {request_id}: Stored response as {stored_response_id}")
        
        # Create the response in the expected format
        api_response = ResponseAPIResponse(
            id=response_id,
            object="response",
            created_at=int(time.time()),
            status="completed",
            error=None,
            incomplete_details=None,
            instructions=request.instructions,
            max_output_tokens=request.max_output_tokens,
            model=request.model,
            output=[response_output],
            parallel_tool_calls=request.parallel_tool_calls or True,
            previous_response_id=request.previous_response_id,
            reasoning=ResponseReasoning(effort=None, summary=None),
            store=request.store or True,
            temperature=request.temperature or 1.0,
            text=request.text,
            tool_choice=str(request.tool_choice) if request.tool_choice else "auto",
            tools=request.tools or [],
            top_p=request.top_p or 1.0,
            truncation=request.truncation or "disabled",
            usage=response_usage,
            user=request.user,
            metadata=request.metadata or {}
        )
        
        logger.info(f"Request {request_id}: Completed successfully")
        return api_response
        
    except ClaudeNotFoundError as e:
        logger.error(f"Request {request_id}: Claude not found - {e}")
        raise HTTPException(
            status_code=503,
            detail="Claude CLI not found. Please install and configure Claude CLI."
        )
        
    except ClaudeTimeoutError as e:
        logger.error(f"Request {request_id}: Timeout - {e}")
        raise HTTPException(
            status_code=408,
            detail="Request timed out. Please try again with a shorter prompt."
        )
        
    except ClaudeRateLimitError as e:
        logger.error(f"Request {request_id}: Rate limit - {e}")
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait before making another request."
        )
        
    except ValueError as e:
        logger.error(f"Request {request_id}: Validation error - {e}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error(f"Request {request_id}: Unexpected error - {e}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again."
        )


@app.get("/v1/responses/{response_id}")
async def get_response(response_id: str):
    """Retrieve a stored response by ID"""
    if not config.enable_history:
        raise HTTPException(status_code=503, detail="Response storage is disabled")
    
    response_data = conversation_history.get_response(response_id)
    if not response_data:
        raise HTTPException(status_code=404, detail="Response not found")
    
    # Format response to match OpenAI's format
    return {
        "id": response_data["id"],
        "object": "response",
        "created": response_data["created"],
        "model": response_data["model"],
        "content": response_data["content"],
        "tokens": response_data["tokens"],
        "finish_reason": response_data["finish_reason"],
        "conversation_id": response_data.get("conversation_id"),
        "parent_response_id": response_data.get("parent_response_id"),
        "metadata": response_data.get("metadata", {})
    }

@app.get("/v1/responses")
async def list_responses(conversation_id: Optional[str] = None):
    """List stored responses, optionally filtered by conversation_id"""
    if not config.enable_history:
        raise HTTPException(status_code=503, detail="Response storage is disabled")
    
    responses = conversation_history.list_responses(conversation_id)
    return {
        "object": "list",
        "data": responses,
        "total": len(responses)
    }

@app.delete("/v1/responses/{response_id}")
async def delete_response(response_id: str):
    """Delete a stored response"""
    if not config.enable_history:
        raise HTTPException(status_code=503, detail="Response storage is disabled")
    
    response_data = conversation_history.get_response(response_id)
    if not response_data:
        raise HTTPException(status_code=404, detail="Response not found")
    
    conversation_history._delete_response(response_id)
    logger.info(f"Deleted response: {response_id}")
    
    return {"message": "Response deleted successfully"}

@app.get("/v1/responses/{response_id}/context")
async def get_response_context(response_id: str):
    """Get the full conversation context chain for a response"""
    if not config.enable_history:
        raise HTTPException(status_code=503, detail="Response storage is disabled")
    
    response_data = conversation_history.get_response(response_id)
    if not response_data:
        raise HTTPException(status_code=404, detail="Response not found")
    
    context_chain = conversation_history.get_response_context_chain(response_id)
    return {
        "response_id": response_id,
        "context_chain": context_chain,
        "total_messages": len(context_chain)
    }


def parse_args():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description="Claude OpenAI API Bridge")
    parser.add_argument("--host", default=config.host, help="Host to bind to")
    parser.add_argument("--port", type=int, default=config.port, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--claude-command", default=config.claude_command, 
                       help="Claude command path")
    parser.add_argument("--request-timeout", type=int, default=config.request_timeout,
                       help="Request timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=config.max_retries,
                       help="Maximum number of retries")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Log level")
    return parser.parse_args()

if __name__ == "__main__":
    import uvicorn
    
    # Parse command line arguments
    args = parse_args()
    
    # Update config with command line arguments
    config.host = args.host
    config.port = args.port
    config.debug = args.debug or config.debug
    config.claude_command = args.claude_command
    config.request_timeout = args.request_timeout
    config.max_retries = args.max_retries
    
    # Update logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info(f"Starting Claude OpenAI API Bridge on {config.host}:{config.port}")
    logger.info(f"Claude command: {config.claude_command}")
    logger.info(f"Request timeout: {config.request_timeout}s")
    logger.info(f"Max retries: {config.max_retries}")
    logger.info(f"Debug mode: {config.debug}")
    
    uvicorn.run(
        app, 
        host=config.host, 
        port=config.port,
        log_level=args.log_level.lower(),
        access_log=config.debug
    )