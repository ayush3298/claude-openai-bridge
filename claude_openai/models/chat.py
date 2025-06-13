"""Chat completion API models compatible with OpenAI's API"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


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