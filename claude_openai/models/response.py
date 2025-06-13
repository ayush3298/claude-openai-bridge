"""Response API models for enhanced Claude interactions"""
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


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