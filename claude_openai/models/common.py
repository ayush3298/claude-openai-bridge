"""Common models shared across the application"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


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