from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class DocumentType(str, Enum):
    PDF = "pdf"
    TEXT = "text"
    CODE = "code"
    MARKDOWN = "markdown"

class Chunk(BaseModel):
    """A chunk of text from a document"""
    id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    start_char: int = 0
    end_char: int = 0

class Document(BaseModel):
    """Document model"""
    id: str
    filename: str
    file_type: DocumentType
    content: str
    chunks: List[Chunk] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

class SearchRequest(BaseModel):
    """Search request model"""
    query: str
    top_k: int = 10
    filters: Optional[Dict[str, Any]] = None

class SearchResult(BaseModel):
    """Search result model"""
    chunk: Chunk
    score: float
    document_id: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    services: Dict[str, str]
