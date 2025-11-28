from typing import List
import re
from src.models import Chunk
import uuid

class ChunkingService:
    """Service for chunking documents"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, metadata: dict = None) -> List[Chunk]:
        """
        Chunk text using sliding window with overlap
        
        Strategy:
        1. Split by paragraphs first
        2. Group paragraphs into chunks
        3. Add overlap between chunks
        """
        
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        start_char = 0
        
        for para in paragraphs:
            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) > self.chunk_size:
                if current_chunk:
                    # Save current chunk
                    chunk = Chunk(
                        id=str(uuid.uuid4()),
                        text=current_chunk.strip(),
                        metadata=metadata or {},
                        start_char=start_char,
                        end_char=start_char + len(current_chunk)
                    )
                    chunks.append(chunk)
                    
                    # Start new chunk with overlap
                    overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                    start_char = start_char + len(current_chunk) - len(overlap_text)
                    current_chunk = overlap_text + "\n\n" + para
                else:
                    current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Add final chunk
        if current_chunk:
            chunk = Chunk(
                id=str(uuid.uuid4()),
                text=current_chunk.strip(),
                metadata=metadata or {},
                start_char=start_char,
                end_char=start_char + len(current_chunk)
            )
            chunks.append(chunk)
        
        return chunks
    
    def chunk_code(self, code: str, metadata: dict = None) -> List[Chunk]:
        """
        Chunk code by functions/classes
        """
        # Simple implementation: split by function definitions
        patterns = [
            r'def \w+\(',  # Python functions
            r'class \w+',   # Python classes
            r'function \w+\(',  # JavaScript
        ]
        
        # For now, use same chunking as text
        # TODO: Implement smarter code chunking
        return self.chunk_text(code, metadata)

def get_chunking_service() -> ChunkingService:
    """Get chunking service instance"""
    return ChunkingService()
