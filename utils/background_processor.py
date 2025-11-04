import asyncio
import threading
from typing import Dict, Any
from sqlalchemy.orm import Session
from database.database import SessionLocal
from database.models import Document, DocumentStatus
from utils.document_processor import DocumentProcessor
from utils.simple_vector_db import SimpleVectorDB
from datetime import datetime
import os

class BackgroundProcessor:
    def __init__(self, vector_db=None, chunk_mode: str = "balanced", hf_model_name: str = "all-MiniLM-L6-v2"):
        self.processor = DocumentProcessor()
        self.vector_db = vector_db or SimpleVectorDB()
        self.processing_queue = {}
        # chunking preferences
        self.chunk_mode = chunk_mode  # "fast" | "balanced" | "quality"
        self.hf_model_name = hf_model_name
    
    def process_document(self, doc_id: int):
        """Process a document in the background"""
        try:
            # Get database session
            db = SessionLocal()
            
            try:
                # Update status to in progress
                document = db.query(Document).filter(Document.id == doc_id).first()
                if not document:
                    print(f"Document {doc_id} not found")
                    return
                
                document.status = DocumentStatus.IN_PROGRESS
                db.commit()
                
                # Extract text from document
                print(f"Processing document {doc_id}: {document.filename}")
                text = self.processor.extract_text(document.file_path, document.file_type)
                print(f"✓ Extracted text length: {len(text)} characters")
                
                # Decide doc type for chunker
                mime = (document.file_type or "").lower()
                if "markdown" in mime or document.filename.endswith((".md", ".markdown")):
                    doc_type = "markdown"
                elif "pdf" in mime:
                    doc_type = "pdf"
                elif "wordprocessingml" in mime or document.filename.endswith((".docx", ".doc")):
                    doc_type = "docx"
                else:
                    doc_type = "plain"

                # Intelligent chunking with quality/speed trade-off
                chunks = self.processor.chunk_text_intelligent(
                    text,
                    mode=self.chunk_mode,
                    doc_type=doc_type,
                    hf_model_name=self.hf_model_name,
                )
                print(f"✓ Created {len(chunks)} chunks using intelligent '{self.chunk_mode}' mode ({doc_type})")
                
                # Add chunks to vector database
                print(f"✓ Adding {len(chunks)} chunks to vector database...")
                success = self.vector_db.add_document_chunks(
                    doc_id=doc_id,
                    chunks=chunks,
                    metadata={
                        "filename": document.filename,
                        "original_filename": document.original_filename,
                        "file_type": document.file_type
                    }
                )
                print(f"✓ Vector database add result: {success}")
                
                if success:
                    # Update status to completed
                    document.status = DocumentStatus.COMPLETED
                    document.completed_at = datetime.utcnow()
                    db.commit()
                    print(f"Document {doc_id} processed successfully")
                else:
                    # Update status to failed
                    document.status = DocumentStatus.FAILED
                    document.error_message = "Failed to add chunks to vector database"
                    db.commit()
                    print(f"Document {doc_id} processing failed")
                
            finally:
                db.close()
                
        except Exception as e:
            print(f"Error processing document {doc_id}: {str(e)}")
            # Update status to failed
            db = SessionLocal()
            try:
                document = db.query(Document).filter(Document.id == doc_id).first()
                if document:
                    document.status = DocumentStatus.FAILED
                    document.error_message = str(e)
                    db.commit()
            finally:
                db.close()
    
    def start_processing(self, doc_id: int):
        """Start processing a document in a separate thread"""
        if doc_id in self.processing_queue:
            print(f"Document {doc_id} is already being processed")
            return
        
        # Mark as in queue
        self.processing_queue[doc_id] = True
        
        # Start processing in background thread
        thread = threading.Thread(target=self._process_with_cleanup, args=(doc_id,))
        thread.daemon = True
        thread.start()
    
    def _process_with_cleanup(self, doc_id: int):
        """Process document and clean up queue"""
        try:
            self.process_document(doc_id)
        finally:
            # Remove from processing queue
            if doc_id in self.processing_queue:
                del self.processing_queue[doc_id]
    
    def is_processing(self, doc_id: int) -> bool:
        """Check if a document is currently being processed"""
        return doc_id in self.processing_queue

# Global instance
background_processor = BackgroundProcessor()
