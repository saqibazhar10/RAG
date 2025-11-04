import chromadb
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import uuid
import hashlib

load_dotenv()

class SimpleEmbeddingFunction:
    """Simple text-based embedding function that doesn't require onnxruntime"""
    
    def __call__(self, texts):
        """Convert texts to simple hash-based embeddings"""
        embeddings = []
        for text in texts:
            # Create a simple hash-based embedding
            text_hash = hashlib.md5(text.encode()).hexdigest()
            # Convert hash to 128-dimensional vector
            embedding = [ord(c) / 255.0 for c in text_hash[:16]] * 8
            embeddings.append(embedding)
        return embeddings

class VectorDBManager:
    def __init__(self, db_path: str = "./chroma_db"):
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Create custom embedding function
        self.embedding_function = SimpleEmbeddingFunction()
        
        # Create or get collection with custom embedding function
        try:
            self.collection = self.client.get_or_create_collection(
                name="documents",
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            print("✓ Using custom embedding function")
        except Exception as e:
            print(f"Warning: Using fallback method: {e}")
            # Fallback to simple collection
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
    
    def add_document_chunks(self, doc_id: int, chunks: List[str], metadata: Dict[str, Any] = None) -> bool:
        """Add document chunks to vector database"""
        try:
            if not chunks:
                return False
            
            # Generate unique IDs for chunks
            chunk_ids = [f"doc_{doc_id}_chunk_{i}" for i in range(len(chunks))]
            
            # Add chunks to collection
            self.collection.add(
                documents=chunks,
                metadatas=[{
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    **(metadata or {})
                } for i in range(len(chunks))],
                ids=chunk_ids
            )
            
            return True
        except Exception as e:
            print(f"Error adding document chunks to vector DB: {str(e)}")
            return False
    
    def search_document(self, doc_id: int, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant chunks within a specific document"""
        try:
            # Search with document ID filter
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where={"doc_id": doc_id}
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    formatted_results.append({
                        "chunk_id": results['ids'][0][i],
                        "content": doc,
                        "metadata": metadata,
                        "similarity_score": 1 - distance,  # Convert distance to similarity
                        "chunk_index": metadata.get("chunk_index", i)
                    })
            
            return formatted_results
        except Exception as e:
            print(f"Error searching document in vector DB: {str(e)}")
            return []
    
    def search_all_documents(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant chunks across all documents"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    formatted_results.append({
                        "chunk_id": results['ids'][0][i],
                        "content": doc,
                        "metadata": metadata,
                        "similarity_score": 1 - distance,
                        "doc_id": metadata.get("doc_id"),
                        "chunk_index": metadata.get("chunk_index", i)
                    })
            
            return formatted_results
        except Exception as e:
            print(f"Error searching all documents in vector DB: {str(e)}")
            return []
    
    def delete_document(self, doc_id: int) -> bool:
        """Delete all chunks for a specific document"""
        try:
            # Get all chunks for the document
            results = self.collection.get(
                where={"doc_id": doc_id}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
            
            return True
        except Exception as e:
            print(f"Error deleting document from vector DB: {str(e)}")
            return False
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database"""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection.name,
                "db_path": self.db_path
            }
        except Exception as e:
            print(f"Error getting vector DB stats: {str(e)}")
            return {}
