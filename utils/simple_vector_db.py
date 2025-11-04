import os
from typing import List, Dict, Any
import numpy as np
import math
from collections import defaultdict

# Hugging Face embeddings
from sentence_transformers import SentenceTransformer

class SimpleVectorDB:
    """Simple in-memory vector database for document storage and search"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize vector database with Hugging Face embeddings
        
        Args:
            model_name: Hugging Face model name for embeddings
        """
        self.documents = {}  # doc_id -> list of chunks
        self.chunk_vectors = {}  # chunk_id -> vector
        self.chunk_metadata = {}  # chunk_id -> metadata
        self.chunk_content = {}  # chunk_id -> content
        
        # Initialize Hugging Face embedding model
        try:
            print(f"🔄 Loading Hugging Face embedding model: {model_name}")
            local_model_path = os.path.join(os.path.dirname(__file__), '..', 'local_models', 'all-MiniLM-L6-v2')
            self.embedding_model = SentenceTransformer(local_model_path)
            print(f"✅ Successfully loaded embedding model: {model_name}")
            print(f"   Model dimensions: {self.embedding_model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"❌ Error loading embedding model: {str(e)}")
            print("   Falling back to simple embeddings...")
            self.embedding_model = None
        
    def _create_embedding(self, text: str) -> List[float]:
        """Create embedding using Hugging Face model or fallback to simple method"""
        if self.embedding_model:
            try:
                # Use Hugging Face embeddings
                embedding = self.embedding_model.encode(text, convert_to_tensor=False)
                return embedding.tolist()
            except Exception as e:
                print(f"⚠️  Hugging Face embedding failed: {str(e)}")
                print("   Falling back to simple embeddings...")
                return self._create_simple_embedding(text)
        else:
            # Fallback to simple embeddings
            return self._create_simple_embedding(text)
    
    def _create_simple_embedding(self, text: str) -> List[float]:
        """Create a simple embedding using text characteristics (fallback method)"""
        # Normalize text
        text = text.lower().strip()
        
        # Create feature vector based on text characteristics
        features = []
        
        # Character frequency features
        char_counts = defaultdict(int)
        for char in text:
            if char.isalpha():
                char_counts[char] += 1
        
        # Add character frequency features
        for char in 'abcdefghijklmnopqrstuvwxyz':
            features.append(char_counts[char] / max(len(text), 1))
        
        # Word length features
        words = text.split()
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            features.append(avg_word_length / 20.0)  # Normalize
        else:
            features.append(0.0)
        
        # Sentence count features
        sentences = text.split('.')
        features.append(len(sentences) / 10.0)  # Normalize
        
        # Text length feature
        features.append(len(text) / 1000.0)  # Normalize
        
        # Pad to 128 dimensions
        while len(features) < 128:
            features.append(0.0)
        
        return features[:128]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(a * a for a in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def add_document_chunks(self, doc_id: int, chunks: List[str], metadata: Dict[str, Any] = None) -> bool:
        """Add document chunks to vector database"""
        try:
            if not chunks:
                return False
            
            # Store chunks for this document
            self.documents[doc_id] = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"doc_{doc_id}_chunk_{i}"
                
                # Create embedding
                embedding = self._create_embedding(chunk)
                
                # Store everything
                self.chunk_vectors[chunk_id] = embedding
                self.chunk_content[chunk_id] = chunk
                self.chunk_metadata[chunk_id] = {
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    **(metadata or {})
                }
                
                self.documents[doc_id].append(chunk_id)
            
            print(f"✓ Added {len(chunks)} chunks for document {doc_id}")
            return True
            
        except Exception as e:
            print(f"Error adding document chunks: {str(e)}")
            return False
    
    def get_max_chunks(self,doc_ids: list,top_k:int = 50)-> List[Dict[str, Any]]:
        chunks = []
        for doc_id in doc_ids:
            for chunk_id in self.documents[doc_id]:
                chunks.append({
                        "doc_id":doc_id,
                        "chunk_id": chunk_id,
                        "content": self.chunk_content[chunk_id],
                        "metadata": self.chunk_metadata[chunk_id]
                    })
        results = []

        if len(doc_ids) > 1:
            # For each doc_id, pick top 50 similarity items that match this doc_id
            for doc_id in doc_ids:
                count = 0
                for item in chunks:
                    if item["doc_id"] == doc_id:
                        results.append({
                            "chunk_id": item["chunk_id"],
                            "content": item["content"],
                            "metadata": item["metadata"],
                            "chunk_index": item["metadata"]["chunk_index"]
                        })
                        count += 1
                        if count >= 50:
                            break 
        else:
            for item in chunks[:top_k]:
                results.append({
                    "chunk_id": item["chunk_id"],
                    "content": item["content"],
                    "metadata": item["metadata"],
                    "chunk_index": item["metadata"]["chunk_index"]
                })
        return results
    def search_document(self, doc_ids: list, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant chunks within a specific document"""
        try:
            similarity_threshold = 0.2
            similarities = []
            for doc_id in doc_ids:
                if doc_id not in self.documents:
                    return []
                
                # Create query embedding
                query_embedding = self._create_embedding(query)
                
                # Calculate similarities for chunks in this document
                
                for chunk_id in self.documents[doc_id]:
                    chunk_embedding = self.chunk_vectors[chunk_id]
                    similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                    
                    similarities.append({
                        "doc_id":doc_id,
                        "chunk_id": chunk_id,
                        "similarity_score": similarity,
                        "content": self.chunk_content[chunk_id],
                        "metadata": self.chunk_metadata[chunk_id]
                    })
                
                # Sort by similarity and return top_k
                similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
                
            # Format results
            results = []

            if len(doc_ids) > 1:
                # For each doc_id, pick top 20 similarity items that match this doc_id
                for doc_id in doc_ids:
                    count = 0
                    for item in similarities:
                        if item["doc_id"] == doc_id:
                            results.append({
                                "chunk_id": item["chunk_id"],
                                "content": item["content"],
                                "metadata": item["metadata"],
                                "similarity_score": item["similarity_score"],
                                "chunk_index": item["metadata"]["chunk_index"]
                            })
                            count += 1
                            if count >= 20:
                                break 
            else:
                for item in similarities[:top_k]:
                    results.append({
                        "chunk_id": item["chunk_id"],
                        "content": item["content"],
                        "metadata": item["metadata"],
                        "similarity_score": item["similarity_score"],
                        "chunk_index": item["metadata"]["chunk_index"]
                    })
            print(results)
            filtered_results = []
            for result in results:
                if result["similarity_score"] >= similarity_threshold:
                    filtered_results.append(result)

            results = filtered_results
            return results
            
        except Exception as e:
            print(f"Error searching document: {str(e)}")
            return []
    
    def search_all_documents(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search across all documents"""
        try:
            # Create query embedding
            query_embedding = self._create_embedding(query)
            
            # Calculate similarities for all chunks
            similarities = []
            for chunk_id, chunk_embedding in self.chunk_vectors.items():
                similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                
                similarities.append({
                    "chunk_id": chunk_id,
                    "similarity_score": similarity,
                    "content": self.chunk_content[chunk_id],
                    "metadata": self.chunk_metadata[chunk_id]
                })
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            # Format results
            results = []
            for item in similarities[:top_k]:
                results.append({
                    "chunk_id": item["chunk_id"],
                    "content": item["content"],
                    "metadata": item["metadata"],
                    "similarity_score": item["similarity_score"],
                    "doc_id": item["metadata"]["doc_id"],
                    "chunk_index": item["metadata"]["chunk_index"]
                })
            
            return results
            
        except Exception as e:
            print(f"Error searching all documents: {str(e)}")
            return []
    
    def delete_document(self, doc_id: int) -> bool:
        """Delete all chunks for a specific document"""
        try:
            if doc_id not in self.documents:
                return True
            
            # Remove all chunks for this document
            for chunk_id in self.documents[doc_id]:
                del self.chunk_vectors[chunk_id]
                del self.chunk_content[chunk_id]
                del self.chunk_metadata[chunk_id]
            
            # Remove document entry
            del self.documents[doc_id]
            
            return True
            
        except Exception as e:
            print(f"Error deleting document: {str(e)}")
            return False
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database"""
        try:
            total_chunks = len(self.chunk_vectors)
            total_docs = len(self.documents)
            
            stats = {
                "total_chunks": total_chunks,
                "total_documents": total_docs,
                "collection_name": "simple_vector_db",
                "db_type": "in_memory"
            }
            
            # Add embedding model info
            if self.embedding_model:
                stats["embedding_model"] = {
                    "name": self.embedding_model.get_model_name(),
                    "dimensions": self.embedding_model.get_sentence_embedding_dimension(),
                    "status": "active"
                }
            else:
                stats["embedding_model"] = {
                    "name": "simple_fallback",
                    "dimensions": 128,
                    "status": "fallback"
                }
            
            return stats
            
        except Exception as e:
            print(f"Error getting stats: {str(e)}")
            return {}
    
    def get_available_models(self) -> List[str]:
        """Get list of recommended Hugging Face embedding models"""
        return [
            "all-MiniLM-L6-v2",      # Fast, good quality (384d)
            "all-mpnet-base-v2",     # High quality (768d)
            "all-MiniLM-L12-v2",     # Better quality, slower (384d)
            "paraphrase-multilingual-MiniLM-L12-v2",  # Multilingual (384d)
            "distiluse-base-multilingual-cased-v2",   # Multilingual (512d)
        ]
    
    def change_model(self, model_name: str) -> bool:
        """Change the embedding model"""
        try:
            print(f"🔄 Changing embedding model to: {model_name}")
            new_model = SentenceTransformer(model_name)
            
            # Update the model
            self.embedding_model = new_model
            print(f"✅ Successfully changed to model: {model_name}")
            print(f"   Model dimensions: {new_model.get_sentence_embedding_dimension()}")
            
            # Re-embed all existing chunks with new model
            if self.chunk_vectors:
                print("🔄 Re-embedding existing chunks with new model...")
                for chunk_id, content in self.chunk_content.items():
                    new_embedding = self._create_embedding(content)
                    self.chunk_vectors[chunk_id] = new_embedding
                print("✅ All chunks re-embedded successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ Error changing model: {str(e)}")
            return False
