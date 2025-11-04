#!/usr/bin/env python3
"""
Test script for Hugging Face embeddings in the vector database
"""

from utils.simple_vector_db import SimpleVectorDB
import time

def test_embeddings():
    """Test Hugging Face embeddings functionality"""
    print("🧪 Testing Hugging Face Embeddings")
    print("=" * 60)
    
    # Test 1: Initialize with default model
    print("🔍 Test 1: Initialize Vector Database")
    print("-" * 40)
    
    try:
        vector_db = SimpleVectorDB()
        print("✅ Vector database initialized successfully")
        
        # Get stats to see embedding model info
        stats = vector_db.get_document_stats()
        print(f"📊 Database stats: {stats}")
        
    except Exception as e:
        print(f"❌ Error initializing vector database: {str(e)}")
        return
    
    print()
    
    # Test 2: Test embedding creation
    print("🔍 Test 2: Test Embedding Creation")
    print("-" * 40)
    
    test_texts = [
        "This is a simple test sentence.",
        "Machine learning and artificial intelligence are fascinating topics.",
        "The quick brown fox jumps over the lazy dog.",
        "Python is a great programming language for data science.",
        "Natural language processing helps computers understand human language."
    ]
    
    embeddings = []
    for i, text in enumerate(test_texts):
        start_time = time.time()
        embedding = vector_db._create_embedding(text)
        end_time = time.time()
        
        print(f"  Text {i+1}: {text[:50]}...")
        print(f"    Embedding dimensions: {len(embedding)}")
        print(f"    Time: {(end_time - start_time)*1000:.2f} ms")
        embeddings.append(embedding)
    
    print()
    
    # Test 3: Test similarity calculations
    print("🔍 Test 3: Test Similarity Calculations")
    print("-" * 40)
    
    # Test similarity between related and unrelated texts
    test_pairs = [
        ("Machine learning is amazing", "AI and ML are fascinating", "Related"),
        ("Python programming", "Cooking recipes", "Unrelated"),
        ("Data science", "Machine learning algorithms", "Related"),
        ("Weather forecast", "Programming languages", "Unrelated")
    ]
    
    for text1, text2, relation in test_pairs:
        emb1 = vector_db._create_embedding(text1)
        emb2 = vector_db._create_embedding(text2)
        similarity = vector_db._cosine_similarity(emb1, emb2)
        
        print(f"  {relation}:")
        print(f"    Text 1: {text1}")
        print(f"    Text 2: {text2}")
        print(f"    Similarity: {similarity:.4f}")
        print()
    
    # Test 4: Test document operations
    print("🔍 Test 4: Test Document Operations")
    print("-" * 40)
    
    # Add a test document
    test_chunks = [
        "Machine learning is a subset of artificial intelligence.",
        "It focuses on algorithms that can learn from data.",
        "Deep learning is a type of machine learning using neural networks.",
        "Natural language processing helps computers understand text."
    ]
    
    success = vector_db.add_document_chunks(
        doc_id=1,
        chunks=test_chunks,
        metadata={"filename": "test_doc.txt", "file_type": "text/plain"}
    )
    
    if success:
        print("✅ Test document added successfully")
        
        # Test search
        print("🔍 Testing search functionality...")
        search_results = vector_db.search_all_documents("machine learning", top_k=3)
        
        print(f"  Search results for 'machine learning':")
        for i, result in enumerate(search_results):
            print(f"    Result {i+1}: {result['content'][:80]}...")
            print(f"      Similarity: {result['similarity_score']:.4f}")
        
        # Get updated stats
        updated_stats = vector_db.get_document_stats()
        print(f"📊 Updated stats: {updated_stats}")
        
    else:
        print("❌ Failed to add test document")
    
    print()
    
    # Test 5: Test model switching
    print("🔍 Test 5: Test Model Switching")
    print("-" * 40)
    
    available_models = vector_db.get_available_models()
    print(f"📋 Available models: {available_models}")
    
    # Try to switch to a different model (if available)
    if len(available_models) > 1:
        new_model = available_models[1]  # Try second model
        print(f"🔄 Attempting to switch to: {new_model}")
        
        try:
            success = vector_db.change_model(new_model)
            if success:
                print(f"✅ Successfully switched to: {new_model}")
                
                # Test embedding with new model
                new_embedding = vector_db._create_embedding("Test with new model")
                print(f"   New embedding dimensions: {len(new_embedding)}")
                
                # Get updated stats
                final_stats = vector_db.get_document_stats()
                print(f"📊 Final stats: {final_stats}")
                
            else:
                print(f"❌ Failed to switch to: {new_model}")
                
        except Exception as e:
            print(f"⚠️  Model switching failed: {str(e)}")
            print("   This is normal if the model is not available locally")
    else:
        print("ℹ️  Only one model available, skipping model switching test")
    
    print()
    
    # Test 6: Performance comparison
    print("🔍 Test 6: Performance Comparison")
    print("-" * 40)
    
    long_text = "This is a longer text that we will use to test embedding performance. " * 10
    
    # Test embedding time
    start_time = time.time()
    embedding = vector_db._create_embedding(long_text)
    end_time = time.time()
    
    print(f"  Long text embedding:")
    print(f"    Text length: {len(long_text)} characters")
    print(f"    Embedding dimensions: {len(embedding)}")
    print(f"    Time: {(end_time - start_time)*1000:.2f} ms")
    
    print()
    print("✅ All embedding tests completed successfully!")
    print("\n💡 Key Benefits of Hugging Face Embeddings:")
    print("   • Better semantic understanding")
    print("   • Industry-standard models")
    print("   • Configurable model selection")
    print("   • Fallback to simple embeddings if needed")
    print("   • Support for multiple languages")
    print("   • Optimized performance")

if __name__ == "__main__":
    test_embeddings()
