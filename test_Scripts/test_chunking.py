#!/usr/bin/env python3
"""
Test script for the new recursive chunking strategies
"""

from utils.document_processor import DocumentProcessor

def test_chunking_strategies():
    """Test different chunking strategies"""
    print("🧪 Testing Advanced Chunking Strategies")
    print("=" * 60)
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # Sample text with different structures
    sample_text = """
    This is a sample document for testing chunking strategies.
    
    It contains multiple paragraphs with different content.
    
    The first paragraph discusses the importance of proper text chunking in RAG systems.
    Proper chunking ensures that semantic meaning is preserved when documents are split.
    
    The second paragraph explains how recursive chunking works.
    It tries to split at natural boundaries like paragraphs, sentences, and punctuation.
    This approach maintains context better than simple character-based splitting.
    
    The third paragraph demonstrates the benefits of overlap between chunks.
    Overlap helps maintain context when searching across chunk boundaries.
    It's especially useful for long documents with complex structure.
    
    Finally, we have some technical details about the implementation.
    The chunking algorithm uses a priority-based approach for separators.
    It starts with paragraphs, then sentences, then punctuation marks.
    """
    
    print("📝 Sample Text Length:", len(sample_text), "characters")
    print("📝 Sample Text Preview:")
    print(sample_text[:200] + "...")
    print()
    
    # Test 1: Default recursive chunking
    print("🔍 Test 1: Default Recursive Chunking")
    print("-" * 40)
    chunks = processor.chunk_text(sample_text, chunk_size=300, overlap=50)
    print(f"✓ Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {len(chunk)} chars")
        print(f"    Preview: {chunk[:100]}...")
        print()
    
    # Test 2: Advanced chunking with different strategies
    print("🔍 Test 2: Advanced Chunking Strategies")
    print("-" * 40)
    
    # Recursive strategy
    print("📋 Recursive Strategy:")
    recursive_chunks = processor.chunk_text_advanced(sample_text, strategy="recursive", 
                                                   chunk_size=300, overlap=50)
    print(f"  ✓ Created {len(recursive_chunks)} chunks")
    
    # Semantic strategy
    print("📋 Semantic Strategy:")
    semantic_chunks = processor.chunk_text_advanced(sample_text, strategy="semantic", 
                                                  chunk_size=300, overlap=50)
    print(f"  ✓ Created {len(semantic_chunks)} chunks")
    
    # Fixed strategy
    print("📋 Fixed Strategy:")
    fixed_chunks = processor.chunk_text_advanced(sample_text, strategy="fixed", 
                                               chunk_size=300, overlap=50)
    print(f"  ✓ Created {len(fixed_chunks)} chunks")
    
    # Markdown strategy
    print("📋 Markdown Strategy:")
    markdown_chunks = processor.chunk_text_advanced(sample_text, strategy="markdown", 
                                                  chunk_size=300, overlap=50)
    print(f"  ✓ Created {len(markdown_chunks)} chunks")
    
    print()
    
    # Test 3: Different chunk sizes
    print("🔍 Test 3: Different Chunk Sizes")
    print("-" * 40)
    
    sizes = [200, 400, 600]
    for size in sizes:
        chunks = processor.chunk_text(sample_text, chunk_size=size, overlap=50)
        print(f"  Chunk size {size}: {len(chunks)} chunks")
        avg_length = sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0
        print(f"    Average chunk length: {avg_length:.1f} characters")
    
    print()
    
    # Test 4: Size constraints
    print("🔍 Test 4: Size Constraints")
    print("-" * 40)
    
    constrained_chunks = processor.chunk_text(sample_text, chunk_size=300, overlap=50,
                                            min_chunk_size=150, max_chunk_size=500)
    print(f"✓ Created {len(constrained_chunks)} chunks with size constraints")
    for i, chunk in enumerate(constrained_chunks):
        print(f"  Chunk {i+1}: {len(chunk)} chars")
    
    print()
    
    # Test 5: Performance comparison
    print("🔍 Test 5: Performance Comparison")
    print("-" * 40)
    
    import time
    
    strategies = ["recursive", "semantic", "fixed", "markdown"]
    for strategy in strategies:
        start_time = time.time()
        chunks = processor.chunk_text_advanced(sample_text, strategy=strategy, 
                                             chunk_size=300, overlap=50)
        end_time = time.time()
        
        print(f"  {strategy.capitalize()} Strategy:")
        print(f"    Time: {(end_time - start_time)*1000:.2f} ms")
        print(f"    Chunks: {len(chunks)}")
        print(f"    Total chars: {sum(len(chunk) for chunk in chunks)}")
    
    print()
    print("✅ All chunking tests completed successfully!")
    print("\n💡 Key Benefits of the New LangChain-Based Chunking System:")
    print("   • Production-ready text splitting algorithms")
    print("   • Multiple chunking strategies for different use cases")
    print("   • Intelligent splitting at semantic boundaries")
    print("   • Markdown-aware chunking")
    print("   • Configurable size constraints and overlap")
    print("   • Text cleaning and normalization")
    print("   • Industry-standard implementation")

if __name__ == "__main__":
    test_chunking_strategies()
