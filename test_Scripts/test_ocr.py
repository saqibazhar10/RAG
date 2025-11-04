#!/usr/bin/env python3
"""
Test script for OCR functionality in PDF processing
"""

import os
import sys
from utils.document_processor import DocumentProcessor

def test_pdf_ocr():
    """Test PDF OCR functionality"""
    print("🧪 Testing PDF OCR Functionality")
    print("=" * 50)
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # Check if uploads directory exists and has PDFs
    uploads_dir = "./uploads"
    if not os.path.exists(uploads_dir):
        print("❌ Uploads directory not found. Please upload some PDFs first.")
        return
    
    # Find PDF files
    pdf_files = [f for f in os.listdir(uploads_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("❌ No PDF files found in uploads directory.")
        print("Please upload some PDFs with images first using the API.")
        return
    
    print(f"📁 Found {len(pdf_files)} PDF file(s):")
    for pdf in pdf_files:
        print(f"   - {pdf}")
    
    print("\n🔍 Testing OCR on first PDF...")
    
    # Test first PDF
    pdf_path = os.path.join(uploads_dir, pdf_files[0])
    print(f"Processing: {pdf_path}")
    
    try:
        # Extract text
        text = processor.extract_text_from_pdf(pdf_path)
        
        print(f"\n✅ Successfully extracted text:")
        print(f"   Length: {len(text)} characters")
        print(f"   Preview: {text[:200]}...")
        
        # Check for OCR content
        if "[Image" in text:
            print("\n🎯 OCR Content Found:")
            lines = text.split('\n')
            for line in lines:
                if "[Image" in line:
                    print(f"   {line}")
        else:
            print("\nℹ️  No OCR content markers found (may be text-only PDF)")
            
    except Exception as e:
        print(f"❌ Error processing PDF: {str(e)}")
        print("\n💡 Make sure you have:")
        print("   1. Installed Tesseract OCR on your system")
        print("   2. Installed the Python dependencies: pip install -r requirements.txt")
        print("   3. Tesseract is in your PATH")

if __name__ == "__main__":
    test_pdf_ocr()
