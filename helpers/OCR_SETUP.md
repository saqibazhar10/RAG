# 🖼️ OCR Setup Guide for PDF Image Text Extraction

This guide will help you set up Tesseract OCR to extract text from images within PDF documents.

## 🎯 What This Enables

- **PDF Text Extraction**: Extract text from PDF pages
- **Image OCR**: Extract text from images embedded in PDFs
- **Fallback Support**: If OCR fails, falls back to regular PDF text extraction
- **Better Text Quality**: PyMuPDF provides better text extraction than PyPDF2

## 📋 Prerequisites

### 1. Install Tesseract OCR

#### Windows
```bash
# Option 1: Using Chocolatey (recommended)
choco install tesseract

# Option 2: Download from GitHub
# Visit: https://github.com/UB-Mannheim/tesseract/wiki
# Download and install the latest version
```

#### macOS
```bash
# Using Homebrew
brew install tesseract

# Verify installation
tesseract --version
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install tesseract-ocr

# Install additional language packs if needed
sudo apt install tesseract-ocr-eng tesseract-ocr-fra tesseract-ocr-deu
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## 🔧 Configuration

### 1. Verify Tesseract Installation

```bash
# Check if Tesseract is in PATH
tesseract --version

# Expected output:
# tesseract 5.3.0
# leptonica-1.82.0
# ...
```

### 2. Set Tesseract Path (if needed)

If Tesseract is not in your PATH, you can set it explicitly:

```python
# In your code, before using pytesseract
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
# or
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Linux/Mac
```

## 🧪 Testing the Setup

### 1. Test OCR Functionality

```bash
python test_ocr.py
```

### 2. Test with API

1. **Upload a PDF with images:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@your_pdf_with_images.pdf"
```

2. **Check processing status:**
```bash
curl -X GET "http://localhost:8000/documents/1/status"
```

3. **Query the processed document:**
```bash
curl -X POST "http://localhost:8000/documents/query" \
  -H "Content-Type: application/json" \
  -d "{\"doc_id\": 1, \"query\": \"text from image\", \"top_k\": 3}"
```

## 📊 How It Works

### 1. **Enhanced PDF Processing**
- Uses PyMuPDF (fitz) for better text extraction
- Detects images on each page
- Extracts images and converts to PIL format

### 2. **OCR Processing**
- Converts PDF images to PIL Image objects
- Uses Tesseract OCR to extract text
- Marks OCR content with `[Image X Text]:` prefix

### 3. **Fallback Support**
- If PyMuPDF fails, falls back to PyPDF2
- If OCR fails for an image, continues with other images
- Graceful error handling throughout

### 4. **Text Chunking**
- Combines regular text and OCR text
- Chunks the combined text for vector storage
- Maintains context between text and image content

## 🚨 Troubleshooting

### Common Issues

#### 1. **Tesseract not found**
```
Error: The system cannot find the path specified
```
**Solution**: Install Tesseract and add to PATH

#### 2. **Import errors**
```
ModuleNotFoundError: No module named 'fitz'
```
**Solution**: Install PyMuPDF: `pip install PyMuPDF`

#### 3. **OCR quality issues**
- Ensure good image quality in PDFs
- Check Tesseract language packs
- Consider image preprocessing if needed

#### 4. **Memory issues with large PDFs**
- Large PDFs with many images may use significant memory
- Consider processing in smaller batches

### Performance Tips

1. **Image Quality**: Higher resolution images = better OCR accuracy
2. **Language**: Install appropriate language packs for your documents
3. **Batch Processing**: Process multiple documents sequentially to avoid memory issues

## 📈 Expected Results

After successful setup, you should see:

1. **Console Output:**
```
Processing document 1: document.pdf
Page 1 contains 2 images, extracting text using OCR...
✓ Extracted text from image 1: 45 characters
✓ Extracted text from image 2: 32 characters
✓ Created 3 chunks
✓ Adding 3 chunks to vector database...
```

2. **Search Results:**
- Text from both PDF content and images
- Relevant chunks containing image text
- Improved search accuracy for image-heavy documents

## 🔍 Advanced Usage

### Custom OCR Configuration

```python
# Configure Tesseract parameters
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(image, config=custom_config)
```

### Language Support

```python
# Extract text in specific language
text = pytesseract.image_to_string(image, lang='eng+fra')
```

### Image Preprocessing

```python
# Enhance image before OCR
import cv2
import numpy as np

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Apply threshold
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

## 🎉 Success Indicators

- ✅ Tesseract command works in terminal
- ✅ Python imports work without errors
- ✅ PDF processing shows OCR activity
- ✅ Search returns results from both text and images
- ✅ Console shows successful image text extraction

Your RAG system is now ready to handle PDFs with embedded images! 🚀
