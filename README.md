# RAG Document Processing API

A FastAPI-based Retrieval-Augmented Generation (RAG) system that allows you to upload different types of documents, process them into vector embeddings, and query them using semantic search.

## Features

- **File Upload Support**: PDF, DOCX, Excel, and text files
- **Background Processing**: Asynchronous document ingestion with status tracking
- **Vector Database**: ChromaDB integration for semantic search
- **PostgreSQL**: Document metadata and status tracking
- **RESTful API**: Complete API for all operations
- **Status Tracking**: Real-time document processing status

## Project Structure

```
RAG/
├── database/
│   ├── models.py          # SQLAlchemy models
│   └── database.py        # Database connection
├── utils/
│   ├── document_processor.py  # File processing utilities
│   ├── vector_db.py           # ChromaDB manager
│   └── background_processor.py # Background task processor
├── schemas/
│   └── document.py            # Pydantic schemas
├── main.py                    # FastAPI application
├── requirements.txt           # Python dependencies
├── env.example               # Environment variables template
├── alembic.ini              # Database migration config
└── README.md                 # This file
```

## Prerequisites

- Python 3.8+
- PostgreSQL database
- pip (Python package manager)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd RAG
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your database credentials
   ```

5. **Set up PostgreSQL database**
   ```bash
   # Create database
   createdb rag_database
   
   # Update DATABASE_URL in .env file
   DATABASE_URL=postgresql://username:password@localhost:5432/rag_database
   ```

6. **Run the application**
   ```bash
   python main.py
   ```

The API will be available at `http://localhost:8000`

## API Endpoints

### File Upload
- **POST** `/upload` - Upload a document for processing

### Document Management
- **GET** `/documents` - List all documents with pagination
- **GET** `/documents/{doc_id}` - Get document details
- **GET** `/documents/{doc_id}/status` - Get document processing status
- **DELETE** `/documents/{doc_id}` - Delete document and vector data

### Document Querying
- **POST** `/documents/query` - Query a specific document using vector search
- **POST** `/documents/search` - Search across all documents

### System
- **GET** `/health` - Health check
- **GET** `/stats` - System statistics

## Usage Examples

### 1. Upload a Document

**Using Postman:**
- Method: `POST`
- URL: `http://localhost:8000/upload`
- Body: `form-data`
- Key: `file` (File type)
- Value: Select your document file

**Response:**
```json
{
  "message": "Document uploaded successfully and processing started",
  "document_id": 1,
  "status": "pending"
}
```

### 2. Check Document Status

**GET** `http://localhost:8000/documents/1/status`

**Response:**
```json
{
  "id": 1,
  "filename": "uuid_filename.pdf",
  "original_filename": "document.pdf",
  "file_size": 1024000,
  "file_type": "application/pdf",
  "status": "completed",
  "error_message": null,
  "created_at": "2024-01-01T10:00:00",
  "updated_at": "2024-01-01T10:05:00",
  "completed_at": "2024-01-01T10:05:00"
}
```

### 3. Query a Document

**POST** `http://localhost:8000/documents/query`
```json
{
  "doc_id": 1,
  "query": "What is the main topic discussed?",
  "top_k": 5
}
```

**Response:**
```json
{
  "doc_id": 1,
  "query": "What is the main topic discussed?",
  "results": [
    {
      "chunk_id": "doc_1_chunk_0",
      "content": "The main topic discussed in this document...",
      "metadata": {
        "doc_id": 1,
        "chunk_index": 0,
        "total_chunks": 5
      },
      "similarity_score": 0.95,
      "chunk_index": 0
    }
  ],
  "total_results": 1
}
```

## Document Processing Flow

1. **Upload**: File is uploaded and saved to filesystem
2. **Database Record**: Document metadata is stored in PostgreSQL
3. **Background Processing**: Document is processed asynchronously
4. **Text Extraction**: Text is extracted based on file type
5. **Chunking**: Text is split into overlapping chunks
6. **Vector Storage**: Chunks are embedded and stored in ChromaDB
7. **Status Update**: Document status is updated to "completed"

## Supported File Types

- **PDF** (.pdf) - Using PyPDF2
- **DOCX** (.docx) - Using python-docx
- **Excel** (.xlsx, .xls) - Using openpyxl
- **Text** (.txt) - Plain text files

## Configuration

### Environment Variables

- `DATABASE_URL`: PostgreSQL connection string
- `CHROMA_DB_PATH`: Path for ChromaDB storage
- `UPLOAD_DIR`: Directory for uploaded files
- `MAX_FILE_SIZE`: Maximum file size in bytes
- `API_HOST`: API server host
- `API_PORT`: API server port

### File Size Limits

Default maximum file size is 10MB. You can modify this in the `.env` file.

## Development

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest
```

### Database Migrations
```bash
# Initialize Alembic
alembic init alembic

# Create migration
alembic revision --autogenerate -m "Initial migration"

# Apply migration
alembic upgrade head
```

### Code Formatting
```bash
# Install formatting tools
pip install black isort

# Format code
black .
isort .
```

## Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Verify PostgreSQL is running
   - Check database credentials in `.env`
   - Ensure database exists

2. **File Upload Fails**
   - Check file size limits
   - Verify supported file types
   - Check upload directory permissions

3. **Vector Search Returns No Results**
   - Ensure document processing is completed
   - Check ChromaDB collection exists
   - Verify document ID is correct

### Logs

Check the console output for detailed error messages and processing status updates.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions, please create an issue in the repository.
