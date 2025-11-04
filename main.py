from datetime import datetime
from uuid import UUID, uuid4
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import os
from dotenv import load_dotenv

# Import our modules
from database.database import get_db, create_tables
from database.models import (
    Document,
    DocumentStatus,
    Conversations,
    Messages,
    SenderType,
    Agents,
)
from schemas.document import (
    DocumentResponse,
    DocumentStatusResponse,
    DocumentQuery,
    DocumentQueryResponse,
    DocumentListResponse,
    UploadResponse,
)
from schemas.chat import LLMRequest, AgentCreateReq
from utils.document_processor import DocumentProcessor
from utils.simple_vector_db import SimpleVectorDB
from utils.groq_service import generate_answer, call_groq_llm, generate_conv_title,get_user_intent

load_dotenv()

app = FastAPI(
    title="RAG Document Processing API",
    description="API for uploading, processing, and querying documents using vector search",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
document_processor = DocumentProcessor()
vector_db = SimpleVectorDB(
    model_name="all-MiniLM-L6-v2"
)  # Fast, good quality embeddings

# Initialize background processor with the same vector database instance
from utils.background_processor import BackgroundProcessor

background_processor = BackgroundProcessor(vector_db=vector_db)


@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup"""
    create_tables()
    print("Database tables created/verified")


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Upload a document for processing and ingestion into vector database
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Check file size
        max_size = int(os.getenv("MAX_FILE_SIZE", 10485760))  # 10MB default
        file_content = await file.read()
        if len(file_content) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum allowed size of {max_size} bytes",
            )

        # Save file and get file info
        file_info = document_processor.save_uploaded_file(file_content, file.filename)

        # Create document record in database
        db_document = Document(
            filename=file_info["filename"],
            original_filename=file_info["original_filename"],
            file_path=file_info["file_path"],
            file_size=file_info["file_size"],
            file_type=file_info["file_type"],
            status=DocumentStatus.PENDING,
        )

        db.add(db_document)
        db.commit()
        db.refresh(db_document)

        # Start background processing
        background_processor.start_processing(db_document.id)

        return UploadResponse(
            message="Document uploaded successfully and processing started",
            document_id=db_document.id,
            status="pending",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    skip: int = 0, limit: int = 100, db: Session = Depends(get_db)
):
    """
    List all documents with pagination
    """
    try:
        documents = db.query(Document).offset(skip).limit(limit).all()
        total_count = db.query(Document).count()

        return DocumentListResponse(documents=documents, total_count=total_count)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/{doc_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(doc_id: int, db: Session = Depends(get_db)):
    """
    Get the current status of a document
    """
    try:
        document = db.query(Document).filter(Document.id == doc_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        return document
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/query", response_model=DocumentQueryResponse)
async def query_document(query_data: DocumentQuery):
    """
    Query a specific document using vector search
    """
    try:
        # Search for relevant chunks in the specific document
        results = vector_db.search_document(
            doc_id=query_data.doc_id, query=query_data.query, top_k=query_data.top_k
        )

        return DocumentQueryResponse(
            doc_id=query_data.doc_id,
            query=query_data.query,
            results=results,
            total_results=len(results),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/search")
async def search_all_documents(query: str, top_k: int = 10):
    """
    Search across all documents using vector search
    """
    try:
        results = vector_db.search_all_documents(query=query, top_k=top_k)

        return {"query": query, "results": results, "total_results": len(results)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/{doc_id}")
async def get_document(doc_id: int, db: Session = Depends(get_db)):
    """
    Get document details by ID
    """
    try:
        document = db.query(Document).filter(Document.id == doc_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        return document
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: int, db: Session = Depends(get_db)):
    """
    Delete a document and its associated vector data
    """
    try:
        document = db.query(Document).filter(Document.id == doc_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Delete from vector database
        vector_db.delete_document(doc_id)

        # Delete file from filesystem
        if os.path.exists(document.file_path):
            os.remove(document.file_path)

        # Delete from database
        db.delete(document)
        db.commit()

        return {"message": f"Document {doc_id} deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/del_all_conv")
async def delete_document(db: Session = Depends(get_db)):
    """
    Delete a document and its associated vector data
    """
    try:
        # Delete  all Conv from database
        db.query(Conversations).delete()
        db.commit()

        return {"message": f"All Conversations deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "message": "RAG API is running"}


@app.get("/stats")
async def get_stats():
    """
    Get system statistics
    """
    try:
        vector_stats = vector_db.get_document_stats()
        return {"vector_database": vector_stats, "api_status": "running"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/vector-db")
async def debug_vector_db():
    """
    Debug endpoint to check vector database contents
    """
    try:
        # Get stats
        stats = vector_db.get_document_stats()

        # Try to get some sample data
        sample_results = vector_db.search_all_documents("test", top_k=5)

        return {
            "stats": stats,
            "sample_search_results": sample_results,
            "total_sample_results": len(sample_results),
        }
    except Exception as e:
        return {"error": str(e), "stats": vector_db.get_document_stats()}


@app.get("/embeddings/models")
async def get_available_models():
    """Get list of available embedding models"""
    try:
        models = vector_db.get_available_models()
        current_model = (
            vector_db.embedding_model.get_model_name()
            if vector_db.embedding_model
            else "simple_fallback"
        )

        return {
            "available_models": models,
            "current_model": current_model,
            "recommendations": {
                "fast": "all-MiniLM-L6-v2",
                "quality": "all-mpnet-base-v2",
                "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embeddings/change-model")
async def change_embedding_model(model_name: str):
    """Change the embedding model"""
    try:
        success = vector_db.change_model(model_name)
        if success:
            return {"message": f"Successfully changed to model: {model_name}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to change model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/doc_chat")
async def llm_chat(query_data: LLMRequest, db: Session = Depends(get_db)):
    # Combine top-k results into a single context string
    if query_data.conversation_id:
        conversation = (
            db.query(Conversations).filter_by(id=query_data.conversation_id).first()
        )
        conv_title = conversation.conv_title
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        conv_title = generate_conv_title(
            api_key=os.getenv("GROK_KEY"),
            messages=[{"role": "user", "content": query_data.query}],
        )
        conversation = Conversations(
            id=uuid4(),
            conv_title=conv_title,
            created_at=datetime.now(),
            doc_id=query_data.doc_id,
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    user_message = Messages(
        conversation_id=conversation.id,
        content=query_data.query,
        sender=SenderType.USER,
        timestamp=datetime.now(),
    )

    db.add(user_message)
    db.commit()
    db.refresh(user_message)

    user_intent = get_user_intent(
        api_key=os.getenv("GROK_KEY"),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an intent detection model for a document chat system. "
                    "Determine the user's intent from their message.\n\n"
                    "If the user asks things like 'what’s in the document', 'give summary', "
                    "'summarize this', 'explain this document', 'what is this about', or any "
                    "similar phrasing that means they want to understand or summarize the document, "
                    "classify the intent as 'understand_doc'.\n\n"
                    "For all other types of questions or messages, classify the intent as 'general'.\n\n"
                    "Output only a JSON object in this format:\n"
                    "{\n"
                    '  "intent": "<understand_doc or general>",\n'
                    '  "confidence": <float between 0 and 1>,\n'
                    '  "details": "<brief reasoning>"\n'
                    "}"
                ),
            },
            {"role": "user", "content": query_data.query},
        ],
    )
    if user_intent == "general":
        results = vector_db.search_document(
            doc_ids=query_data.doc_id, query=query_data.query, top_k=20
        )
    else:
        results = vector_db.get_max_chunks(
            doc_ids=query_data.doc_id, top_k=50
        )
    print(results, "=" * 100)
    context = "\n\n".join([r["content"] for r in results])

    # Generate final LLM answer
    llm_response = generate_answer(query_data.query, context,user_intent)
    # Step 4: Save LLM response as reply
    llm_message = Messages(
        conversation_id=conversation.id,
        content=llm_response,
        sender=SenderType.LLM,
        parent_message_id=user_message.id,
        timestamp=datetime.now(),
        chunks_used=context,
    )
    db.add(llm_message)
    db.commit()

    return {
        "conversation_id": str(conversation.id),
        "doc_id": query_data.doc_id,
        "chunks_used": str(context),
        "user_message": query_data.query,
        "llm_response": llm_response,
        "conv_title": conv_title,
    }


@app.get("/create_conversations")
async def create_conversations(db: Session = Depends(get_db)):
    conversation = Conversations()
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation

# @app.post("/chat_agent")
# async def create_agent(query_data: LLMRequest, db: Session = Depends(get_db)):
#     agent = db.query(Agents).filter(Agents.id == "").one()
#     instructions = agent.instructions
#     creativity = agent.creativity
#     return creativity

@app.post("/create_agent")
async def create_agent(query_data: AgentCreateReq, db: Session = Depends(get_db)):
    to_create = db.query(Agents).filter(Agents.name == query_data.name).first()
    if not to_create:
        to_add = Agents(
            id=uuid4(),
            name=query_data.name,
            description=query_data.description,
            instructions=query_data.instructions,
            creativity=query_data.creativity,
        )
        db.add(to_add)
        db.commit()
        db.refresh(to_add)
    else:
        raise HTTPException(
            status_code=400, detail="Agent with this name already exists"
        )

    return {"result": "success"}


@app.get("/all_agents")
def get_all_conversations(db: Session = Depends(get_db)):
    agents = db.query(Agents).all()
    return [
        {
            "id": agent.id,
            "name": agent.name,
            "instructions": agent.instructions,
            "description": agent.description,
            "creativity": agent.creativity,
        }
        for agent in agents
    ]


@app.get("/conversation/{conversation_id}")
def get_conversation_with_responses(
    conversation_id: UUID, db: Session = Depends(get_db)
):
    conversation = (
        db.query(Conversations).filter(Conversations.id == conversation_id).first()
    )

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Get all user messages (exclude LLM ones)
    user_messages = (
        db.query(Messages)
        .filter(
            Messages.conversation_id == conversation_id,
            Messages.sender == SenderType.USER,
        )
        .order_by(Messages.timestamp)
        .all()
    )

    result = {
        "conversation_id": str(conversation.id),
        "conv_title": conversation.conv_title,
        "messages": [],
    }
    if conversation.doc_id:
        result["doc_id"] = conversation.doc_id

    for msg in user_messages:
        llm_response = (
            db.query(Messages)
            .filter(
                Messages.parent_message_id == msg.id, Messages.sender == SenderType.LLM
            )
            .first()
        )

        result["messages"].append(
            {
                "user_message": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "llm_response": llm_response.content if llm_response else None,
                "chunks_used": (
                    llm_response.chunks_used if llm_response.chunks_used else None
                ),
            }
        )

    return result


@app.get("/all_conversations")
def get_all_conversations(db: Session = Depends(get_db)):
    conversations = db.query(Conversations).all()
    return [
        {
            "id": str(conv.id),
            "conv_title": conv.conv_title,
            "created_at": conv.created_at,
        }
        for conv in conversations
    ]


@app.post("/free_chat")
async def free_chat(query_data: LLMRequest, db: Session = Depends(get_db)):
    # Step 1: Get or create conversation
    if query_data.conversation_id:
        conversation = (
            db.query(Conversations).filter_by(id=query_data.conversation_id).first()
        )
        conv_title = conversation.conv_title
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        # Making a conversation_title
        conv_title = generate_conv_title(
            api_key=os.getenv("GROK_KEY"),
            messages=[{"role": "user", "content": query_data.query}],
        )
        conversation = Conversations(
            id=uuid4(), conv_title=conv_title, created_at=datetime.now()
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    # Step 2: Save the user's message
    user_message = Messages(
        conversation_id=conversation.id,
        content=query_data.query,
        sender=SenderType.USER,
        timestamp=datetime.now(),
    )
    db.add(user_message)
    db.commit()
    db.refresh(user_message)
    history = (
        db.query(Messages)
        .filter(Messages.conversation_id == conversation.id)
        .order_by(Messages.timestamp.asc())
        .all()
    )

    # Step 4: Convert history to Groq/OpenAI chat format
    messages = []
    for msg in history:
        role = "user" if msg.sender == SenderType.USER else "assistant"
        messages.append({"role": role, "content": msg.content})

    # Step 3: Call LLM
    llm_response_text = call_groq_llm(
        api_key=os.getenv("GROK_KEY"),
        messages=messages,
    )

    # Step 4: Save LLM response as reply
    llm_message = Messages(
        conversation_id=conversation.id,
        content=llm_response_text,
        sender=SenderType.LLM,
        parent_message_id=user_message.id,
        timestamp=datetime.now(),
    )
    db.add(llm_message)
    db.commit()

    # Step 5: Return the response
    return {
        "conversation_id": str(conversation.id),
        "user_message": query_data.query,
        "llm_response": llm_response_text,
        "conv_title": conv_title,
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))

    uvicorn.run(app, host=host, port=port)
