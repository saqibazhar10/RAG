from sqlalchemy import Column, Integer, String, DateTime, Text, Enum,ForeignKey,JSON,Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import enum
from sqlalchemy.dialects.postgresql import UUID 
from sqlalchemy.orm import relationship,backref
import uuid
from datetime import datetime

Base = declarative_base()

class DocumentStatus(enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_type = Column(String(100), nullable=False)
    status = Column(Enum(DocumentStatus), default=DocumentStatus.PENDING)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}', status='{self.status}')>"



class Conversations(Base):
    __tablename__ = "conversations"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        unique=True,
        nullable=False
    )
    conv_title = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    doc_id = Column(JSON, nullable=True)
    messages = relationship("Messages", back_populates="conversation", cascade="all, delete")

class SenderType(str, enum.Enum):
    USER = "user"
    LLM = "llm"

class Messages(Base):
    __tablename__ = "messages"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        unique=True,
        nullable=False
    )

    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id"),
        nullable=False
    )
    
    parent_message_id = Column(
        UUID(as_uuid=True),
        ForeignKey("messages.id"),
        nullable=True 
    )

    content = Column(String, nullable=False)
    sender = Column(Enum(SenderType), nullable=False)
    timestamp = Column(DateTime, default=datetime.now)

    chunks_used = Column(String,nullable=True)

    # Relationships
    conversation = relationship("Conversations", back_populates="messages")
    responses = relationship("Messages", backref=backref("parent", remote_side=[id]), cascade="all, delete")

class Agents(Base):
    __tablename__ = "agents"
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        unique=True,
        nullable=False
    )
    name =Column(String(255), nullable=False)
    description = Column(String, nullable=True)
    instructions = Column(String, nullable=False)
    creativity = Column(Float, nullable=False)