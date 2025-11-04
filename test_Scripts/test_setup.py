#!/usr/bin/env python3
"""
Test script to verify RAG project setup
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        # Test database imports
        from database.database import get_db, create_tables
        from database.models import Document, DocumentStatus
        print("✓ Database modules imported successfully")
        
        # Test utility imports
        from utils.document_processor import DocumentProcessor
        from utils.vector_db import VectorDBManager
        from utils.background_processor import background_processor
        print("✓ Utility modules imported successfully")
        
        # Test schema imports
        from schemas.document import DocumentResponse, DocumentQuery
        print("✓ Schema modules imported successfully")
        
        # Test FastAPI
        from fastapi import FastAPI
        print("✓ FastAPI imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_environment():
    """Test environment configuration"""
    print("\nTesting environment...")
    
    # Check if .env file exists
    if os.path.exists('.env'):
        print("✓ .env file found")
    else:
        print("⚠ .env file not found (copy from env.example)")
    
    # Check required directories
    required_dirs = ['uploads', 'chroma_db']
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ {dir_name} directory exists")
        else:
            print(f"⚠ {dir_name} directory not found (will be created on startup)")
    
    return True

def test_dependencies():
    """Test if required packages are installed"""
    print("\nTesting dependencies...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'sqlalchemy',
        'psycopg2',
        'chromadb',
        'pypdf2',
        'python-docx',
        'openpyxl',
        'python-magic-bin'  # Windows-compatible version
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            # Handle special cases for package imports
            if package == 'python-magic-bin':
                import magic
                print(f"✓ {package}")
            elif package == 'python-docx':
                import docx
                print(f"✓ {package}")
            elif package == 'pypdf2':
                import PyPDF2
                print(f"✓ {package}")
            else:
                __import__(package.replace('-', '_'))
                print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Run all tests"""
    print("RAG Project Setup Test")
    print("=" * 30)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test environment
    test_environment()
    
    # Test dependencies
    if not test_dependencies():
        success = False
    
    print("\n" + "=" * 30)
    if success:
        print("✓ All tests passed! Project is ready to run.")
        print("\nNext steps:")
        print("1. Set up PostgreSQL database")
        print("2. Configure .env file with database credentials")
        print("3. Run: python start.py")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Check import paths and file structure")
        print("3. Verify Python environment and dependencies")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
