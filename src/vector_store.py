from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from typing import List, Optional, Dict, Any
import numpy as np
import pickle
import os
import shutil
from datetime import datetime
from .embeddings import get_embeddings
from .document_processor import DocumentProcessor
from .cache import Cache
from .config import Config
from .storage_manager import StorageManager
from langchain_community.embeddings import HuggingFaceEmbeddings

class VectorStore:
    def __init__(self, persist_dir: str = "data/vector_store"):
        self.persist_dir = persist_dir
        self.index_file = os.path.join(persist_dir, "faiss_index")
        self.store_file = os.path.join(persist_dir, "store.pkl")
        self.embeddings = get_embeddings()
        self.doc_processor = DocumentProcessor()
        self.cache = Cache()
        self.storage = StorageManager()
        self.vector_store = self._load_or_create_store()
        
        # Create backup directory
        self.backup_dir = os.path.join(persist_dir, "backups")
        os.makedirs(self.backup_dir, exist_ok=True)

    def _create_new_store(self) -> FAISS:
        """Create a new FAISS vector store with a dummy document."""
        # Create a dummy document to initialize the store
        dummy_text = ["initialization document"]
        return FAISS.from_texts(
            dummy_text,
            self.embeddings,
            metadatas=[{"source": "initialization"}]
        )

    def _load_or_create_store(self) -> FAISS:
        """Load existing vector store or create a new one."""
        os.makedirs(self.persist_dir, exist_ok=True)
        
        if os.path.exists(self.index_file) and os.path.exists(self.store_file):
            try:
                return FAISS.load_local(self.persist_dir, self.embeddings)
            except Exception as e:
                print(f"Error loading existing store: {e}")
                return self._create_new_store()
        return self._create_new_store()

    def _save_store(self) -> None:
        """Save the vector store to disk."""
        self.vector_store.save_local(self.persist_dir)

    def create_backup(self) -> str:
        """Create a backup of the current vector store."""
        if not os.path.exists(self.index_file):
            return "No vector store to backup"
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.backup_dir, f"backup_{timestamp}")
        os.makedirs(backup_path, exist_ok=True)
        
        # Copy the index and store files
        shutil.copy2(self.index_file, backup_path)
        shutil.copy2(self.store_file, backup_path)
        
        return f"Backup created at {backup_path}"

    def restore_backup(self, backup_timestamp: str) -> str:
        """Restore from a specific backup."""
        backup_path = os.path.join(self.backup_dir, f"backup_{backup_timestamp}")
        if not os.path.exists(backup_path):
            return f"Backup {backup_timestamp} not found"
            
        # Replace current store with backup
        shutil.copy2(os.path.join(backup_path, "faiss_index"), self.index_file)
        shutil.copy2(os.path.join(backup_path, "store.pkl"), self.store_file)
        
        # Reload the store
        self.vector_store = self._load_or_create_store()
        return f"Restored from backup {backup_timestamp}"

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        try:
            # Process documents into chunks
            chunks = self.doc_processor.chunk_documents(documents)
            
            # Prepare texts and metadata
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            # Create backup before adding new documents
            self.create_backup()
            
            # Add to vector store
            if not texts:  # Skip if no texts to add
                return
                
            self.vector_store.add_texts(
                texts=texts,
                metadatas=metadatas
            )
            
            # Save to disk
            self.vector_store.save_local(self.persist_dir)
            
            # Clear cache as store has been updated
            self.cache.cache.clear()
            
        except Exception as e:
            raise Exception(f"Error adding documents: {str(e)}")

    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search for similar documents."""
        try:
            return self.vector_store.similarity_search(
                query, 
                k=k,
                filter=filter
            )
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def delete_document(self, document_id: str) -> None:
        """Delete a document from the vector store."""
        try:
            # Create backup before deletion
            self.create_backup()
            
            # Delete document
            self.vector_store.delete(document_id)
            
            # Save changes
            self._save_store()
            
            # Clear cache as store has been updated
            self.cache.cache.clear()
            
        except Exception as e:
            raise Exception(f"Error deleting document: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            total_docs = len(self.vector_store.index_to_docstore_id)
            total_vectors = self.vector_store.index.ntotal
            dimension = self.vector_store.index.d
            
            return {
                "total_documents": total_docs,
                "total_vectors": total_vectors,
                "vector_dimension": dimension,
                "store_size_mb": os.path.getsize(self.index_file) / (1024 * 1024)
            }
        except Exception as e:
            return {"error": str(e)}

def initialize_vector_store(texts=None):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if texts:
        vector_store = FAISS.from_texts(texts, embeddings)
    else:
        # Initialize an empty vector store
        vector_store = FAISS.from_texts(["placeholder"], embeddings)
        # Remove the placeholder
        vector_store.delete(["placeholder"])
    
    return vector_store 