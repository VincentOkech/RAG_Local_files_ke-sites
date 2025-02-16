import os
from pathlib import Path
from typing import List, Optional
import shutil
from datetime import datetime

class StorageManager:
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.vector_store_dir = self.base_dir / "vector_store"
        self.documents_dir = self.base_dir / "documents"
        self.cache_dir = self.base_dir / "cache"
        self.backup_dir = self.vector_store_dir / "backups"
        
        # Create directory structure
        self._create_directories()

    def _create_directories(self) -> None:
        """Create necessary directory structure."""
        os.makedirs(self.vector_store_dir, exist_ok=True)
        os.makedirs(self.documents_dir / "pdf", exist_ok=True)
        os.makedirs(self.documents_dir / "txt", exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)

    def save_document(self, file_content: bytes, filename: str) -> Path:
        """Save uploaded document to appropriate directory."""
        file_extension = filename.lower().split('.')[-1]
        target_dir = self.documents_dir / ("pdf" if file_extension == "pdf" else "txt")
        file_path = target_dir / filename
        
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        return file_path

    def list_documents(self, file_type: Optional[str] = None) -> List[Path]:
        """List all documents or documents of specific type."""
        if file_type:
            return list((self.documents_dir / file_type).glob("*"))
        
        pdf_files = list((self.documents_dir / "pdf").glob("*"))
        txt_files = list((self.documents_dir / "txt").glob("*"))
        return pdf_files + txt_files

    def delete_document(self, filename: str) -> bool:
        """Delete a document."""
        file_extension = filename.lower().split('.')[-1]
        file_path = self.documents_dir / (file_extension) / filename
        
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def create_backup(self) -> str:
        """Create a backup of the vector store."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"backup_{timestamp}"
        
        if (self.vector_store_dir / "faiss_index").exists():
            os.makedirs(backup_path, exist_ok=True)
            shutil.copy2(
                self.vector_store_dir / "faiss_index",
                backup_path / "faiss_index"
            )
            shutil.copy2(
                self.vector_store_dir / "store.pkl",
                backup_path / "store.pkl"
            )
            return timestamp
        return ""

    def list_backups(self) -> List[str]:
        """List all available backups."""
        return [d.name.replace("backup_", "") for d in self.backup_dir.glob("backup_*")]

    def restore_backup(self, timestamp: str) -> bool:
        """Restore from a specific backup."""
        backup_path = self.backup_dir / f"backup_{timestamp}"
        
        if backup_path.exists():
            shutil.copy2(
                backup_path / "faiss_index",
                self.vector_store_dir / "faiss_index"
            )
            shutil.copy2(
                backup_path / "store.pkl",
                self.vector_store_dir / "store.pkl"
            )
            return True
        return False

    def get_document_path(self, filename: str) -> Optional[Path]:
        """Get the full path for a document."""
        file_extension = filename.lower().split('.')[-1]
        file_path = self.documents_dir / file_extension / filename
        return file_path if file_path.exists() else None

    def clear_cache(self) -> None:
        """Clear the cache directory."""
        shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True) 