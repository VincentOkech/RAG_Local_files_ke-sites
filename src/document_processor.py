from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from typing import List, Union
from langchain.schema import Document
import hashlib
import json

class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def process_file(self, file_path: str) -> List[Document]:
        """Load and chunk a document file."""
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path)
            
            documents = loader.load()
            return self.chunk_documents(documents)
        except Exception as e:
            raise Exception(f"Error processing file: {str(e)}")

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        try:
            chunks = self.text_splitter.split_documents(documents)
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_id': i,
                    'chunk_hash': self._generate_chunk_hash(chunk.page_content),
                    'chunk_size': len(chunk.page_content),
                    'total_chunks': len(chunks)
                })
            
            return chunks
        except Exception as e:
            raise Exception(f"Error chunking documents: {str(e)}")

    def _generate_chunk_hash(self, content: str) -> str:
        """Generate a unique hash for the chunk content."""
        return hashlib.sha256(content.encode()).hexdigest()

    def get_chunk_stats(self, chunks: List[Document]) -> dict:
        """Get statistics about the chunks."""
        try:
            return {
                'total_chunks': len(chunks),
                'average_chunk_size': sum(len(c.page_content) for c in chunks) / len(chunks),
                'largest_chunk': max(len(c.page_content) for c in chunks),
                'smallest_chunk': min(len(c.page_content) for c in chunks)
            }
        except Exception as e:
            raise Exception(f"Error calculating chunk statistics: {str(e)}")

    def process_document(self, file):
        try:
            # Extract text from document
            text_chunks = self._split_text(file)
            
            # Initialize vector store if not already initialized
            if self.vector_store is None:
                self.vector_store = initialize_vector_store(text_chunks)
            else:
                # Add to existing vector store
                self.vector_store.add_texts(text_chunks)
                
            return True
        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")
    
    def _split_text(self, file):
        # Your existing text splitting logic
        # ... existing code ...
        pass

    def _split_text(self, file):
        # Your existing text splitting logic
        # ... existing code ...
        pass 