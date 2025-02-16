import torch
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from .config import Config

def get_embeddings():
    """Initialize and return the embedding model."""
    # Set compute device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Configure model parameters
    model_kwargs = {
        'device': device,
        'trust_remote_code': True
    }
    
    # Configure encoding parameters
    encode_kwargs = {
        'normalize_embeddings': True,
        'batch_size': 8,
        'show_progress_bar': False
    }
    
    try:
        return HuggingFaceBgeEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder="data/models"  # Add local cache for models
        )
    except Exception as e:
        print(f"Error initializing embeddings: {e}")
        raise 