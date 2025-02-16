import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Model Configuration
    MODEL_NAME = "llama-3.3-70b-versatile"  # Updated to Llama 3.3 70B Versatile
    TEMPERATURE = 0.7
    MAX_TOKENS = 1000
    
    # Embedding Configuration
    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
    
    # Vector Store Configuration
    VECTOR_DB_PATH = "data/vector_store"
    
    # Web Search Configuration
    MAX_SEARCH_RESULTS = 5
    
    # Document Processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    KENYA_DOMAINS = [".ke", "co.ke", "ac.ke", "go.ke", "or.ke"] 