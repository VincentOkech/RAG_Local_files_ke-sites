from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from pydantic import BaseModel
from src.agent import RAGAgent
import uvicorn
import os

# Initialize FastAPI app
app = FastAPI(
    title="Kenyan RAG Agent API",
    description="API for document processing and question answering with Kenya-specific context",
    version="1.0.0"
)

# Initialize RAG agent
agent = RAGAgent()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class Query(BaseModel):
    text: str

class Response(BaseModel):
    response: str

# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint that displays API information."""
    return """
    <html>
        <head>
            <title>Kenyan RAG Agent API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #2c3e50; }
                .endpoints { margin-top: 20px; }
                .endpoint { margin: 10px 0; padding: 10px; background: #f7f9fc; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>Welcome to Kenyan RAG Agent API</h1>
            <p>This API provides document processing and question answering capabilities with Kenya-specific context.</p>
            <div class="endpoints">
                <h2>Available Endpoints:</h2>
                <div class="endpoint">
                    <strong>POST /query</strong> - Submit a question to the RAG agent
                </div>
                <div class="endpoint">
                    <strong>POST /upload</strong> - Upload documents for processing
                </div>
                <div class="endpoint">
                    <strong>GET /docs</strong> - View API documentation
                </div>
            </div>
        </body>
    </html>
    """

@app.post("/query", response_model=Response)
async def process_query(query: Query):
    """
    Process a query using the RAG agent.
    
    Args:
        query (Query): The query text
        
    Returns:
        Response: The agent's response
    """
    try:
        response = await agent.get_response(query.text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and process a document.
    
    Args:
        file (UploadFile): The file to upload (PDF or TXT)
        
    Returns:
        dict: Upload status message
    """
    try:
        content = await file.read()
        file_path = f"data/documents/{file.filename}"
        
        # Ensure directory exists
        os.makedirs("data/documents", exist_ok=True)
        
        with open(file_path, "wb") as f:
            f.write(content)

        if file.filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
            
        documents = loader.load()
        agent.vector_store.add_documents(documents)
        
        return {"message": "File uploaded and processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Check the health status of the API.
    
    Returns:
        dict: Health status information
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "agent_status": "initialized"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 