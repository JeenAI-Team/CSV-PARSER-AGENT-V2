"""
FastAPI Main Application for Data Analysis Agent
Production-ready with parallel request support and API key authentication
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import tempfile
import os
import base64
import uvicorn

from csv_agent import CSVAnalysisAgent
from config import (
    get_agent_config,
    validate_config,
    API_KEY,
    SERVER_PORT,
    SERVER_TIMEOUT,
    GEMMA_3_DEPLOYMENT,
    VLLM_ENDPOINT
)


# Validate configuration on startup
try:
    validate_config()
except ValueError as e:
    print(f"Configuration Error: {e}")
    print("Please set required environment variables in .env file")
    exit(1)


# Initialize FastAPI app
app = FastAPI(
    title="Data Analysis Agent API",
    description="LangChain ReAct agent for CSV/XLSX analysis with vLLM Gemma",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent globally
agent = CSVAnalysisAgent(**get_agent_config())


# Request/Response Models
class Base64AnalysisRequest(BaseModel):
    """Request model for base64 file analysis - compatible with playground-backend"""
    content: str
    question: str
    file_type: str = "csv"  # csv, xlsx, xls
    model: Optional[str] = None  # From playground-backend
    provider: Optional[str] = None  # From playground-backend
    encoding: Optional[str] = None
    delimiter: Optional[str] = None


class AnalysisResponse(BaseModel):
    """Response model for analysis results"""
    status: str
    answer: Optional[str] = None
    executed_code: Optional[str] = None
    result: Optional[str] = None
    attempts: int
    error: Optional[str] = None
    explanation: Optional[str] = None


# Security: API Key Authentication
async def verify_api_key(x_jeen_csv_service: str = Header(None, alias="x-jeen-csv-service")):
    """Verify API key from header (case-insensitive)"""
    if not x_jeen_csv_service or x_jeen_csv_service != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )
    return x_jeen_csv_service


# Background task for file cleanup
def cleanup_file(file_path: str):
    """Delete temporary file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error cleaning up {file_path}: {e}")


# Endpoints
@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "Data Analysis Agent API is running",
        "model": GEMMA_3_DEPLOYMENT,
        "endpoint": VLLM_ENDPOINT,
        "version": "2.0.0"
    }


@app.post("/analyze/form", response_model=AnalysisResponse)
async def analyze_form(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CSV or Excel file to analyze"),
    question: str = Form(..., description="Question to ask about the data"),
    model: Optional[str] = Form(None, description="Model to use (ignored, uses GEMMA_3_DEPLOYMENT)"),
    encoding: Optional[str] = Form(None, description="File encoding (ignored for auto-detection)"),
    delimiter: Optional[str] = Form(None, description="CSV delimiter (ignored for auto-detection)"),
    api_key: str = Depends(verify_api_key)
):
    """
    Analyze an uploaded CSV or Excel file via form-data.
    Compatible with data-analysis-api endpoint format.
    
    Args:
        file: CSV or Excel file to analyze
        question: Natural language question about the data
        model: Model parameter (ignored, uses environment config)
        encoding: Encoding parameter (ignored, auto-detected)
        delimiter: Delimiter parameter (ignored, auto-detected)
        
    Returns:
        Analysis result with answer and generated code
    """
    # Save uploaded file to temp location
    suffix = ".csv" if file.filename and file.filename.endswith(".csv") else ".xlsx"
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        temp_path = tmp.name
    
    try:
        # Analyze file
        result = agent.analyze(temp_path, question)
        
        # Add explanation if successful
        if result['status'] == 'success' and result.get('executed_code'):
            result['explanation'] = f"Analysis completed using LangChain ReAct agent with {GEMMA_3_DEPLOYMENT}"
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_file, temp_path)
        
        return AnalysisResponse(**result)
    
    except Exception as e:
        # Cleanup on error
        background_tasks.add_task(cleanup_file, temp_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/base64", response_model=AnalysisResponse)
async def analyze_base64(
    background_tasks: BackgroundTasks,
    request: Base64AnalysisRequest,
    x_jeen_csv_service: str = Header(None, alias="x-jeen-csv-service")
):
    """
    Analyze a base64-encoded CSV or Excel file.
    Compatible with playground-backend format.
    
    Args:
        request: Request containing base64 content, file type, question, model, provider
        
    Returns:
        Analysis result with answer and generated code
    """
    # Verify API key
    await verify_api_key(x_jeen_csv_service)
    
    # Log request for debugging
    print(f"\n[API] Received analysis request:")
    print(f"- Question: {request.question[:100]}...")
    print(f"- Model: {request.model}")
    print(f"- Provider: {request.provider}")
    print(f"- File type: {request.file_type}\n")
    
    # Decode base64 content
    try:
        content = request.content
        if "," in content:
            content = content.split(",", 1)[1]
        decoded = base64.b64decode(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 content: {e}")
    
    # Save to temp file
    suffix = f".{request.file_type.lower()}"
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix=suffix) as tmp:
        tmp.write(decoded)
        temp_path = tmp.name
    
    try:
        # Analyze file
        result = agent.analyze(temp_path, request.question)
        
        # Add explanation if successful
        if result['status'] == 'success' and result.get('executed_code'):
            result['explanation'] = f"Analysis completed using LangChain ReAct agent with {GEMMA_3_DEPLOYMENT}"
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_file, temp_path)
        
        return AnalysisResponse(**result)
    
    except Exception as e:
        # Cleanup on error
        background_tasks.add_task(cleanup_file, temp_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/excel/base64", response_model=AnalysisResponse)
async def analyze_excel_base64(
    background_tasks: BackgroundTasks,
    request: Base64AnalysisRequest,
    x_jeen_csv_service: str = Header(None, alias="x-jeen-csv-service")
):
    """
    Analyze a base64-encoded Excel file (alternative endpoint for Excel).
    Compatible with playground-backend format.
    
    Args:
        request: Request containing base64 content, question, model, provider
        
    Returns:
        Analysis result with answer and generated code
    """
    # Verify API key
    await verify_api_key(x_jeen_csv_service)
    
    # Log request
    print(f"\n[API] Received Excel analysis request:")
    print(f"- Question: {request.question[:100]}...")
    print(f"- Model: {request.model}")
    print(f"- Provider: {request.provider}\n")
    
    # Force Excel file type
    request.file_type = "xlsx"
    
    # Decode base64 content
    try:
        content = request.content
        if "," in content:
            content = content.split(",", 1)[1]
        decoded = base64.b64decode(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 content: {e}")
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.xlsx') as tmp:
        tmp.write(decoded)
        temp_path = tmp.name
    
    try:
        # Analyze file
        result = agent.analyze(temp_path, request.question)
        
        # Add explanation if successful
        if result['status'] == 'success' and result.get('executed_code'):
            result['explanation'] = f"Analysis completed using LangChain ReAct agent with {GEMMA_3_DEPLOYMENT}"
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_file, temp_path)
        
        return AnalysisResponse(**result)
    
    except Exception as e:
        # Cleanup on error
        background_tasks.add_task(cleanup_file, temp_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def get_models(api_key: str = Depends(verify_api_key)):
    """Get current model configuration"""
    return {
        "current_model": GEMMA_3_DEPLOYMENT,
        "endpoint": VLLM_ENDPOINT,
        "provider": "vLLM",
        "description": "LangChain ReAct agent with Gemma 3 12B"
    }


@app.get("/config")
async def get_config(api_key: str = Depends(verify_api_key)):
    """Get current configuration (admin endpoint)"""
    return {
        "model": GEMMA_3_DEPLOYMENT,
        "endpoint": VLLM_ENDPOINT,
        "port": SERVER_PORT,
        "timeout": SERVER_TIMEOUT,
        "max_retries": agent.max_retries,
        "temperature": agent.temperature
    }


if __name__ == "__main__":
   # print("=" * 80)
    #print("Data Analysis Agent API - Starting Server")
    #print("=" * 80)
    #print(f"Model: {GEMMA_3_DEPLOYMENT}")
    #print(f"Endpoint: {VLLM_ENDPOINT}")
    #print(f"Port: {SERVER_PORT}")
    #print(f"Timeout: {SERVER_TIMEOUT}s")
    print("=" * 80)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=SERVER_PORT,
        timeout_keep_alive=SERVER_TIMEOUT,
        log_level="info"
    )

