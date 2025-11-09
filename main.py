"""
FastAPI Main Application for Data Analysis Agent
Production-ready with parallel request support and API key authentication
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import tempfile
import os
import base64
import uvicorn

from agent import DataAnalysisAgent
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
agent = DataAnalysisAgent(**get_agent_config())


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


class MessageContent(BaseModel):
    """Content item in a message (text or file)"""
    type: str  # "text" or "file"
    text: Optional[str] = None
    filename: Optional[str] = None
    data: Optional[str] = None  # base64 encoded file data


class Message(BaseModel):
    """A message in the conversation"""
    role: str  # "user", "assistant", etc.
    content: List[MessageContent]


class AgentRunRequest(BaseModel):
    """Request model for /api/v1/agent/run endpoint - new format"""
    model: str
    messages: List[Message]


class AgentRunResponse(BaseModel):
    """Response model for /api/v1/agent/run endpoint"""
    response: str


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


@app.post("/api/v1/agent/run", response_model=AgentRunResponse)
async def agent_run(
    background_tasks: BackgroundTasks,
    request: AgentRunRequest,
    x_jeen_csv_service: str = Header(None, alias="x-jeen-csv-service")
):
    """
    New agent run endpoint compatible with updated CsvParserService.
    Accepts messages array with text and file content.
    
    Args:
        request: Request containing model and messages array
        
    Returns:
        Response with analysis result in response field
    """
    # Verify API key
    await verify_api_key(x_jeen_csv_service)
    
    # Extract the last user message (should contain text and file)
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    
    last_message = user_messages[-1]  # Get the last user message
    
    # Extract text instruction and file content
    instruction = None
    file_content = None
    filename = None
    
    for content_item in last_message.content:
        if content_item.type == "text":
            instruction = content_item.text
        elif content_item.type == "file":
            file_content = content_item.data
            filename = content_item.filename
    
    if not instruction:
        raise HTTPException(status_code=400, detail="No text instruction found in message")
    
    if not file_content:
        raise HTTPException(status_code=400, detail="No file content found in message")
    
    if not filename:
        raise HTTPException(status_code=400, detail="No filename found in message")
    
    # Log request
    print(f"\n[API] Received agent run request:")
    print(f"- Model: {request.model}")
    print(f"- Instruction: {instruction[:100]}...")
    print(f"- Filename: {filename}\n")
    
    # Decode base64 file content
    try:
        if "," in file_content:
            file_content = file_content.split(",", 1)[1]
        decoded = base64.b64decode(file_content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 file content: {e}")
    
    # Determine file type from filename
    file_type = "csv"
    if filename.lower().endswith(('.xlsx', '.xls')):
        file_type = "xlsx"
    
    # Save to temp file
    suffix = f".{file_type}"
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix=suffix) as tmp:
        tmp.write(decoded)
        temp_path = tmp.name
    
    try:
        # Analyze file
        result = agent.analyze(temp_path, instruction)
        
        # Format response to match new interface expectations
        if result['status'] == 'success':
            # Return the answer directly as response field
            response_text = result.get('answer', '')
        else:
            # Return error message
            response_text = f"Error: {result.get('error', 'Unknown error')}"
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_file, temp_path)
        
        return AgentRunResponse(response=response_text)
    
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

