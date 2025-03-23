# hybrid-rag.py - Main application file
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Union
import lancedb
import openai
import os
import time
import uuid
import json
import numpy as np
import pyarrow as pa
import pandas as pd
from dotenv import load_dotenv
import logging
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("rag-a-muffin.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag-a-muffin")

# Initialize FastAPI
app = FastAPI(
    title="RAG-A-Muffin", 
    description="Hybrid RAG System using LanceDB and OpenAI Assistants",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME)

# Initialize connections
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Connect to LanceDB - ensure the directory exists
lancedb_uri = os.getenv("LANCEDB_URI", "./lancedb-data")
os.makedirs(lancedb_uri, exist_ok=True)
db = lancedb.connect(lancedb_uri)

# Create or get Assistant (done once, or you can store the ID)
def get_or_create_assistant():
    """
    Retrieves an existing assistant or creates a new one if none exists.
    
    Returns:
        Assistant object from OpenAI
    """
    assistant_id = os.getenv("OPENAI_ASSISTANT_ID")
    if assistant_id:
        try:
            logger.info(f"Retrieving existing assistant: {assistant_id}")
            return openai_client.beta.assistants.retrieve(assistant_id)
        except Exception as e:
            logger.warning(f"Could not retrieve assistant {assistant_id}: {e}")
    
    # Create a new assistant if none exists
    logger.info("Creating new assistant")
    assistant = openai_client.beta.assistants.create(
        name="LanceDB Hybrid Assistant",
        instructions="""You are an assistant that answers questions based on retrieved information and your knowledge.
When information is retrieved from the database, prioritize that information over your general knowledge.
Always provide sources for information when available from the retrieved context.
Be concise and direct in your responses while being comprehensive and accurate.""",
        model=os.getenv("OPENAI_MODEL", "gpt-4-turbo"),
        tools=[{"type": "function", "function": {
            "name": "search_database",
            "description": "Search the database for relevant information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "collections": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The collections to search"
                    }
                },
                "required": ["query"]
            }
        }}]
    )
    # Save the assistant ID to environment variable
    os.environ["OPENAI_ASSISTANT_ID"] = assistant.id
    logger.info(f"Created new assistant with ID: {assistant.id}")
    return assistant

# Get embedding from OpenAI
def get_embedding(text):
    """
    Generates an embedding vector for the given text.
    
    Args:
        text (str): The text to embed
        
    Returns:
        list: The embedding vector
    """
    logger.debug(f"Generating embedding for text: {text[:50]}...")
    response = openai_client.embeddings.create(
        input=text,
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    return response.data[0].embedding

# Search LanceDB with progress tracking
def search_lancedb(query, collections=None, limit=5):
    """
    Searches for relevant documents in LanceDB collections.
    
    Args:
        query (str): The search query
        collections (list): Collections to search
        limit (int): Maximum number of results per collection
        
    Returns:
        list: Retrieved documents with metadata
    """
    if not collections:
        collections = ["default"]
    
    logger.info(f"Searching for '{query}' in collections: {collections}")
    
    try:
        query_embedding = get_embedding(query)
        query_embedding_np = np.array(query_embedding, dtype=np.float32)
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise Exception(f"Failed to generate embedding: {str(e)}")
    
    results = []
    
    # Log the total number of collections for progress reporting
    total_collections = len(collections)
    processed = 0
    
    for collection_name in collections:
        processed += 1
        logger.info(f"Searching collection {processed}/{total_collections}: {collection_name}")
        
        try:
            # Ensure the table exists
            if collection_name not in db.table_names():
                logger.warning(f"Collection {collection_name} does not exist, skipping")
                continue
                
            table = db.open_table(collection_name)
            
            # Search using the query embedding
            table_results = table.search(
                query_vector=query_embedding_np,
                vector_column_name="embedding",
                limit=limit
            ).to_pandas()
            
            logger.info(f"Found {len(table_results)} results in collection {collection_name}")
            
            # Process each result
            for _, row in table_results.iterrows():
                # Extract text and metadata
                text = row.get("text", "")
                metadata = {k: v for k, v in row.items() 
                           if k not in ["text", "embedding", "_distance"]}
                metadata["source"] = collection_name
                metadata["relevance_score"] = float(row.get("_distance", 0))
                
                results.append({
                    "text": text,
                    "metadata": metadata
                })
        except Exception as e:
            logger.error(f"Error searching collection {collection_name}: {e}")
    
    # Sort results by relevance score
    results.sort(key=lambda x: x["metadata"]["relevance_score"])
    logger.info(f"Total results across all collections: {len(results)}")
    return results

# API Request/Response Models
class MessageRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    collections: Optional[List[str]] = None
    model: Optional[str] = None
    retrieval_enabled: bool = True
    max_results: int = 5

class MessageResponse(BaseModel):
    thread_id: str
    message: str
    retrieved_context: Optional[List[Dict]] = None
    assistant_id: str
    run_id: str
    created_at: int
    status: str = "processing"

class DocumentRequest(BaseModel):
    text: str
    collection: str = "default"
    metadata: Optional[Dict[str, Union[str, int, float]]] = None

class CollectionStats(BaseModel):
    collection: str
    document_count: int
    last_updated: Optional[str] = None

# API Key verification
def verify_api_key(api_key: str = Depends(api_key_header)):
    """
    Verifies the API key from the request header.
    
    Args:
        api_key (str): The API key from the request header
        
    Returns:
        bool: True if verified, raises exception otherwise
    """
    expected_key = os.getenv("API_KEY")
    if not expected_key:
        logger.warning("API_KEY not set in environment variables")
        raise HTTPException(status_code=500, detail="API key not configured on server")
        
    if api_key != expected_key:
        logger.warning("Invalid API key attempted")
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return True

# Middleware for logging requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log all API requests.
    """
    start_time = time.time()
    path = request.url.path
    method = request.method
    
    logger.info(f"Request: {method} {path}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"Response: {method} {path} - Status: {response.status_code} - Time: {process_time:.3f}s")
    
    return response

# Endpoint to start or continue a conversation
@app.post("/api/chat", response_model=MessageResponse)
async def chat(
    request: MessageRequest,
    background_tasks: BackgroundTasks,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Handles chat requests by retrieving relevant information and sending to OpenAI Assistant.
    
    Args:
        request (MessageRequest): The chat request
        background_tasks (BackgroundTasks): FastAPI background tasks handler
        authenticated (bool): Verified by the API key dependency
        
    Returns:
        MessageResponse: Initial response with thread and run IDs
    """
    start_time = time.time()
    logger.info(f"Processing chat request: '{request.message[:50]}...'")
    
    # 1. Get or create assistant
    assistant = get_or_create_assistant()
    
    # 2. Get or create thread
    thread_id = request.thread_id
    if not thread_id:
        thread = openai_client.beta.threads.create()
        thread_id = thread.id
        logger.info(f"Created new thread: {thread_id}")
    else:
        logger.info(f"Using existing thread: {thread_id}")
    
    # 3. Perform retrieval if enabled
    retrieved_context = None
    if request.retrieval_enabled:
        try:
            logger.info("Retrieving context from LanceDB")
            retrieved_context = search_lancedb(
                query=request.message,
                collections=request.collections,
                limit=request.max_results
            )
            
            # Add retrieved context to the message
            if retrieved_context:
                logger.info(f"Adding {len(retrieved_context)} retrieved items to thread")
                context_text = "Here is relevant information from our database:\n\n"
                for i, item in enumerate(retrieved_context):
                    context_text += f"[{i+1}] {item['text']}\n"
                    if item['metadata'].get('source'):
                        context_text += f"Source: {item['metadata']['source']}\n"
                    context_text += "\n"
                
                # Add context as a system message
                openai_client.beta.threads.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content=context_text
                )
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            # Continue without retrieval if it fails
    
    # 4. Add user message
    logger.info("Adding user message to thread")
    openai_client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=request.message
    )
    
    # 5. Run the assistant
    logger.info("Starting assistant run")
    run = openai_client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant.id
    )
    
    # Calculate processing time
    process_time = time.time() - start_time
    logger.info(f"Initial request processed in {process_time:.3f}s, run started with ID: {run.id}")
    
    # 6. Poll for completion (in background for async operation)
    def check_run_status():
        """Background task to monitor the run status"""
        run_status = run.status
        start_time = time.time()
        
        logger.info(f"Background task started for run {run.id}")
        
        # Poll until the run completes or fails
        while run_status in ["queued", "in_progress"]:
            # Check if we've been waiting too long
            elapsed = time.time() - start_time
            if elapsed > 120:  # 2 minute timeout
                logger.warning(f"Run {run.id} timed out after {elapsed:.1f}s")
                break
            
            # Log progress every 5 seconds
            if int(elapsed) % 5 == 0:
                logger.info(f"Run {run.id} still {run_status} after {elapsed:.1f}s")
            
            # Wait before checking again
            time.sleep(1)
            
            # Get the updated run
            try:
                current_run = openai_client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id
                )
                run_status = current_run.status
            except Exception as e:
                logger.error(f"Error checking run status: {e}")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Run {run.id} completed with status {run_status} after {total_time:.1f}s")

    # Start background task to check status
    background_tasks.add_task(check_run_status)
    
    # 7. Return initial response
    return {
        "thread_id": thread_id,
        "message": "Processing your request...",
        "retrieved_context": retrieved_context,
        "assistant_id": assistant.id,
        "run_id": run.id,
        "created_at": int(time.time()),
        "status": "processing"
    }

# Endpoint to get a message
@app.get("/api/messages/{thread_id}/{run_id}")
async def get_message(
    thread_id: str,
    run_id: str,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Retrieves the status and message for a specific thread and run.
    
    Args:
        thread_id (str): The thread ID
        run_id (str): The run ID
        authenticated (bool): Verified by the API key dependency
        
    Returns:
        dict: Status and message information
    """
    logger.info(f"Checking status for thread {thread_id}, run {run_id}")
    
    # Check run status
    try:
        run = openai_client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id
        )
    except Exception as e:
        logger.error(f"Error retrieving run: {e}")
        raise HTTPException(status_code=404, detail=f"Run not found: {str(e)}")
    
    logger.info(f"Run status: {run.status}")
    
    if run.status in ["queued", "in_progress"]:
        return {
            "status": run.status,
            "message": "Processing your request..."
        }
    
    if run.status == "completed":
        # Get the latest assistant message
        try:
            messages = openai_client.beta.threads.messages.list(
                thread_id=thread_id
            )
        except Exception as e:
            logger.error(f"Error retrieving messages: {e}")
            raise HTTPException(status_code=404, detail=f"Messages not found: {str(e)}")
        
        # Find the first assistant message
        for message in messages.data:
            if message.role == "assistant":
                # Extract text content
                content_text = ""
                for content in message.content:
                    if content.type == "text":
                        content_text += content.text.value
                
                logger.info(f"Found assistant response of length {len(content_text)}")
                
                return {
                    "status": "completed",
                    "message": content_text,
                    "message_id": message.id,
                    "created_at": message.created_at
                }
        
        logger.warning("No assistant response found in completed run")
        return {
            "status": "error",
            "message": "No assistant response found"
        }
    
    # If run failed or requires action
    logger.warning(f"Run ended with non-completion status: {run.status}")
    return {
        "status": run.status,
        "message": f"Run ended with status: {run.status}"
    }

# Endpoint to register documents (add to the LanceDB database)
@app.post("/api/documents")
async def register_document(
    document: DocumentRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Adds a document to the LanceDB database.
    
    Args:
        document (DocumentRequest): The document to add
        authenticated (bool): Verified by the API key dependency
        
    Returns:
        dict: Status and document ID
    """
    # Extract document fields
    text = document.text
    collection = document.collection
    metadata = document.metadata or {}
    
    logger.info(f"Adding document to collection '{collection}': {text[:50]}...")
    
    # Generate embedding
    try:
        embedding = get_embedding(text)
        # Convert to numpy array
        embedding_np = np.array(embedding, dtype=np.float32)
        logger.info(f"Generated embedding of dimension {len(embedding)}")
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate embedding: {str(e)}")
    
    # Ensure table exists
    if collection not in db.table_names():
        logger.info(f"Creating new collection: {collection}")
        
        # Create table schema with PyArrow
        # Base schema fields
        fields = [
            pa.field("text", pa.string()),
            pa.field("embedding", pa.list_(pa.float32())),
            pa.field("timestamp", pa.timestamp('s'))  # Add timestamp field
        ]
        
        # Add fields for metadata
        for key, value in metadata.items():
            if isinstance(value, int):
                fields.append(pa.field(key, pa.int64()))
            elif isinstance(value, float):
                fields.append(pa.field(key, pa.float64()))
            else:
                fields.append(pa.field(key, pa.string()))
        
        schema = pa.schema(fields)
        
        # Create empty table with schema
        try:
            db.create_table(collection, schema=schema)
            logger.info(f"Table {collection} created with schema: {schema}")
        except Exception as e:
            logger.error(f"Failed to create table: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create table: {str(e)}")
    
    # Open the table
    try:
        table = db.open_table(collection)
    except Exception as e:
        logger.error(f"Failed to open table: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to open table: {str(e)}")
    
    # Add timestamp to data
    current_time = pd.Timestamp.now()
    
    # Prepare data
    data = {
        "text": text,
        "embedding": embedding_np,
        "timestamp": current_time,
        **metadata
    }
    
    # Add to table
    try:
        df = pd.DataFrame([data])
        table.add(df)
        
        # Generate a document ID
        document_id = str(uuid.uuid4())
        
        logger.info(f"Document added successfully with ID: {document_id}")
        
        return {
            "status": "success",
            "message": f"Document added to collection {collection}",
            "document_id": document_id,
            "timestamp": current_time.isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to add document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add document: {str(e)}")

# Endpoint to list available collections with stats
@app.get("/api/collections")
async def list_collections(authenticated: bool = Depends(verify_api_key)):
    """
    Lists all available collections with statistics.
    
    Args:
        authenticated (bool): Verified by the API key dependency
        
    Returns:
        dict: List of collections with stats
    """
    logger.info("Listing collections")
    
    collection_names = db.table_names()
    logger.info(f"Found {len(collection_names)} collections")
    
    # Get statistics for each collection
    stats = []
    for name in collection_names:
        try:
            table = db.open_table(name)
            # Get document count
            count = len(table)
            
            # Try to get the latest timestamp
            last_updated = None
            try:
                if "timestamp" in table.schema.names:
                    # Get most recent timestamp
                    latest = table.to_pandas().sort_values(
                        by="timestamp", ascending=False
                    ).head(1)
                    
                    if not latest.empty and "timestamp" in latest:
                        last_updated = latest["timestamp"].iloc[0].isoformat()
            except Exception as e:
                logger.warning(f"Could not get timestamp for {name}: {e}")
            
            stats.append(CollectionStats(
                collection=name,
                document_count=count,
                last_updated=last_updated
            ))
        except Exception as e:
            logger.error(f"Error getting stats for collection {name}: {e}")
            stats.append(CollectionStats(
                collection=name,
                document_count=-1  # Error indicator
            ))
    
    return {
        "collections": stats
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    
    Returns:
        dict: Status and version information
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"Starting RAG-A-Muffin on {host}:{port}")
    print("Press Ctrl+C to stop")
    
    uvicorn.run("hybrid-rag:app", host=host, port=port, reload=True)
