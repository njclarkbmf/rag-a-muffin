# Troubleshooting Guide for RAG-A-Muffin

This document provides solutions for common issues you might encounter when setting up and using the RAG-A-Muffin system.

## Installation Issues

### Python Dependency Errors

**Issue**: `pip install -r requirements.txt` fails with dependency conflicts.

**Solution**: 
1. Try creating a fresh virtual environment:
   ```bash
   python -m venv venv-new
   source venv-new/bin/activate  # On Windows: venv-new\Scripts\activate
   pip install -r requirements.txt
   ```

2. If specific packages are causing issues, you can try installing them manually:
   ```bash
   pip install fastapi==0.105.0 uvicorn==0.24.0
   pip install openai==1.3.7
   pip install lancedb==0.3.3
   # Continue with other packages
   ```

### Docker Build Failures

**Issue**: Docker build fails with errors.

**Solution**:
1. Make sure Docker is properly installed and running:
   ```bash
   docker --version
   docker-compose --version
   ```

2. Check for disk space issues:
   ```bash
   df -h  # On Linux/Mac
   ```

3. Try cleaning Docker's cache and rebuilding:
   ```bash
   docker system prune -f
   docker-compose build --no-cache
   ```

## Connection Issues

### Cannot Connect to the API

**Issue**: Client cannot connect to the API server.

**Solution**:
1. Verify the server is running:
   ```bash
   curl http://localhost:8000/api/collections -H "X-API-Key: your-api-key"
   ```

2. Check firewall settings and make sure port 8000 is open.

3. If using Docker, ensure port mapping is correct:
   ```bash
   docker-compose ps
   ```

### Authentication Failures

**Issue**: Receiving 401 Unauthorized errors.

**Solution**:
1. Verify the API key in your `.env` file matches the one you're using in requests.

2. Ensure the `X-API-Key` header is included in all API requests.

3. Check the server logs for any authentication issues:
   ```bash
   docker-compose logs
   ```

## LanceDB Issues

### LanceDB Connection Failures

**Issue**: Application fails to connect to LanceDB.

**Solution**:
1. Verify the LanceDB URI in your `.env` file:
   ```
   LANCEDB_URI=/data/lancedb  # For Docker
   LANCEDB_URI=./lancedb-data  # For local installation
   ```

2. Ensure the directory exists and has appropriate permissions:
   ```bash
   mkdir -p ./lancedb-data
   chmod 777 ./lancedb-data  # Only for testing, use proper permissions in production
   ```

3. Check if there's any corruption in the database:
   ```bash
   rm -rf ./lancedb-data  # WARNING: This deletes your database
   mkdir -p ./lancedb-data
   ```

### Vector Embedding Generation Failures

**Issue**: Error when generating embeddings: "Error: The model: text-embedding-3-small does not exist".

**Solution**:
1. Verify your OpenAI API key is correct and has access to the embedding model.

2. Try using a different embedding model in your `.env` file:
   ```
   EMBEDDING_MODEL=text-embedding-ada-002
   ```

3. Check the OpenAI API status page for any service disruptions.

## OpenAI Assistant Issues

### Assistant Creation Failures

**Issue**: System fails to create an assistant with OpenAI.

**Solution**:
1. Verify your OpenAI API key has appropriate permissions.

2. Check your OpenAI API usage limits and billing status.

3. Try specifying an existing assistant ID in your `.env` file:
   ```
   OPENAI_ASSISTANT_ID=asst_abc123
   ```

### Slow Responses

**Issue**: Assistant responses take a very long time to generate.

**Solution**:
1. Reduce the context window by retrieving fewer documents:
   ```bash
   python client.py --chat "Your question" --max_results 3
   ```

2. Use a faster model in your `.env` file:
   ```
   OPENAI_MODEL=gpt-3.5-turbo
   ```

3. Check your internet connection speed and stability.

### Response Timeout Errors

**Issue**: Requests time out before receiving a response.

**Solution**:
1. Increase the timeout setting in the client:
   ```python
   # In client.py, increase the timeout value
   start_time = time.time()
   while status in ["processing", "queued", "in_progress"]:
       # Check timeout - change from 60 to a higher value
       if time.time() - start_time > 120:  # 2 minutes
           print("✗ Timed out waiting for response.")
           break
   ```

2. For long-running queries, consider using the asynchronous API pattern:
   ```bash
   # Start the request
   python client.py --chat "Complex question" --wait_for_response=False
   
   # Check for the response later using the thread and run IDs
   python client.py --check_response --thread thread_abc123 --run run_xyz789
   ```

## Client Issues

### Command-Line Client Errors

**Issue**: Client script throws Python errors when running commands.

**Solution**:
1. Verify you're using the correct Python version:
   ```bash
   python --version  # Should be 3.9 or higher
   ```

2. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

3. Check for syntax errors in client.py:
   ```bash
   python -m py_compile client.py
   ```

### Interactive Mode Problems

**Issue**: Interactive mode crashes or behaves unexpectedly.

**Solution**:
1. Try running in debug mode for more information:
   ```bash
   python client.py --interactive --debug
   ```

2. Check the terminal supports the input method being used.

3. Try updating the client code to use a more compatible input method:
   ```python
   # Replace
   user_input = input("\n> ")
   
   # With
   import sys
   print("\n> ", end="", flush=True)
   user_input = sys.stdin.readline().strip()
   ```

## Data Management Issues

### Failed to Add Documents

**Issue**: Error when adding documents to LanceDB.

**Solution**:
1. Check the document format:
   ```bash
   python client.py --add "Short test document" --collection "test"
   ```

2. Verify metadata is valid JSON:
   ```bash
   python client.py --add "Test" --metadata '{"key": "value"}'  # Valid
   python client.py --add "Test" --metadata '{key: value}'  # Invalid
   ```

3. Check LanceDB table permissions and schema compatibility.

### Missing or Incomplete Search Results

**Issue**: Searches return no results or irrelevant results.

**Solution**:
1. Verify documents exist in the collection:
   ```bash
   python client.py --list_documents --collection "your_collection"
   ```

2. Try a more general query:
   ```bash
   python client.py --chat "What information do you have?"
   ```

3. Check embedding similarity thresholds:
   ```python
   # In app.py, adjust the similarity threshold
   table_results = table.search(query_embedding).limit(10).to_pandas()  # Increase limit
   ```

## Performance Optimization

### Slow Vector Search

**Issue**: Vector searches take too long with large document collections.

**Solution**:
1. Use LanceDB's optimized index:
   ```python
   # When creating the table, add indexing configuration
   db.create_table(
       collection,
       data=pd.DataFrame(...),
       mode="overwrite",
       index_params={
           "vector": {
               "type": "IVF",
               "metric_type": "L2",
               "num_partitions": 256,
               "num_sub_vectors": 96
           }
       }
   )
   ```

2. Optimize document chunking size:
   ```python
   # Use smaller chunks for faster retrieval
   chunks = chunk_text(text, chunk_size=500, overlap=100)
   ```

3. Implement caching for frequent queries:
   ```python
   # Add a simple in-memory cache
   query_cache = {}
   
   def search_with_cache(query, collection):
       cache_key = f"{query}:{collection}"
       if cache_key in query_cache:
           return query_cache[cache_key]
       
       results = search_lancedb(query, [collection])
       query_cache[cache_key] = results
       return results
   ```

### Memory Usage Issues

**Issue**: Application uses too much memory with large collections.

**Solution**:
1. Implement pagination for large result sets:
   ```python
   # In client.py, add pagination options
   parser.add_argument("--page", type=int, default=1, help="Page number")
   parser.add_argument("--page_size", type=int, default=10, help="Results per page")
   ```

2. Optimize embedding storage by using quantization in LanceDB.

3. Use streaming responses for large output:
   ```python
   # In app.py, use streaming responses
   @app.get("/api/stream_documents/{collection}")
   async def stream_documents(collection: str, authenticated: bool = Depends(verify_api_key)):
       def generate():
           table = db.open_table(collection)
           for i, row in enumerate(table.to_lance().to_pandas().iterrows()):
               yield json.dumps({"index": i, "document": row[1].to_dict()}) + "\n"
               
       return StreamingResponse(generate(), media_type="application/x-ndjson")
   ```

## Advanced Issues

### Deployment to Production

**Issue**: Need to deploy the system to a production environment.

**Solution**:
1. Set up HTTPS with a proper certificate:
   ```
   # In docker-compose.yml, add a reverse proxy like Nginx or Traefik
   services:
     rag-api:
       # Existing configuration...
       
     nginx:
       image: nginx:latest
       ports:
         - "443:443"
       volumes:
         - ./nginx.conf:/etc/nginx/conf.d/default.conf
         - ./certs:/etc/nginx/certs
   ```

2. Implement proper API key management:
   ```
   # Use environment-specific .env files
   .env.production
   .env.development
   ```

3. Set up monitoring and logging:
   ```
   # Add logging configuration in app.py
   import logging
   
   logging.basicConfig(
       filename="rag-a-muffin.log",
       level=logging.INFO,
       format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
   )
   ```

### Handling Multiple Models

**Issue**: Need to support multiple OpenAI models or alternative providers.

**Solution**:
1. Implement a model abstraction layer:
   ```python
   # Create a models.py file with provider abstractions
   class ModelProvider:
       def get_embedding(self, text):
           pass
           
       def create_completion(self, messages):
           pass
   
   class OpenAIProvider(ModelProvider):
       # OpenAI-specific implementation
       
   class AnthropicProvider(ModelProvider):
       # Anthropic-specific implementation
   ```

2. Make the provider configurable:
   ```
   # In .env
   MODEL_PROVIDER=openai
   ALTERNATIVE_PROVIDER=anthropic
   ```

3. Implement fallback strategies:
   ```python
   def get_completion_with_fallback(messages):
       try:
           return primary_provider.create_completion(messages)
       except Exception as e:
           logging.warning(f"Primary provider failed: {e}")
           return fallback_provider.create_completion(messages)
   ```

### Data Privacy Concerns

**Issue**: Need to handle sensitive data that shouldn't be sent to OpenAI.

**Solution**:
1. Implement PII detection and redaction:
   ```python
   def redact_sensitive_information(text):
       # Redact email addresses
       text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
       
       # Redact phone numbers
       text = re.sub(r'\b(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b', '[PHONE]', text)
       
       # Redact credit card numbers
       text = re.sub(r'\b(?:\d{4}[- ]?){3}\d{4}\b', '[CREDIT_CARD]', text)
       
       return text
   ```

2. Create a collection classification system:
   ```python
   # In .env
   SENSITIVE_COLLECTIONS=customer_data,financial_records
   PUBLIC_COLLECTIONS=marketing,documentation
   ```

3. Implement different handling based on collection sensitivity:
   ```python
   def is_sensitive_collection(collection_name):
       sensitive_collections = os.getenv("SENSITIVE_COLLECTIONS", "").split(",")
       return collection_name in sensitive_collections
       
   def process_query(query, collection):
       if is_sensitive_collection(collection):
           # Use more restricted handling
           # Maybe use a local model or additional safeguards
       else:
           # Standard handling
   ```

## Need More Help?

If you're still experiencing issues after trying these solutions:

1. Check the GitHub repository for updates and open issues
2. Join the community discussion on Discord/Slack (if available)
3. Create a detailed issue on GitHub with:
   - System information
   - Steps to reproduce
   - Exact error messages
   - Logs and configuration (with sensitive information removed)

For urgent support or security-related issues, contact the maintainers directly.
