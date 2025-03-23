# RAG-A-Muffin

A hybrid Retrieval Augmented Generation (RAG) system that combines OpenAI's Assistants API with LanceDB for efficient vector search and retrieval.

## Overview

RAG-A-Muffin provides a powerful, flexible framework for building AI applications that leverage both the conversational capabilities of OpenAI's models and the custom retrieval capabilities of LanceDB. This hybrid approach gives you complete control over your knowledge base while maintaining the benefits of OpenAI's advanced conversation management.

The system allows you to:

- Store and organize documents in LanceDB collections
- Search for relevant information using vector similarity
- Create and manage conversations through OpenAI's Assistants API
- Combine retrieved information with the assistant's knowledge
- Build applications that respond intelligently to user queries

## Features

- **Multiple Knowledge Collections**: Organize information into separate collections in LanceDB
- **Thread Persistence**: Continue conversations across multiple interactions
- **Customizable Retrieval**: Control which collections to search and how many results to retrieve
- **Secure API**: API key authentication for all endpoints
- **Interactive Client**: Command-line interface for interacting with the system
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **Asynchronous Processing**: Efficient handling of long-running requests

## Installation

### Prerequisites

- Python 3.9 or higher
- OpenAI API key
- Docker and Docker Compose (optional, for containerized deployment)

### Option 1: Local Installation

1. Clone the repository:

```bash
git clone https://github.com/johnclark/rag-a-muffin.git
cd rag-a-muffin
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your configuration:

```bash
cp .env.example .env
# Edit .env with your settings
```

5. Run the application:

```bash
python hybrid-rag.py
```

### Option 2: Docker Deployment

1. Clone the repository:

```bash
git clone https://github.com/johnclark/rag-a-muffin.git
cd rag-a-muffin
```

2. Create a `.env` file with your configuration:

```bash
cp .env.example .env
# Edit .env with your settings
```

3. Build and start the containers:

```bash
docker-compose up -d
```

## Usage Examples

### Adding Documents to Your Knowledge Base

The system allows you to add documents to different collections in LanceDB. Each document consists of text content and optional metadata.

```bash
# Add a single document to the default collection
python client.py --add "LanceDB is a vector database designed for AI applications. It provides efficient storage and retrieval of vector embeddings, making it useful for similarity search and retrieval augmented generation (RAG) applications."

# Add a document to a specific collection with metadata
python client.py --add "OpenAI's Assistants API provides a way to build AI assistants within your applications. It handles conversation state, provides function calling capabilities, and manages threading." --collection "ai_technologies" --metadata '{"source":"OpenAI Documentation","date":"2023-11-15","tags":["OpenAI","API","Assistants"]}'

# Add a document about vector databases
python client.py --add "Vector databases are specialized database systems designed to store and query vector embeddings efficiently. They typically support similarity search operations like k-nearest neighbors (kNN) which are essential for AI applications." --collection "databases" --metadata '{"category":"database technology","complexity":"intermediate"}'
```

### Asking Questions and Getting Answers

You can ask questions directly through the command-line interface:

```bash
# Ask a question using the default collection
python client.py --chat "What is LanceDB?"

# Ask a question using a specific collection
python client.py --chat "Explain the benefits of OpenAI's Assistants API" --collection "ai_technologies"

# Continue a conversation using a thread ID
python client.py --chat "What are the alternatives?" --thread "thread_abc123"

# Search across multiple collections
python client.py --chat "Compare vector databases with traditional databases" --collection "databases,ai_technologies"
```

### Interactive Mode

For a more conversational experience, you can use the interactive mode:

```bash
python client.py --interactive
```

This will start a command-line chat interface where you can:
- Type messages to send to the assistant
- Switch between collections using `use collection1,collection2`
- List available collections by typing `collections`
- Exit the application by typing `exit`

Example interactive session:

```
🤖 Welcome to the Hybrid RAG Assistant!
Type 'exit' to quit, 'collections' to list available collections,
or 'use <collection1,collection2>' to switch collections.

> collections

📋 Available collections:
  1. default
  2. ai_technologies
  3. databases

> use ai_technologies,databases
✓ Now using collections: ['ai_technologies', 'databases']

> What's the difference between vector databases and traditional relational databases?

📚 Retrieved context:
  [1] From databases (score: 0.82)
      Vector databases are specialized database systems designed to store and query vector embeddings efficien...
  [2] From ai_technologies (score: 0.65)
      OpenAI's Assistants API provides a way to build AI assistants within your applications. It handles conv...

🤖 Assistant response:
Vector databases and traditional relational databases serve different purposes and have distinct characteristics:

Vector Databases (like LanceDB):
- Specifically designed to store and efficiently query vector embeddings (numerical representations of data)
- Optimize for similarity search operations like k-nearest neighbors (kNN)
- Excel at finding "similar" items rather than exact matches
- Primarily used for AI applications, machine learning, and semantic search
- Support operations like cosine similarity, Euclidean distance calculations
- Often have specialized indexing structures like HNSW (Hierarchical Navigable Small World) graphs

Traditional Relational Databases:
- Organize data in tables with rows and columns
- Optimize for structured queries with exact matching (SQL)
- Excel at transactions, joins, and aggregations
- Based on relational algebra and set theory
- Use indexes like B-trees for efficient lookups
- Enforce ACID properties (Atomicity, Consistency, Isolation, Durability)

Vector databases are ideal for AI applications where you need to find conceptually similar content, while relational databases excel at storing and querying structured data with precise relationships.

> exit
Exiting...
```

### Managing Collections

You can list all available collections:

```bash
python client.py --list
```

## API Endpoints

The system exposes the following RESTful API endpoints:

- `POST /api/chat`: Send a message to the assistant
- `GET /api/messages/{thread_id}/{run_id}`: Get a message from a specific thread and run
- `POST /api/documents`: Add a document to the LanceDB database
- `GET /api/collections`: List all available collections

All API requests require the `X-API-Key` header for authentication.

### Example API Request

```python
import requests

# Configuration
API_URL = "http://localhost:8000/api"
API_KEY = "your-api-key"

# Headers
headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

# Send a chat message
response = requests.post(
    f"{API_URL}/chat",
    headers=headers,
    json={
        "message": "What is the difference between RAG and traditional chatbots?",
        "collections": ["ai_technologies"],
        "retrieval_enabled": True,
        "max_results": 5
    }
)

result = response.json()
thread_id = result["thread_id"]
run_id = result["run_id"]

# Get the assistant's response
response = requests.get(
    f"{API_URL}/messages/{thread_id}/{run_id}",
    headers=headers
)

print(response.json()["message"])
```

## Architecture

RAG-A-Muffin follows a hybrid architecture that combines the best of both worlds:

1. **OpenAI Assistants API**: Handles conversation state, message processing, and response generation
2. **LanceDB**: Stores document embeddings and performs vector similarity search
3. **FastAPI Server**: Provides RESTful API endpoints and orchestrates the workflow
4. **Client Application**: Offers a user-friendly interface for interacting with the system

The workflow for a typical query is:

1. User sends a message through the client
2. Server generates an embedding for the message
3. LanceDB searches for relevant documents using vector similarity
4. Retrieved documents are formatted and added to the conversation
5. OpenAI Assistant processes the message and retrieved context
6. Assistant generates a response that incorporates the retrieved information
7. Response is returned to the user

## Configuration

The system can be configured through environment variables in the `.env` file:

- `API_KEY`: Secret key for API authentication
- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_MODEL`: Model to use (default: gpt-4-turbo)
- `EMBEDDING_MODEL`: Embedding model to use (default: text-embedding-3-small)
- `OPENAI_ASSISTANT_ID`: Optional ID of an existing assistant
- `LANCEDB_URI`: Path to the LanceDB database (default: ./lancedb-data)

## Troubleshooting

For common issues and their solutions, please see the [TROUBLESHOOTING.md](TROUBLESHOOTING.md) file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Created by John Clark

## Acknowledgments

- OpenAI for their Assistants API
- LanceDB for their vector database
- FastAPI for the web framework
