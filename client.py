# client.py - Command-line client for the RAG-A-Muffin system
import requests
import time
import json
import argparse
import os
import sys
from dotenv import load_dotenv
from tqdm import tqdm
import textwrap
import colorama
from colorama import Fore, Style
import concurrent.futures
import datetime
import csv
import pandas as pd

# Initialize colorama
colorama.init()

# Load environment variables
load_dotenv()

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000/api")
API_KEY = os.getenv("API_KEY", "your-api-key")

# Headers
headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

# Debug mode flag
DEBUG = False

# Configure how many documents to retrieve per collection
DEFAULT_MAX_RESULTS = int(os.getenv("DEFAULT_MAX_RESULTS", "5"))

# Configure timeout in seconds
DEFAULT_TIMEOUT = int(os.getenv("DEFAULT_TIMEOUT", "120"))

def debug_print(message):
    """Print debug messages if debug mode is enabled"""
    if DEBUG:
        print(f"{Fore.CYAN}[DEBUG] {message}{Style.RESET_ALL}")

def print_banner():
    """Print the application banner"""
    banner = f"""
{Fore.GREEN}╭───────────────────────────────────────╮
│                                       │
│           RAG-A-MUFFIN Client         │
│      Hybrid RAG with LanceDB & OpenAI │
│                                       │
╰───────────────────────────────────────╯{Style.RESET_ALL}
"""
    print(banner)

def add_document(text, collection="default", metadata=None, batch_mode=False):
    """
    Add a document to the LanceDB collection.
    
    Args:
        text (str): The document text
        collection (str): Collection name
        metadata (dict): Optional metadata
        batch_mode (bool): If True, reduces output for batch processing
        
    Returns:
        dict: Response from the API
    """
    if metadata is None:
        metadata = {}
    
    data = {
        "text": text,
        "collection": collection,
        "metadata": metadata
    }
    
    if not batch_mode:
        print(f"\n{Fore.BLUE}Adding document to {collection}...{Style.RESET_ALL}")
        # Show a progress bar for the API call
        with tqdm(total=100, desc="Adding document", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            pbar.update(10)  # Start progress
            
            try:
                response = requests.post(
                    f"{API_URL}/documents",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                pbar.update(90)  # Complete progress
            except requests.exceptions.RequestException as e:
                pbar.update(90)  # Complete progress even on error
                print(f"{Fore.RED}✗ Error: {str(e)}{Style.RESET_ALL}")
                return None
    else:
        # Simple version for batch operations
        try:
            response = requests.post(
                f"{API_URL}/documents",
                headers=headers,
                json=data,
                timeout=30
            )
        except requests.exceptions.RequestException as e:
            return None
    
    if response.status_code == 200:
        if not batch_mode:
            print(f"{Fore.GREEN}✓ Document added successfully!{Style.RESET_ALL}")
        return response.json()
    else:
        if not batch_mode:
            print(f"{Fore.RED}✗ Error adding document: {response.text}{Style.RESET_ALL}")
        return None

def batch_add_documents(documents):
    """
    Add multiple documents in batch mode with progress bar.
    
    Args:
        documents (list): List of document dicts with text, collection, and metadata
        
    Returns:
        int: Number of successfully added documents
    """
    print(f"\n{Fore.BLUE}Adding {len(documents)} documents in batch mode...{Style.RESET_ALL}")
    
    successful = 0
    failed = 0
    
    with tqdm(total=len(documents), desc="Adding documents", 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Start the load operations and mark each future with its index
            future_to_index = {
                executor.submit(
                    add_document, 
                    doc["text"], 
                    doc.get("collection", "default"), 
                    doc.get("metadata", {}),
                    True  # batch mode
                ): i for i, doc in enumerate(documents)
            }
            
            # Process as they complete
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result = future.result()
                    if result:
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"{Fore.RED}Error processing document {idx}: {e}{Style.RESET_ALL}")
                    failed += 1
                
                pbar.update(1)
    
    print(f"{Fore.GREEN}✓ Added {successful} documents successfully{Style.RESET_ALL}")
    if failed > 0:
        print(f"{Fore.RED}✗ Failed to add {failed} documents{Style.RESET_ALL}")
    
    return successful

def chat(message, thread_id=None, collections=None, wait_for_response=True, max_results=DEFAULT_MAX_RESULTS):
    """
    Send a message to the assistant and optionally wait for response.
    
    Args:
        message (str): The message to send
        thread_id (str): Optional thread ID for conversation continuity
        collections (list): Collections to search
        wait_for_response (bool): Whether to wait for the full response
        max_results (int): Maximum number of results to retrieve per collection
        
    Returns:
        dict: Response from the API
    """
    if collections is None:
        collections = ["default"]
    
    data = {
        "message": message,
        "thread_id": thread_id,
        "collections": collections,
        "retrieval_enabled": True,
        "max_results": max_results
    }
    
    # Initial request to start processing
    print(f"\n{Fore.BLUE}Sending message to assistant...{Style.RESET_ALL}")
    
    try:
        response = requests.post(
            f"{API_URL}/chat",
            headers=headers,
            json=data,
            timeout=30
        )
    except requests.exceptions.RequestException as e:
        print(f"{Fore.RED}✗ Error: {str(e)}{Style.RESET_ALL}")
        return None
    
    if response.status_code != 200:
        print(f"{Fore.RED}✗ Error: {response.text}{Style.RESET_ALL}")
        return None
    
    result = response.json()
    thread_id = result["thread_id"]
    run_id = result["run_id"]
    
    # Report on retrieved context
    if result.get("retrieved_context"):
        print(f"\n{Fore.YELLOW}📚 Retrieved context:{Style.RESET_ALL}")
        for i, ctx in enumerate(result["retrieved_context"]):
            source = ctx["metadata"].get("source", "unknown")
            score = ctx["metadata"].get("relevance_score", 0)
            # Format the first 100 chars of text nicely
            preview = ctx['text'][:100].replace('\n', ' ')
            if len(ctx['text']) > 100:
                preview += "..."
            print(f"  {Fore.CYAN}[{i+1}]{Style.RESET_ALL} From {Fore.MAGENTA}{source}{Style.RESET_ALL} (score: {score:.2f})")
            print(f"      {preview}")
    
    # If we don't need to wait for the full response
    if not wait_for_response:
        print(f"\n{Fore.YELLOW}⏳ Processing in thread {thread_id}, run {run_id}{Style.RESET_ALL}")
        return result
    
    # Poll for completion with a nice progress bar
    print(f"\n{Fore.BLUE}⏳ Waiting for assistant response...{Style.RESET_ALL}")
    status = "processing"
    
    start_time = time.time()
    spinner = ["|", "/", "-", "\\"]
    spinner_idx = 0
    
    # Create a progress bar that doesn't know the total time
    with tqdm(desc="Processing", bar_format="{l_bar}{bar}| {elapsed}", leave=True) as pbar:
        while status in ["processing", "queued", "in_progress"]:
            # Check timeout
            if time.time() - start_time > DEFAULT_TIMEOUT:
                print(f"{Fore.RED}✗ Timed out waiting for response after {DEFAULT_TIMEOUT} seconds.{Style.RESET_ALL}")
                break
                
            # Update spinner for visual feedback
            spinner_char = spinner[spinner_idx % len(spinner)]
            pbar.set_description(f"Processing {spinner_char}")
            spinner_idx += 1
            
            # Update the progress bar
            pbar.update(1)
            
            # Wait before checking again
            time.sleep(1)
            
            # Check status
            try:
                response = requests.get(
                    f"{API_URL}/messages/{thread_id}/{run_id}",
                    headers=headers
                )
            except requests.exceptions.RequestException as e:
                print(f"{Fore.RED}✗ Error checking status: {str(e)}{Style.RESET_ALL}")
                break
                
            if response.status_code != 200:
                print(f"{Fore.RED}✗ Error checking status: {response.text}{Style.RESET_ALL}")
                break
                
            result = response.json()
            status = result.get("status")
            
            if status == "completed":
                # Clear progress bar for cleaner output
                pbar.close()
                print(f"\n{Fore.GREEN}🤖 Assistant response:{Style.RESET_ALL}")
                
                # Format and print the response
                response_text = result["message"]
                # Format the response with word wrapping
                wrapper = textwrap.TextWrapper(width=80, break_long_words=False, replace_whitespace=False)
                formatted_response = "\n".join(wrapper.fill(line) for line in response_text.splitlines())
                
                print(formatted_response)
                
                # Add thread_id to result for future reference
                result["thread_id"] = thread_id
                return result
    
    print(f"{Fore.RED}✗ Request ended with status: {status}{Style.RESET_ALL}")
    return None

def list_collections():
    """
    List all available collections in the database with statistics.
    
    Returns:
        list: Available collections with stats
    """
    print(f"\n{Fore.BLUE}Fetching collections...{Style.RESET_ALL}")
    
    try:
        with tqdm(total=100, desc="Loading", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            pbar.update(30)
            
            response = requests.get(
                f"{API_URL}/collections",
                headers=headers,
                timeout=30
            )
            
            pbar.update(70)
            
    except requests.exceptions.RequestException as e:
        print(f"{Fore.RED}✗ Error: {str(e)}{Style.RESET_ALL}")
        return []
    
    if response.status_code == 200:
        collections = response.json().get("collections", [])
        
        print(f"\n{Fore.YELLOW}📋 Available collections:{Style.RESET_ALL}")
        if collections:
            # Calculate column widths for nice formatting
            max_name_len = max(len(c["collection"]) for c in collections) + 2
            
            # Print header
            print(f"  {Fore.CYAN}{'Collection'.ljust(max_name_len)} | {'Documents'.center(10)} | {'Last Updated'.center(25)}{Style.RESET_ALL}")
            print(f"  {'-' * max_name_len}-+-{'-' * 10}-+-{'-' * 25}")
            
            # Print each collection with stats
            for i, collection in enumerate(collections):
                name = collection["collection"]
                count = collection["document_count"]
                last_updated = collection.get("last_updated", "N/A")
                
                # Format the date nicer if it exists
                if last_updated and last_updated != "N/A":
                    try:
                        dt = datetime.datetime.fromisoformat(last_updated)
                        last_updated = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        pass
                
                print(f"  {Fore.GREEN}{name.ljust(max_name_len)}{Style.RESET_ALL} | {str(count).center(10)} | {last_updated.center(25)}")
            
        else:
            print(f"  {Fore.YELLOW}No collections found{Style.RESET_ALL}")
        
        return collections
    else:
        print(f"{Fore.RED}✗ Error listing collections: {response.text}{Style.RESET_ALL}")
        return []

def import_csv(filename, collection="default", text_column=None, batch_size=100):
    """
    Import documents from a CSV file.
    
    Args:
        filename (str): Path to the CSV file
        collection (str): Collection to import into
        text_column (str): Column to use as document text
        batch_size (int): Number of documents to process in each batch
        
    Returns:
        int: Number of documents imported
    """
    print(f"\n{Fore.BLUE}Importing documents from {filename}...{Style.RESET_ALL}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(filename)
        total_rows = len(df)
        print(f"{Fore.YELLOW}Found {total_rows} rows in CSV file{Style.RESET_ALL}")
        
        # If text_column not specified, try to guess or use the first column
        if not text_column:
            columns = df.columns.tolist()
            text_column_candidates = ['text', 'content', 'document', 'description']
            for candidate in text_column_candidates:
                if candidate in columns:
                    text_column = candidate
                    break
            
            if not text_column:
                text_column = columns[0]
                
            print(f"{Fore.YELLOW}Using column '{text_column}' as document text{Style.RESET_ALL}")
        
        # Check if the column exists
        if text_column not in df.columns:
            print(f"{Fore.RED}✗ Column '{text_column}' not found in CSV{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Available columns: {', '.join(df.columns)}{Style.RESET_ALL}")
            return 0
        
        # Process in batches
        batches = [df[i:i+batch_size] for i in range(0, total_rows, batch_size)]
        total_imported = 0
        
        print(f"{Fore.BLUE}Processing {len(batches)} batches with batch size {batch_size}{Style.RESET_ALL}")
        
        for i, batch_df in enumerate(batches):
            print(f"\n{Fore.BLUE}Processing batch {i+1}/{len(batches)}{Style.RESET_ALL}")
            
            # Convert batch to list of document dicts
            documents = []
            for _, row in batch_df.iterrows():
                text = str(row[text_column])
                if pd.isna(text) or text.strip() == "":
                    continue  # Skip empty entries
                    
                # Extract other columns as metadata
                metadata = {}
                for col in df.columns:
                    if col != text_column and not pd.isna(row[col]):
                        val = row[col]
                        # Convert pandas types to native Python types
                        if isinstance(val, (pd.Timestamp, pd._libs.tslibs.timestamps.Timestamp)):
                            val = val.isoformat()
                        elif isinstance(val, (pd.Int64Dtype, pd.Float64Dtype)):
                            val = float(val) if '.' in str(val) else int(val)
                        metadata[col] = val
                        
                documents.append({
                    "text": text,
                    "collection": collection,
                    "metadata": metadata
                })
            
            # Add the batch
            successful = batch_add_documents(documents)
            total_imported += successful
            
            # Show progress
            progress = ((i+1) / len(batches)) * 100
            print(f"{Fore.GREEN}Overall progress: {progress:.1f}% ({total_imported}/{total_rows} documents){Style.RESET_ALL}")
        
        print(f"\n{Fore.GREEN}✓ Import complete! {total_imported} documents imported into collection '{collection}'{Style.RESET_ALL}")
        return total_imported
        
    except Exception as e:
        print(f"{Fore.RED}✗ Error importing CSV: {str(e)}{Style.RESET_ALL}")
        return 0

def interactive_mode():
    """
    Run an interactive chat session with the assistant.
    """
    thread_id = None
    collections = ["default"]
    
    print_banner()
    print(f"\n{Fore.GREEN}🤖 Welcome to the RAG-A-Muffin Interactive Console!{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Type 'exit' to quit, 'help' for commands, or start chatting.{Style.RESET_ALL}")
    
    # Show available collections
    list_collections()
    
    while True:
        try:
            user_input = input(f"\n{Fore.GREEN}> {Style.RESET_ALL}")
            
            # Handle special commands
            if user_input.lower() in ["exit", "quit"]:
                print(f"\n{Fore.YELLOW}👋 Goodbye!{Style.RESET_ALL}")
                break
                
            elif user_input.lower() == "help":
                print(f"\n{Fore.CYAN}Available commands:{Style.RESET_ALL}")
                print(f"  {Fore.YELLOW}exit, quit{Style.RESET_ALL} - Exit the application")
                print(f"  {Fore.YELLOW}help{Style.RESET_ALL} - Show this help message")
                print(f"  {Fore.YELLOW}collections{Style.RESET_ALL} - List available collections")
                print(f"  {Fore.YELLOW}use <collection1,collection2>{Style.RESET_ALL} - Switch collections")
                print(f"  {Fore.YELLOW}thread{Style.RESET_ALL} - Show current thread ID")
                print(f"  {Fore.YELLOW}reset{Style.RESET_ALL} - Reset conversation (start new thread)")
                print(f"  {Fore.YELLOW}import <filename.csv> [collection] [text_column]{Style.RESET_ALL} - Import CSV file")
                print(f"  {Fore.YELLOW}add <collection> <text>{Style.RESET_ALL} - Add a document to a collection")
                continue
                
            elif user_input.lower() == "collections":
                list_collections()
                continue
                
            elif user_input.lower().startswith("use "):
                # Change collections
                collection_list = user_input[4:].strip()
                new_collections = [c.strip() for c in collection_list.split(",")]
                collections = new_collections
                print(f"{Fore.GREEN}✓ Now using collections: {collections}{Style.RESET_ALL}")
                continue
                
            elif user_input.lower() == "thread":
                if thread_id:
                    print(f"{Fore.YELLOW}Current thread ID: {thread_id}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}No active thread (will create a new one on next message){Style.RESET_ALL}")
                continue
                
            elif user_input.lower() == "reset":
                thread_id = None
                print(f"{Fore.GREEN}✓ Conversation reset. Starting a new thread on next message.{Style.RESET_ALL}")
                continue
                
            elif user_input.lower().startswith("import "):
                # Import CSV
                parts = user_input[7:].strip().split()
                if len(parts) < 1:
                    print(f"{Fore.RED}✗ Import requires a filename{Style.RESET_ALL}")
                    continue
                    
                filename = parts[0]
                import_collection = parts[1] if len(parts) > 1 else "default"
                text_column = parts[2] if len(parts) > 2 else None
                
                import_csv(filename, import_collection, text_column)
                continue
                
            elif user_input.lower().startswith("add "):
                # Add document
                parts = user_input[4:].split(maxsplit=1)
                if len(parts) < 2:
                    print(f"{Fore.RED}✗ Add requires collection and text{Style.RESET_ALL}")
                    continue
                    
                add_collection = parts[0].strip()
                add_text = parts[1].strip()
                
                add_document(add_text, add_collection)
                continue
                
            # Normal chat
            result = chat(user_input, thread_id, collections)
            if result:
                thread_id = result["thread_id"]
                
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}👋 Exiting...{Style.RESET_ALL}")
            break
            
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
            if DEBUG:
                import traceback
                traceback.print_exc()

def check_health():
    """Check if the API is accessible"""
    try:
        response = requests.get(f"{API_URL.replace('/api', '')}/health", timeout=5)
        if response.status_code == 200:
            return True
        return False
    except:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG-A-Muffin Client")
    
    # Add arguments
    parser.add_argument("--add", help="Add document text")
    parser.add_argument("--collection", help="Collection name", default="default")
    parser.add_argument("--metadata", help="JSON metadata for document")
    parser.add_argument("--chat", help="Send a chat message")
    parser.add_argument("--thread", help="Thread ID for continuing conversation")
    parser.add_argument("--max-results", type=int, default=DEFAULT_MAX_RESULTS, help="Maximum results per collection")
    parser.add_argument("--wait", action="store_true", help="Wait for response (default)")
    parser.add_argument("--no-wait", action="store_false", dest="wait", help="Don't wait for response")
    parser.add_argument("--interactive", action="store_true", help="Start interactive chat mode")
    parser.add_argument("--list", action="store_true", help="List available collections")
    parser.add_argument("--import-csv", help="Import documents from CSV file")
    parser.add_argument("--text-column", help="Column to use as document text when importing CSV")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for CSV import")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Set debug mode if requested
    if args.debug:
        DEBUG = True
        debug_print("Debug mode enabled")
    
    # Check API health
    if not check_health():
        print(f"{Fore.RED}✗ Cannot connect to API at {API_URL.replace('/api', '')}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Please check that the server is running and API_URL is correctly configured.{Style.RESET_ALL}")
        sys.exit(1)
    
    # Execute requested action
    if args.add:
        metadata = {}
        if args.metadata:
            try:
                metadata = json.loads(args.metadata)
            except json.JSONDecodeError:
                print(f"{Fore.RED}Warning: Invalid JSON metadata, using empty metadata{Style.RESET_ALL}")
        
        add_document(args.add, args.collection, metadata)
    
    elif args.chat:
        collections = [args.collection] if args.collection else ["default"]
        chat(args.chat, args.thread, collections, wait_for_response=args.wait, max_results=args.max_results)
    
    elif args.list:
        list_collections()
    
    elif args.import_csv:
        import_csv(args.import_csv, args.collection, args.text_column, args.batch_size)
    
    elif args.interactive:
        interactive_mode()
    
    else:
        parser.print_help()
