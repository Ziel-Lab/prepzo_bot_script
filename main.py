from __future__ import annotations
import asyncio
import json
import logging
import uuid
import os
import pathlib
import time
import pickle
from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Optional, Tuple, List
from datetime import datetime, timezone
import ipaddress
import requests
from urllib.parse import urlparse
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    WorkerType,
    cli,
    llm,
    metrics,
    multimodal
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero, google
from livekit.plugins.openai import tts as openai_tts
from livekit.agents.llm import ChatMessage, ChatImage
from supabase import create_client, Client
from dotenv import load_dotenv
# from livekit.agents.tts import TTSService
import prompt
from openai import OpenAI
import version
import google.generativeai as genai
from google.genai import types
from pinecone import Pinecone, ServerlessSpec # Import Pinecone
import base64
import concurrent.futures
import unittest.mock
import traceback

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)

# Initialize Supabase client
supabase: Client = None

# Initialize OpenAI client for web search and embeddings
openai_client: OpenAI = None

# Initialize Gemini API flag
gemini_initialized: bool = False

# Initialize Pinecone client and index
pinecone_client: Pinecone = None
pinecone_index = None
PINECONE_INDEX_NAME = "coachingbooks"
EMBEDDING_MODEL = "text-embedding-3-large" # Match the index

# Global variables for conversation tracking
conversation_history = []
session_id = None
user_message = ""
timeout_task = None
agent = None

# OpenAI TTS configuration
OPENAI_VOICE_ID = "alloy"  

# Default interview preparation prompt
DEFAULT_INTERVIEW_PROMPT = prompt.prompt

# Local backup for conversations when Supabase is unreachable
LOCAL_BACKUP_DIR = pathlib.Path("./conversation_backups")
RETRY_QUEUE_FILE = LOCAL_BACKUP_DIR / "retry_queue.pkl"
MAX_RETRY_QUEUE_SIZE = 1000  # Maximum messages to store for retry

# Retry queue for failed message storage attempts
retry_queue = []

# Add a flag to control verbose logging
VERBOSE_LOGGING = False

# Add a helper for conditional logging
def verbose_log(message, level="info", error=None):
    """Conditionally log based on verbose setting to reduce processing during conversation"""
    if not VERBOSE_LOGGING:
        return
        
    if level == "info":
        logger.info(message)
    elif level == "debug":
        logger.debug(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
        if error:
            logger.error(f"Error details: {error}")
            
# Helper function to get current UTC time
def get_utc_now():
    """Get current UTC time in a timezone-aware manner"""
    try:
        # Python 3.11+ approach 
        return datetime.now(datetime.UTC)
    except AttributeError:
        # Fallback for earlier Python versions
        return datetime.now(timezone.utc)

# Helper function to get current UTC time with consistent formatting
def get_current_timestamp():
    """Get current UTC time in ISO 8601 format with timezone information"""
    return get_utc_now().isoformat()

# This function now primarily defines the tool for Gemini
# The actual search execution is handled by `handle_gemini_web_search`

def get_web_search_tool_declaration():
    """Returns the function declaration for the web search tool."""
    return {
        "name": "search_web",
        "description": "Searches the web for current information on a specific topic when internal knowledge is insufficient or outdated. Use only for recent events, specific factual data (like current salaries), or verifying contested information.",
        "parameters": {
            "type": "object",
            "properties": {
                "search_query": {
                    "type": "string",
                    "description": "The specific, optimized query to search for on the web."
                },
                "include_location": {
                    "type": "boolean",
                    "description": "Set to true if the user's location is relevant to the search (e.g., local job market)."
                }
            },
            "required": ["search_query"]
        }
    }

def get_knowledge_base_tool_declaration():
    """Returns the function declaration for the knowledge base tool."""
    return {
        "name": "query_knowledge_base",
        "description": "Searches an internal knowledge base of coaching books and principles for established concepts, strategies, and general career advice. Prioritize this over web search for foundational knowledge.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The specific question or topic to search for in the knowledge base."
                }
            },
            "required": ["query"]
        }
    }

# Removed the main logic from perform_web_search as it's now split:
# 1. Declaration: get_web_search_tool_declaration()
# 2. Execution: handle_gemini_web_search() (triggered by LLM)
# 3. Underlying API call: perform_actual_search()
# --- New Tool ---
# 4. Declaration: get_knowledge_base_tool_declaration()
# 5. Execution: handle_knowledge_base_query() (triggered by LLM)
# 6. Underlying API call: query_pinecone_knowledge_base()

async def perform_actual_search(search_query):
    """
    Perform the actual web search using SerpAPI.
    
    Args:
        search_query (str): The search query to use
        
    Returns:
        str: Formatted search results text or error message
    """
    # Use SerpAPI for real web search results
    serpapi_key = os.environ.get("SERPAPI_KEY")
    if not serpapi_key:
        logger.error("SERPAPI_KEY not set in environment variables")
        return f"Unable to perform web search due to missing SERPAPI_KEY."
        
    try:
        logger.info(f"Querying SerpAPI with: {search_query}")
        response = requests.get(
            "https://serpapi.com/search",
            params={
                "q": search_query,
                "api_key": serpapi_key,
                "engine": "google"
            }
        )
        
        if response.status_code != 200:
            logger.error(f"SerpAPI request failed with status code: {response.status_code}")
            return f"Web search failed with status code: {response.status_code}"
            
        search_data = response.json()
        
        # Extract and format results
        results_text = ""
        
        # Add knowledge graph if available
        if "knowledge_graph" in search_data:
            kg = search_data["knowledge_graph"]
            results_text += f"Knowledge Graph: {kg.get('title', '')}\n"
            if "description" in kg:
                results_text += f"Description: {kg['description']}\n"
            # Add other relevant KG fields concisely
            kg_details = []
            for key, value in kg.items():
                 if key not in ["title", "description", "header_images", "url", "source"] and isinstance(value, (str, int, float)):
                     kg_details.append(f"{key.replace('_', ' ').title()}: {value}")
            if kg_details:
                 results_text += "; ".join(kg_details) + "\n"
            results_text += "\n"
        
        # Add answer box if available
        if "answer_box" in search_data:
            box = search_data["answer_box"]
            results_text += f"Featured Snippet:\n"
            if "title" in box:
                results_text += f"Title: {box['title']}\n"
            if "answer" in box:
                results_text += f"Answer: {box['answer']}\n"
            elif "snippet" in box:
                results_text += f"Snippet: {box['snippet']}\n"
            results_text += "\n"
            
        # Add organic results (limit to top 3 for conciseness)
        if "organic_results" in search_data:
            results_text += "Top Search Results:\n"
            for i, result in enumerate(search_data["organic_results"][:3], 1):
                results_text += f"{i}. {result.get('title', 'No Title')}\n"
                if "snippet" in result:
                    # Keep snippets brief
                    snippet = result['snippet']
                    results_text += f"   Snippet: {snippet[:150]}{'...' if len(snippet) > 150 else ''}\n"
                if "link" in result:
                    results_text += f"   URL: {result['link']}\n"
                results_text += "\n"
                
        if not results_text:
             logger.warning("SerpAPI returned no usable results.")
             return "Web search did not return any relevant results."
             
        logger.info(f"SerpAPI search successful, formatted results generated.")
        return results_text.strip()
            
    except Exception as e:
        logger.error(f"SerpAPI search failed: {str(e)}")
        return f"Web search failed during execution: {str(e)}"

def verify_supabase_table():
    """Verify that the conversation_histories table exists and has the correct structure
    Note: Supabase Python client is not async"""
    if not supabase:
        logger.error("Cannot verify table: Supabase client is not initialized")
        return False
        
    try:
        # Check if table exists - Supabase client is not async
        result = supabase.table("conversation_histories").select("*").limit(1).execute()
        logger.info("Successfully connected to conversation_histories table")
        
        # Log table structure
        if result.data:
            logger.info("Table structure:")
            for key in result.data[0].keys():
                logger.info(f"- {key}")
        else:
            logger.info("Table exists but is empty")
                
        return True
    except Exception as e:
        logger.error(f"Failed to verify conversation_histories table: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {repr(e)}")
        return False

async def init_supabase():
    """Initialize the Supabase client with retry logic - Supabase client is not async"""
    global supabase
    
    # If already initialized, return True
    if supabase is not None:
        try:
            # Test connection with a simple query - Note: Supabase Python client is not async
            response = supabase.table("conversation_histories").select("session_id").limit(1).execute()
            logger.debug("Supabase connection is healthy")
            return True
        except Exception as e:
            # Connection might be stale, attempt to recreate
            logger.warning(f"Supabase connection test failed, will attempt to reconnect: {e}")
            supabase = None
    
    # Initialize or reinitialize Supabase client
    try:
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        
        if not supabase_url or not supabase_key:
            logger.error("Supabase environment variables not set")
            logger.error("Required: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
            return False
        
        # Create client with standard pattern - Supabase Python client is not async    
        supabase = create_client(supabase_url, supabase_key)
        
        # Test the connection immediately to verify it works
        try:
            test_response = supabase.table("conversation_histories").select("session_id").limit(1).execute()
            logger.info("Supabase client initialized and connected successfully")
            return True
        except Exception as test_error:
            logger.error(f"Failed to test Supabase connection: {test_error}")
            supabase = None
            return False
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        supabase = None
        return False

# Function to check Supabase connection health
async def check_supabase_health():
    """Check if Supabase connection is healthy and reconnect if needed"""
    global supabase
    
    if not supabase:
        logger.warning("Supabase client not initialized, attempting to initialize")
        return await init_supabase()
        
    try:
        # Test the connection with a simple query - Supabase Python client is not async
        test_result = supabase.table("conversation_histories").select("session_id").limit(1).execute()
        logger.debug("Supabase connection is healthy")
        return True
    except Exception as e:
        logger.warning(f"Supabase connection error: {e}, attempting to reconnect")
        return await init_supabase()

# Optimize store_conversation_message to be less resource-intensive
async def store_conversation_message(session_id, participant_id, conversation):
    """Store a conversation message in Supabase, with local backup on failure"""
    
    # Skip non-essential database operations if agent is actively speaking
    is_speaking = agent and hasattr(agent, 'is_speaking') and agent.is_speaking
    
    # For system messages during speech, defer to retry queue instead of immediate storage
    if is_speaking and participant_id == "system":
        add_to_retry_queue(session_id, participant_id, conversation)
        return False
    
    # Continue with existing storage logic
    # Check Supabase connection first
    supabase_available = await check_supabase_health()
    
    if not supabase_available:
        logger.error("Supabase client not available, adding message to retry queue")
        add_to_retry_queue(session_id, participant_id, conversation)
        return False
    
    # Ensure we're using a serializable format for conversation
    try:
        # Convert the conversation to a proper JSON object for the jsonb column
        serialized_conversation = json.dumps(conversation) if isinstance(conversation, dict) else conversation
        
        # Extract transcript for raw_conversation field
        content = conversation.get("content", "")
        
        # Get participant email if available, or use default
        user_email = ""
        if isinstance(conversation, dict) and "metadata" in conversation:
            metadata = conversation.get("metadata", {})
            if isinstance(metadata, dict) and "user_email" in metadata:
                user_email = metadata.get("user_email", "")
        
        # Get the timestamp
        timestamp = conversation.get("timestamp", get_current_timestamp())
        
        # Calculate message count (here just set to 1 for individual messages)
        message_count = 1
        
    except TypeError as e:
        logger.error(f"Failed to serialize message data: {e}")
        # Create a simplified version without problematic fields
        safe_message = {
            "role": conversation.get("role", "unknown"),
            "content": conversation.get("content", ""),
            "timestamp": conversation.get("timestamp", get_current_timestamp()),
            "session_id": session_id
        }
        serialized_conversation = json.dumps(safe_message)
        content = safe_message.get("content", "")
        timestamp = safe_message.get("timestamp", get_current_timestamp())
        message_count = 1
        user_email = ""
    
    # Prepare insert data with exact column names from Supabase table
    insert_data = {
        "session_id": session_id,               # text
        "participant_id": participant_id,       # text
        "conversation": serialized_conversation, # jsonb
        "raw_conversation": content,            # text
        "message_count": message_count,         # integer
        "user_email": user_email,               # text
        "timestamp": timestamp,                 # text
        "email_sent": False                     # boolean
    }
    
    # Use conditional logging to reduce verbosity during conversation
    verbose_log(f"Storing message for session {session_id}, participant {participant_id}")
    
    try:
        # Use upsert with the documented format from Supabase docs - NOT async
        response = (
            supabase.table("conversation_histories")
            .upsert([insert_data], on_conflict="session_id,timestamp")  # Use appropriate conflict resolution
            .execute()
        )
        
        # Check response structure following Supabase pattern
        if hasattr(response, 'data') and response.data:
            logger.info(f"Successfully stored message with ID: {session_id}")
            return True
        elif hasattr(response, 'error') and response.error:
            logger.error(f"Supabase upsert error: {response.error}")
            add_to_retry_queue(session_id, participant_id, conversation)
            return False
        else:
            # Handle unexpected response structure
            logger.warning(f"Unexpected response from Supabase: {response}")
            add_to_retry_queue(session_id, participant_id, conversation)
            return False
            
    except Exception as e:
        error_type = type(e).__name__
        error_message = str(e)
        
        # Log different error types differently
        if "duplicate key" in error_message.lower():
            verbose_log(f"Message already exists in database (duplicate key): {error_message}", level="warning")
            return True  # Consider this a success as the message is already stored
        elif "permission" in error_message.lower():
            logger.error(f"Supabase permission error: {error_message}")
        elif "network" in error_message.lower() or "timeout" in error_message.lower():
            logger.error(f"Supabase network error: {error_message}")
        else:
            logger.error(f"Failed to store message: {error_type}: {error_message}")
            # Only log full trace in verbose mode
            verbose_log(f"Error details: {traceback.format_exc()}", level="error")
        
        # Add to retry queue for later processing
        add_to_retry_queue(session_id, participant_id, conversation)
        return False

async def store_full_conversation():
    """Batch store all messages from the current conversation"""
    global conversation_history, session_id
    
    # Don't proceed if no messages or no session ID
    if not conversation_history or not session_id:
        logger.warning("No conversation history or session ID available for batch storage")
        return False
    
    # Check Supabase connection
    supabase_available = await check_supabase_health()
    if not supabase_available:
        logger.error("Supabase client not available, skipping batch conversation storage")
        return False
    
    messages_to_store = []
    user_email = ""
    
    # Process each message
    for message in conversation_history:
        try:
            if not isinstance(message, dict):
                logger.warning(f"Skipping non-dict message in batch storage: {type(message)}")
                continue
            
            # Get participant ID from message role
            participant_id = message.get("participant_id", message.get("role", "unknown"))
            
            # Extract user email if present
            if "metadata" in message and isinstance(message["metadata"], dict):
                if "user_email" in message["metadata"]:
                    user_email = message["metadata"]["user_email"]
            
            # Get session ID, defaulting to current session
            session_id = message.get("session_id", session_id)
            
            # Get message content and timestamp
            content = message.get("content", "")
            timestamp = message.get("timestamp", get_current_timestamp())
            
            # Serialize the message for storage
            serialized_message = json.dumps(message)
            
            # Create insert data with exact column names from Supabase table
            insert_data = {
                "session_id": session_id,
                "participant_id": participant_id,
                "conversation": serialized_message,
                "raw_conversation": content,
                "message_count": 1,
                "user_email": user_email,
                "timestamp": timestamp,
                "email_sent": False
            }
            
            messages_to_store.append(insert_data)
        except TypeError as e:
            logger.error(f"Failed to serialize message for batch storage: {e}")
            # Create a safe version without problematic fields
            try:
                safe_message = {
                    "role": message.get("role", "unknown"),
                    "content": message.get("content", ""),
                    "timestamp": message.get("timestamp", get_current_timestamp()),
                    "session_id": message.get("session_id")
                }
                
                serialized_safe_message = json.dumps(safe_message)
                content = safe_message.get("content", "")
                timestamp = safe_message.get("timestamp", get_current_timestamp())
                participant_id = message.get("participant_id", message.get("role", "unknown"))
                
                insert_data = {
                    "session_id": session_id,
                    "participant_id": participant_id,
                    "conversation": serialized_safe_message,
                    "raw_conversation": content,
                    "message_count": 1,
                    "user_email": "",
                    "timestamp": timestamp,
                    "email_sent": False
                }
                
                messages_to_store.append(insert_data)
            except Exception as safe_err:
                logger.error(f"Could not create safe version of message: {safe_err}")
    
    # Don't proceed if no valid messages to store
    if not messages_to_store:
        logger.warning("No valid messages to store in batch operation")
        return False
    
    try:
        # Use batch upsert with appropriate on_conflict resolution
        response = (
            supabase.table("conversation_histories")
            .upsert(messages_to_store, on_conflict="session_id,timestamp")
            .execute()
        )
        
        # Check response structure
        if hasattr(response, 'data') and response.data:
            logger.info(f"Successfully stored {len(messages_to_store)} messages in batch for session {session_id}")
            return True
        elif hasattr(response, 'error') and response.error:
            logger.error(f"Supabase batch upsert error: {response.error}")
            return False
        else:
            # Handle unexpected response structure
            logger.warning(f"Unexpected response from Supabase batch operation: {response}")
            return False
    except Exception as e:
        error_type = type(e).__name__
        error_message = str(e)
        
        # Log the error
        logger.error(f"Failed to store batch messages: {error_type}: {error_message}")
        verbose_log(f"Error details: {traceback.format_exc()}", level="error")
        return False

# Add a helper method to directly await the storage task when critical
async def ensure_storage_completed():
    """Ensure that conversation storage is completed before proceeding."""
    try:
        await store_full_conversation()
        logger.info("Storage operation completed")
        return True
    except Exception as e:
        logger.error(f"Error during ensured storage: {str(e)}")
        return False

def load_env_files():
    # Get the paths to both .env files
    agent_env_path = pathlib.Path(__file__).parent / '.env'
    web_env_path = pathlib.Path(__file__).parent.parent / 'web' / '.env.local'

    logger.info("Agent .env path: %s", agent_env_path.absolute())
    logger.info("Web .env.local path: %s", web_env_path.absolute())
    logger.info("Agent .env exists: %s", agent_env_path.exists())
    logger.info("Web .env.local exists: %s", web_env_path.exists())

    # Try to load from web .env.local first
    if web_env_path.exists():
        load_dotenv(dotenv_path=web_env_path, override=True)
        logger.info("Loaded environment from web .env.local")

    # Then load from agent .env (will override if exists)
    if agent_env_path.exists():
        load_dotenv(dotenv_path=agent_env_path, override=True)
        logger.info("Loaded environment from agent .env")

    # Debug log all environment variables
    logger.info("Environment variables after loading:")
    logger.info("LIVEKIT_URL: %s", os.environ.get('LIVEKIT_URL', 'NOT SET'))
    logger.info("LIVEKIT_API_KEY: %s", os.environ.get('LIVEKIT_API_KEY', 'NOT SET'))
    logger.info("OPENAI_API_KEY exists: %s", os.environ.get('OPENAI_API_KEY') is not None)
    logger.info("ELEVENLABS_API_KEY exists: %s", os.environ.get('ELEVENLABS_API_KEY') is not None)
    logger.info("GEMINI_API_KEY exists: %s", os.environ.get('GEMINI_API_KEY') is not None)
    logger.info("GOOGLE_API_KEY exists: %s", os.environ.get('GOOGLE_API_KEY') is not None)
    logger.info("GOOGLE_APPLICATION_CREDENTIALS exists: %s", os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') is not None)
    
    if os.environ.get('OPENAI_API_KEY'):
        api_key = os.environ.get('OPENAI_API_KEY', '')
        logger.info("OPENAI_API_KEY starts with: %s", api_key[:15] if api_key else 'EMPTY')
    
    if os.environ.get('ELEVENLABS_API_KEY'):
        api_key = os.environ.get('ELEVENLABS_API_KEY', '')
        logger.info("ELEVENLABS_API_KEY starts with: %s", api_key[:15] if api_key else 'EMPTY')
        
    if os.environ.get('GEMINI_API_KEY'):
        api_key = os.environ.get('GEMINI_API_KEY', '')
        logger.info("GEMINI_API_KEY starts with: %s", api_key[:15] if api_key else 'EMPTY')
        
        # Initialize Gemini API if key is available
        try:
            genai.configure(api_key=api_key)
            global gemini_initialized
            gemini_initialized = True
            logger.info("Gemini API initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {str(e)}")

    # Try reading both .env files directly
    try:
        if agent_env_path.exists():
            with open(agent_env_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    logger.info("Agent .env first line: %s", lines[0].strip())
    except Exception as e:
        logger.error("Error reading agent .env file: %s", str(e))

    try:
        if web_env_path.exists():
            with open(web_env_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    logger.info("Web .env.local first line: %s", lines[0].strip())
    except Exception as e:
        logger.error("Error reading web .env.local file: %s", str(e))

# Function to get user's IP location data
def get_ip_location(ip_address: str) -> Dict[str, Any]:
    """
    Get location information from an IP address using a free IP geolocation API.
    Tries multiple APIs in case one fails.
    
    Args:
        ip_address: The IP address to look up
        
    Returns:
        A dictionary with location information or empty dict if not found
    """
    if not ip_address:
        logger.warning("No IP address provided for geolocation")
        return {}
        
    try:
        # Skip private IP addresses
        try:
            if ipaddress.ip_address(ip_address).is_private:
                logger.info(f"Skipping geolocation for private IP: {ip_address}")
                return {}
        except ValueError:
            logger.warning(f"Invalid IP address format: {ip_address}")
            return {}
            
        # Try ip-api.com first (free, no API key required)
        try:
            logger.info(f"Attempting geolocation lookup for IP: {ip_address} via ip-api.com")
            url = f"http://ip-api.com/json/{ip_address}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    logger.info(f"Successfully retrieved geolocation data for {ip_address}")
                    location_data = {
                        "country": data.get("country"),
                        "region": data.get("regionName"),
                        "city": data.get("city"),
                        "timezone": data.get("timezone"),
                        "lat": data.get("lat"),
                        "lon": data.get("lon"),
                        "isp": data.get("isp"),
                        "source": "ip-api.com"
                    }
                    logger.info(f"Location data: {json.dumps(location_data)}")
                    return location_data
                else:
                    logger.warning(f"ip-api.com returned non-success status for {ip_address}: {data.get('status', 'unknown')}")
        except Exception as e:
            logger.warning(f"Error with ip-api.com: {str(e)}, trying fallback API")
        
        # Fallback to ipinfo.io (has free tier with rate limits)
        try:
            logger.info(f"Attempting fallback geolocation lookup for IP: {ip_address} via ipinfo.io")
            ipinfo_token = os.environ.get("IPINFO_TOKEN", "")
            url = f"https://ipinfo.io/{ip_address}/json"
            if ipinfo_token:
                url += f"?token={ipinfo_token}"
                
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if "bogon" not in data and "error" not in data:
                    # Parse location from format like "City, Region, Country"
                    loc_parts = (data.get("loc", "").split(",") if data.get("loc") else [])
                    lat = loc_parts[0] if len(loc_parts) > 0 else None
                    lon = loc_parts[1] if len(loc_parts) > 1 else None
                    
                    location_data = {
                        "country": data.get("country"),
                        "region": data.get("region"),
                        "city": data.get("city"),
                        "timezone": data.get("timezone"),
                        "lat": lat,
                        "lon": lon,
                        "isp": data.get("org"),
                        "source": "ipinfo.io"
                    }
                    logger.info(f"Successfully retrieved fallback location data for {ip_address}")
                    logger.info(f"Location data: {json.dumps(location_data)}")
                    return location_data
                else:
                    logger.warning(f"ipinfo.io indicates invalid IP: {ip_address}")
        except Exception as e:
            logger.warning(f"Error with ipinfo.io: {str(e)}")
        
        # If all methods fail, try to get a default or estimated location
        # Use environment variable if available
        default_country = os.environ.get("DEFAULT_COUNTRY", "")
        default_city = os.environ.get("DEFAULT_CITY", "")
        default_timezone = os.environ.get("DEFAULT_TIMEZONE", "")
        
        if default_country or default_city or default_timezone:
            logger.info(f"Using default location from environment variables")
            return {
                "country": default_country,
                "city": default_city,
                "timezone": default_timezone,
                "source": "default_environment"
            }
        
        logger.warning(f"Failed to get location for IP {ip_address} using all methods")
        return {}
    except Exception as e:
        logger.error(f"Error getting IP location for {ip_address}: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Full error details: {repr(e)}")
        return {}

# Function to get local time based on timezone
def get_local_time(timezone: str) -> Dict[str, Any]:
    """
    Get local time details for a given timezone
    
    Args:
        timezone: The timezone string (e.g., 'America/New_York', 'UTC+2', etc.)
    
    Returns:
        Dictionary with local time information
    """
    if not timezone:
        logger.warning("No timezone provided for local time determination")
        return {
            "local_time": get_utc_now().strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": "UTC",
            "time_of_day": "unknown",
            "is_business_hours": False,
            "day_of_week": get_utc_now().strftime("%A"),
            "source": "fallback_utc"
        }
        
    try:
        # Try to determine offset from various timezone formats
        utc_now = get_utc_now()
        offset_hours = 0
        
        # Check different timezone format patterns
        if timezone.startswith("UTC+") or timezone.startswith("GMT+"):
            try:
                offset_str = timezone.split("+")[1]
                # Handle hours:minutes format
                if ":" in offset_str:
                    hours, minutes = offset_str.split(":")
                    offset_hours = int(hours) + (int(minutes) / 60)
                else:
                    offset_hours = float(offset_str)
            except (IndexError, ValueError) as e:
                logger.warning(f"Error parsing positive UTC offset from {timezone}: {str(e)}")
                
        elif timezone.startswith("UTC-") or timezone.startswith("GMT-"):
            try:
                offset_str = timezone.split("-")[1]
                # Handle hours:minutes format
                if ":" in offset_str:
                    hours, minutes = offset_str.split(":")
                    offset_hours = -1 * (int(hours) + (int(minutes) / 60))
                else:
                    offset_hours = -1 * float(offset_str)
            except (IndexError, ValueError) as e:
                logger.warning(f"Error parsing negative UTC offset from {timezone}: {str(e)}")
                
        # Named timezone handling would be better with pytz, but we can do a simple map
        # for common timezones if pytz is not available
        timezone_map = {
            "America/New_York": -5,  # EST
            "America/Los_Angeles": -8,  # PST
            "America/Chicago": -6,  # CST
            "America/Denver": -7,  # MST
            "Europe/London": 0,  # GMT
            "Europe/Paris": 1,  # CET
            "Europe/Berlin": 1,  # CET
            "Europe/Moscow": 3,
            "Asia/Tokyo": 9,
            "Asia/Shanghai": 8,
            "Asia/Dubai": 4,
            "Asia/Singapore": 8,
            "Australia/Sydney": 10,
            "Pacific/Auckland": 12
        }
        
        # Check if we have this timezone in our map
        if timezone in timezone_map:
            offset_hours = timezone_map[timezone]
            logger.info(f"Found timezone {timezone} in map with offset {offset_hours}")
            
        # Calculate hours accounting for daylight saving time (very simple approach)
        # In a real implementation, use a proper timezone library like pytz or dateutil
        month = utc_now.month
        is_northern_summer = 3 <= month <= 10  # Very rough DST approximation
        
        # Adjust for daylight saving time
        if is_northern_summer and timezone in [
            "America/New_York", "America/Chicago", "America/Denver", 
            "America/Los_Angeles", "Europe/London", "Europe/Paris", "Europe/Berlin"
        ]:
            offset_hours += 1
            logger.info(f"Adjusted for daylight saving time, new offset: {offset_hours}")
        
        # Calculate local time based on offset
        # Add the offset hours
        hour_delta = int(offset_hours)
        minute_delta = int((offset_hours - hour_delta) * 60)
        
        # Add hours and minutes to create the local time
        local_time = utc_now
        local_time = local_time.replace(hour=(utc_now.hour + hour_delta) % 24)
        if minute_delta != 0:
            local_time = local_time.replace(minute=(utc_now.minute + minute_delta) % 60)
            if utc_now.minute + minute_delta >= 60:
                local_time = local_time.replace(hour=(local_time.hour + 1) % 24)
        
        # Handle day rollover for negative or positive offsets
        day_offset = 0
        if utc_now.hour + hour_delta < 0:
            day_offset = -1
        elif utc_now.hour + hour_delta >= 24:
            day_offset = 1
            
        # Apply day offset if needed
        if day_offset != 0:
            # This is a simplified approach without proper day/month boundary handling
            # In production, use a proper datetime library
            local_date = utc_now.date()
            local_time = local_time.replace(day=(local_date.day + day_offset))
        
        # Simple time categories for contextual understanding
        hour = local_time.hour
        if 5 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
        elif 17 <= hour < 22:
            time_of_day = "evening"
        else:
            time_of_day = "night"
        
        # Create the full local time context
        result = {
            "local_time": local_time.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": timezone,
            "timezone_offset": f"UTC{'+' if offset_hours >= 0 else ''}{offset_hours}",
            "time_of_day": time_of_day,
            "is_business_hours": 9 <= hour < 17 and local_time.weekday() < 5,
            "day_of_week": local_time.strftime("%A"),
            "date": local_time.strftime("%Y-%m-%d"),
            "approximate_dst": is_northern_summer,
            "source": "calculated"
        }
        
        logger.info(f"Calculated local time for timezone {timezone}: {result['local_time']}")
        return result
        
    except Exception as e:
        logger.error(f"Error calculating local time for timezone {timezone}: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Full error details: {repr(e)}")
        
        # Return UTC time as fallback
        utc_now = get_utc_now()
        return {
            "local_time": utc_now.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": "UTC",
            "time_of_day": "unknown",
            "is_business_hours": False,
            "day_of_week": utc_now.strftime("%A"),
            "date": utc_now.strftime("%Y-%m-%d"),
            "source": "fallback_error"
        }

# Extract client IP address from participant
def extract_client_ip(participant: rtc.Participant) -> str:
    """
    Try to extract the client IP address from participant information.
    This is a best-effort approach that attempts multiple methods to find the IP.
    
    Args:
        participant: The LiveKit participant
        
    Returns:
        The IP address as a string, or empty string if not found
    """
    try:
        # First, check if the IP is in metadata as a direct field
        if participant.metadata:
            try:
                metadata = json.loads(participant.metadata)
                # Direct ip_address field
                if metadata.get("ip_address"):
                    logger.info(f"Found IP address in metadata.ip_address: {metadata.get('ip_address')}")
                    return metadata.get("ip_address")
                
                # Check headers if provided
                if metadata.get("headers"):
                    headers = metadata.get("headers", {})
                    # Common headers that might contain the real IP
                    ip_headers = [
                        "X-Forwarded-For", 
                        "X-Real-IP", 
                        "CF-Connecting-IP",  # Cloudflare
                        "True-Client-IP",    # Akamai/Cloudflare
                        "X-Client-IP"        # Amazon CloudFront
                    ]
                    
                    for header in ip_headers:
                        if headers.get(header):
                            # X-Forwarded-For can contain multiple IPs - take the first one
                            ip = headers.get(header).split(',')[0].strip()
                            logger.info(f"Found IP address in header {header}: {ip}")
                            return ip
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing participant metadata for IP: {str(e)}")
        
        # Try to get IP from connection info if LiveKit makes it available
        # This is implementation-specific and depends on what LiveKit makes available
        if hasattr(participant, "connection_info") and hasattr(participant.connection_info, "client_ip"):
            logger.info(f"Found IP in connection_info: {participant.connection_info.client_ip}")
            return participant.connection_info.client_ip
        
        # If we couldn't find an IP, log this for debugging
        logger.warning("Could not determine client IP address")
        logger.info(f"Participant identity: {participant.identity}")
        logger.info(f"Participant metadata type: {type(participant.metadata)}")
        if participant.metadata:
            logger.info(f"Metadata content: {participant.metadata[:100]}...")
        
        # Use a placeholder default if needed for development
        default_ip = os.environ.get("DEFAULT_TEST_IP", "")
        if default_ip:
            logger.info(f"Using default test IP address: {default_ip}")
            return default_ip
            
        return ""
    except Exception as e:
        logger.error(f"Error extracting client IP: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Full error details: {repr(e)}")
        return ""

# Load environment variables at startup
load_env_files()

@dataclass
class SessionConfig:
    instructions: str
    voice: str  # ElevenLabs voice ID
    temperature: float
    max_response_output_tokens: str | int
    modalities: list[str]
    turn_detection: Dict[str, Any]

    def __post_init__(self):
        if self.modalities is None:
            self.modalities = self._modalities_from_string("text_and_audio")
        # Set default voice to ElevenLabs voice ID if not specified
        if not self.voice:
            self.voice = OPENAI_VOICE_ID

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def _modalities_from_string(modalities: str) -> list[str]:
        modalities_map = {
            "text_and_audio": ["text", "audio"],
            "text_only": ["text"],
        }
        return modalities_map.get(modalities, ["text", "audio"])

    def __eq__(self, other: SessionConfig) -> bool:
        return self.to_dict() == other.to_dict()

def parse_session_config(data: Dict[str, Any]) -> SessionConfig:
    turn_detection = None

    if data.get("turn_detection"):
        turn_detection = json.loads(data.get("turn_detection"))
    else:
        turn_detection = {
            "threshold": 0.5,
            "prefix_padding_ms": 200,
            "silence_duration_ms": 300,
        }

    # Use default prompt if none provided
    config = SessionConfig(
        instructions=data.get("instructions", DEFAULT_INTERVIEW_PROMPT),
        voice=data.get("voice", OPENAI_VOICE_ID),
        temperature=float(data.get("temperature", 0.8)),
        max_response_output_tokens=data.get("max_output_tokens")
        if data.get("max_output_tokens") == "inf"
        else int(data.get("max_output_tokens") or 2048),
        modalities=SessionConfig._modalities_from_string(
            data.get("modalities", "text_and_audio")
        ),
        turn_detection=turn_detection,
    )
    return config

async def entrypoint(ctx: JobContext):
    load_env_files()

    try:
        if not await init_supabase():
            logger.error("Failed to initialize Supabase in worker process")
            raise Exception("Database connection failed")
        
        # Initialize OpenAI client for web search
        global openai_client
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OPENAI_API_KEY not set, web search functionality will be disabled")
        else:
            try:
                openai_client = OpenAI(api_key=openai_api_key)
                logger.info("OpenAI client initialized for web search")
            except Exception as openai_error:
                logger.error(f"Failed to initialize OpenAI client: {str(openai_error)}")
        
        logger.info(f"connecting to room {ctx.room.name}")
        try:
            await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
            logger.info("Successfully connected to room")
        except Exception as conn_error:
            logger.error(f"Failed to connect to room: {str(conn_error)}")
            raise

        logger.info("Waiting for participant...")
        try:
            participant = await ctx.wait_for_participant()
            logger.info(f"Participant joined: {participant.identity}")
            logger.info(f"Participant metadata: {participant.metadata}")
        except Exception as part_error:
            logger.error(f"Error waiting for participant: {str(part_error)}")
            raise
        
        try:
            await run_multimodal_agent(ctx, participant)
        except Exception as agent_error:
            logger.error(f"Error in multimodal agent: {str(agent_error)}")
            logger.error(f"Participant details - Identity: {participant.identity}, Name: {participant.name}")
            logger.error(f"Participant metadata type: {type(participant.metadata)}")
            logger.error(f"Raw metadata content: {repr(participant.metadata)}")
            raise

        logger.info("agent started successfully")
        
    except Exception as e:
        logger.error(f"Critical error in entrypoint: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Full error details: {repr(e)}")
        raise

async def run_multimodal_agent(room_name, participant_id=None):
    """Start the agent within a LiveKit room"""
    global agent, current_room_name, conversation_history, current_session_id, current_participant_id
    
    # Log both Room SID and Participant SID for better tracking
    logger.info(f"Starting agent for LiveKit Room: {room_name}")
    logger.info(f"Participant SID: {participant_id}")
    
    try:
        # Set global tracking variables for session management
        current_room_name = room_name
        current_participant_id = participant_id
        
        # Use LiveKit room name as session_id 
        current_session_id = room_name
        
        logger.info(f"Session ID set to LiveKit room name: {current_session_id}")
        conversation_history = []
        
        # Extract user email from participant if available
        user_email = ""
        try:
            # Get participant information to extract email
            if participant_id:
                participant = participant.get(participant_id)
                metadata = json.loads(participant.metadata) if isinstance(participant.metadata, str) else participant.metadata
                if isinstance(metadata, dict) and "user_email" in metadata:
                    user_email = metadata.get("user_email", "")
                    logger.info(f"Found user email in metadata: {user_email}")
        except Exception as email_err:
            logger.warning(f"Could not extract user email: {email_err}")
        
        # Store initial session metadata for transcript context
        initial_session_metadata = {
            "type": "session_metadata",
            "data": {
                "room_name": room_name,
                "participant_id": participant_id,
                "session_id": current_session_id,
                "start_time": get_current_timestamp(),
                "user_email": user_email
            }
        }
        
        # Reset agent state for a new conversation
        if agent is None:
            logger.info("Creating new agent instance")
            agent = Agent(room_name=room_name, participant_id=participant_id)
        else:
            logger.info("Resetting existing agent instance")
            await agent.reset(room_name=room_name, participant_id=participant_id)
            
        # Set the current session ID in the agent for consistent tracking
        agent.session_id = current_session_id
        
        # Include session metadata in conversation history
        metadata_message = {
            "role": "system",
            "content": json.dumps(initial_session_metadata),
            "timestamp": get_current_timestamp(),
            "session_id": current_session_id,
            "metadata": {
                "type": "session_start",
                "user_email": user_email
            }
        }
        conversation_history.append(metadata_message)
        
        # Store the initial session metadata
        await store_conversation_message(
            session_id=current_session_id,
            participant_id="system",
            conversation=metadata_message
        )
        
        # Asynchronously start the agent
        await agent.start()
        return True
        
    except Exception as e:
        logger.error(f"Failed to start agent: {e}")
        logger.error(traceback.format_exc())
        return False

# Initialize the OpenAI and Gemini APIs
def init_apis():
    global openai_client, gemini_initialized
    
    # Initialize OpenAI client
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        try:
            openai_client = OpenAI(api_key=openai_api_key)
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    else:
        logger.warning("OPENAI_API_KEY not set, some functionality may be limited")
    
    # Initialize Gemini API
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if gemini_api_key:
        try:
            genai.configure(api_key=gemini_api_key)
            gemini_initialized = True
            logger.info("Gemini API initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {str(e)}")
    else:
        logger.warning("GEMINI_API_KEY not set, web search functionality may be limited")

def sync_init_supabase():
    """Synchronous wrapper for async Supabase initialization"""
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Initialize directly in the synchronous context
        global supabase
        
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        
        if not supabase_url or not supabase_key:
            logger.error("Supabase environment variables not set")
            logger.error("Required: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
            return False
        
        # Create client with standard pattern - Supabase Python client is synchronous
        try:
            supabase = create_client(supabase_url, supabase_key)
            
            # Test the connection immediately to verify it works
            test_response = supabase.table("conversation_histories").select("session_id").limit(1).execute()
            logger.info("Supabase client initialized and connected successfully")
            return True
        except Exception as client_error:
            logger.error(f"Failed to initialize or test Supabase client: {client_error}")
            supabase = None
            return False
        
    except Exception as e:
        logger.error(f"Error during Supabase initialization: {str(e)}")
        return False
    finally:
        # Always close the event loop to avoid resource leaks
        try:
            loop.close()
        except Exception as e:
            logger.error(f"Error closing event loop: {e}")

# --- Pinecone Initialization and Querying --- #

def init_pinecone():
    """Initializes the Pinecone client and connects to the index."""
    global pinecone_client, pinecone_index
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_api_key:
        logger.error("PINECONE_API_KEY not set in environment variables. Knowledge base functionality disabled.")
        return False
    
    try:
        logger.info(f"Initializing Pinecone client...")
        pinecone_client = Pinecone(api_key=pinecone_api_key)
        
        # Correctly get the list of index names
        index_names = [index_info.name for index_info in pinecone_client.list_indexes()]
        logger.info(f"Available Pinecone indexes: {index_names}")

        # Check if index exists and connect
        if PINECONE_INDEX_NAME not in index_names:
            logger.error(f"Pinecone index '{PINECONE_INDEX_NAME}' does not exist. Knowledge base functionality disabled.")
            pinecone_client = None # Disable client if index missing
            return False
            
        logger.info(f"Connecting to Pinecone index: {PINECONE_INDEX_NAME}")
        pinecone_index = pinecone_client.Index(PINECONE_INDEX_NAME)
        logger.info(f"Successfully connected to Pinecone index '{PINECONE_INDEX_NAME}'.")
        # Optional: Log index stats
        try:
            stats = pinecone_index.describe_index_stats()
            logger.info(f"Pinecone index stats: {stats}")
        except Exception as stat_e:
            logger.warning(f"Could not retrieve Pinecone index stats: {stat_e}")
            
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {str(e)}")
        pinecone_client = None # Ensure client is None on failure
        pinecone_index = None
        return False

def get_embedding(text: str, model: str = EMBEDDING_MODEL):
    """Generates embeddings for the given text using OpenAI."""
    if not openai_client:
        logger.error("OpenAI client not initialized. Cannot generate embeddings.")
        return None
    try:
        text = text.replace("\n", " ")
        response = openai_client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Failed to get embedding from OpenAI: {str(e)}")
        return None

async def query_pinecone_knowledge_base(query: str, top_k: int = 3):
    """Queries the Pinecone knowledge base and returns relevant text chunks."""
    if not pinecone_index:
        logger.warning("Pinecone index not available, skipping knowledge base query.")
        return "Internal knowledge base is currently unavailable."
        
    try:
        logger.info(f"Generating embedding for knowledge base query: {query[:50]}...")
        query_embedding = get_embedding(query)
        
        if not query_embedding:
            return "Could not process query for the knowledge base."
            
        logger.info(f"Querying Pinecone index '{PINECONE_INDEX_NAME}'...")
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True # Assuming metadata contains the text
        )
        
        if not results or not results.matches:
            logger.info("No relevant documents found in knowledge base.")
            return "No specific information found in the knowledge base for that query."
            
        # Format results
        context_str = "Found relevant information in the knowledge base:\n"
        for i, match in enumerate(results.matches):
            score = match.score
            text_chunk = match.metadata.get('text', '[No text found in metadata]') # Adjust metadata field if needed
            source = match.metadata.get('source', 'Unknown source') # Example: get source if available
            context_str += f"\n{i+1}. (Score: {score:.2f}) From {source}:\n{text_chunk}\n"
            
        logger.info(f"Returning {len(results.matches)} results from knowledge base.")
        return context_str.strip()
        
    except Exception as e:
        logger.error(f"Error querying Pinecone knowledge base: {str(e)}")
        return f"An error occurred while accessing the knowledge base: {str(e)}"

# --- End Pinecone --- #

# Initialize local backup directory - DISABLED to avoid file serialization issues
def init_local_backup():
    """This function is disabled to prevent serialization errors"""
    logger.info("Local backup functionality is disabled")
    return True

# Save retry queue to disk - DISABLED to avoid file serialization issues
def save_retry_queue():
    """This function is disabled to prevent serialization errors"""
    # Logging disabled to avoid log spam
    return True

# Add a message to retry queue
def add_to_retry_queue(session_id, participant_id, conversation):
    """Add a failed message to the retry queue for later processing"""
    global retry_queue
    
    try:
        # Ensure we're using a serializable format
        if isinstance(conversation, dict):
            # Extract necessary fields
            content = conversation.get("content", "")
            timestamp = conversation.get("timestamp", get_current_timestamp())
            user_email = ""
            
            # Try to extract user email if present
            if "metadata" in conversation and isinstance(conversation["metadata"], dict):
                if "user_email" in conversation["metadata"]:
                    user_email = conversation["metadata"]["user_email"]
            
            # Create a simplified version for storage
            safe_message = {
                "role": conversation.get("role", "unknown"),
                "content": conversation.get("content", ""),
                "timestamp": conversation.get("timestamp", get_current_timestamp()),
                "session_id": conversation.get("session_id", session_id)
            }
            
            # Create a retry item matching the Supabase table structure
            retry_item = {
                "retry_timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "participant_id": participant_id,
                "raw_conversation": content,
                "conversation": json.dumps(safe_message),
                "message_count": 1,
                "user_email": user_email,
                "timestamp": timestamp,
                "email_sent": False
            }
            
            retry_queue.append(retry_item)
            logger.info(f"Added message to retry queue. Queue size: {len(retry_queue)}")
            return True
        else:
            logger.error(f"Cannot add non-dict message to retry queue: {type(conversation)}")
            return False
    except Exception as e:
        logger.error(f"Error adding to retry queue: {str(e)}")
        return False

# Add a dedicated retry processor with adjustable batch size
async def process_retry_queue(batch_size=10):
    """Process the retry queue to attempt to store failed messages"""
    global retry_queue
    
    if not retry_queue:
        verbose_log("Retry queue is empty", level="info")
        return
    
    # Check Supabase connection
    supabase_available = await check_supabase_health()
    if not supabase_available:
        logger.warning("Supabase unavailable for retry queue processing")
        return
    
    logger.info(f"Processing retry queue with {len(retry_queue)} items")
    
    # Take a batch of items to process
    batch = retry_queue[:batch_size]
    
    # Process items in batch
    success_items = []
    
    for item in batch:
        try:
            # Extract the data from the retry item
            session_id = item.get("session_id")
            participant_id = item.get("participant_id")
            conversation_data = item.get("conversation")
            
            # Skip items missing required fields
            if not session_id or not participant_id or not conversation_data:
                logger.warning(f"Skipping retry item with missing data: {item}")
                success_items.append(item)  # Skip invalid items
                continue
            
            # Prepare the insert data with exact column names from Supabase table
            insert_data = {
                "session_id": item.get("session_id"),
                "participant_id": item.get("participant_id"),
                "conversation": item.get("conversation"),
                "raw_conversation": item.get("raw_conversation", ""),
                "message_count": item.get("message_count", 1),
                "user_email": item.get("user_email", ""),
                "timestamp": item.get("timestamp", get_current_timestamp()),
                "email_sent": item.get("email_sent", False)
            }
            
            # Attempt to store the item
            response = (
                supabase.table("conversation_histories")
                .upsert([insert_data], on_conflict="session_id,timestamp")
                .execute()
            )
            
            # Check if successful
            if hasattr(response, 'data') and response.data:
                logger.info(f"Successfully processed retry item for session {session_id}")
                success_items.append(item)
            else:
                logger.warning(f"Failed to process retry item for session {session_id}")
        except Exception as e:
            logger.error(f"Error processing retry item: {e}")
    
    # Remove successful items from the queue
    for item in success_items:
        if item in retry_queue:
            retry_queue.remove(item)
    
    logger.info(f"Processed {len(success_items)}/{len(batch)} retry items. Queue size: {len(retry_queue)}")

# Enhanced periodic retry task with reduced frequency and optimized background processing
async def periodic_retry_processor():
    """Periodically process the retry queue"""
    try:
        # Initial delay to avoid startup contention
        await asyncio.sleep(30)
        
        while True:
            # Increased from 60 to 180 seconds to reduce frequency
            await asyncio.sleep(180)  # Process retry queue every 3 minutes
            
            # Skip if queue is empty (most common case)
            if not retry_queue:
                continue
                
            # Skip if agent is actively speaking to avoid interruptions
            if agent and (hasattr(agent, 'is_speaking') and agent.is_speaking or 
                         hasattr(agent, 'should_pause_background_tasks') and agent.should_pause_background_tasks()):
                logger.debug("Skipping retry processing during active speech")
                continue
                
            # Only process when supabase is connected and retry queue has items
            if await init_supabase():
                # Process a smaller batch size to reduce impact
                try:
                    await process_retry_queue(batch_size=5)
                except Exception as e:
                    logger.error(f"Error during periodic retry processing: {e}")
            else:
                logger.warning("Skipping retry processing due to Supabase connection issues")
            
    except asyncio.CancelledError:
        logger.info("Periodic retry processor task cancelled")
    except Exception as e:
        logger.error(f"Error in periodic retry processor: {e}")

# Custom exception for forced disconnection
class ForceDisconnectError(Exception):
    """Raised to force a disconnection in certain scenarios"""
    pass

# Define an Agent class for our implementation 
class Agent:
    """Class that wraps our voice and chat capabilities"""
    def __init__(self, room_name=None, participant_id=None):
        self.room_name = room_name
        self.participant_id = participant_id
        self.is_speaking = False
        self.session_id = room_name  # LiveKit room name as the session ID
        self.conversation_history = []
        self.callbacks = {}
        logger.info(f"Agent initialized with room: {room_name}, participant: {participant_id}")
    
    async def reset(self, room_name=None, participant_id=None):
        """Reset the agent for a new conversation"""
        self.room_name = room_name or self.room_name
        self.participant_id = participant_id or self.participant_id
        self.is_speaking = False
        self.session_id = room_name  # LiveKit room name as the session ID
        self.conversation_history = []
        logger.info(f"Agent reset with room: {self.room_name}, participant: {self.participant_id}")
        return True
    
    async def start(self):
        """Start the agent and send an initial welcome message"""
        logger.info(f"Starting agent for room: {self.room_name}")
        
        # Register the event handler for user speech
        self.on("user_speech_committed", self.on_user_speech_committed)
        
        # Send welcome message
        initial_message = "Hi, I am Prepzo. I can help you with any professional problem you're having."
        await self.say(initial_message)
        
        return True
    
    async def say(self, message_text, allow_interruptions=True):
        """Speak a message (would be linked to TTS in full implementation)"""
        self.is_speaking = True
        logger.info(f"Agent speaking: {message_text[:50]}...")
        
        # Create and store assistant message
        await self.say_and_store(message_text)
        
        # In a full implementation, this would send audio
        self.is_speaking = False
        return True
    
    async def prepare_say(self, message_text):
        """Prepare speech audio before speaking (for streaming implementations)"""
        logger.info(f"Preparing speech audio for: {message_text[:50]}...")
        # In a full implementation, this would prepare audio
        return True
    
    def should_pause_background_tasks(self):
        """Check if background tasks should be paused"""
        return self.is_speaking
    
    def on(self, event_name, callback):
        """Register a callback for an event"""
        self.callbacks[event_name] = callback
        return True
    
    async def on_user_speech_committed(self, transcript):
        """Handle user speech and store it using session_id"""
        global conversation_history
        
        # Skip minimal processing if speaking to avoid interruptions
        if self.is_speaking:
            logger.debug("Received user speech while agent is speaking, deferring processing")
            return
            
        logger.info(f"User speech: {transcript[:30]}...")
        
        # Extract user email if available
        user_email = ""
        try:
            # In a real implementation, this would extract the email from participant metadata
            pass
        except Exception as email_err:
            logger.warning(f"Could not extract user email: {email_err}")
        
        # Create complete user message
        user_chat_message = {
            "role": "user",
            "content": transcript,
            "timestamp": get_current_timestamp(),
            "session_id": self.session_id,  # Use LiveKit room name
            "metadata": {
                "type": "user_speech", 
                "user_email": user_email,
                "room_name": self.room_name,
                "participant_id": self.participant_id
            }
        }
        
        # Add to conversation history
        if isinstance(conversation_history, list):
            conversation_history.append(user_chat_message)
        
        # Save in agent's conversation history too
        self.conversation_history.append(user_chat_message)
        
        # Store in database
        asyncio.create_task(store_conversation_message(
            session_id=self.session_id,
            participant_id="user",
            conversation=user_chat_message
        ))
        
        # In a real implementation, this would process the user's request
        # and generate a response
        
    async def say_and_store(self, message_text):
        """Generate assistant message, store it, and speak it"""
        # Create a message with complete context
        assistant_message = {
            "role": "assistant",
            "content": message_text,
            "timestamp": get_current_timestamp(),
            "session_id": self.session_id,  # Use LiveKit room name
            "metadata": {
                "type": "agent_speech",
                "room_name": self.room_name,
                "participant_id": self.participant_id
            }
        }
        
        # Add to conversation histories
        global conversation_history
        if isinstance(conversation_history, list):
            conversation_history.append(assistant_message)
        
        # Save in agent's conversation history too
        self.conversation_history.append(assistant_message)
        
        # Store in database
        asyncio.create_task(store_conversation_message(
            session_id=self.session_id,
            participant_id="assistant",
            conversation=assistant_message
        ))

if __name__ == "__main__":
    # Create a synchronous wrapper that runs the async init in a new event loop
    # Run the initialization before starting the app
    if not sync_init_supabase():
        logger.error("Supabase initialization failed, exiting")
        import sys
        sys.exit(1)
    
    # Initialize APIs
    init_apis()
    
    # Initialize Pinecone
    if not init_pinecone():
        logger.warning("Pinecone initialization failed. Knowledge base functionality will be unavailable.")
    
    # Start health check HTTP server for deployment verification
    def start_health_check_server():
        import threading
        import http.server
        import socketserver
        import json
        from datetime import datetime
        import version
        
        PORT = int(os.environ.get("HEALTH_CHECK_PORT", 8080))
        
        class HealthCheckHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    
                    # Get version info
                    ver_info = version.get_version_info()
                    
                    # Prepare health check data
                    health_data = {
                        "status": "ok",
                        "service": "prepzo-bot",
                        "version": ver_info["version"],
                        "build_date": ver_info["build_date"],
                        "git_commit": ver_info["git_commit"],
                        "timestamp": datetime.now().isoformat(),
                        "uptime": time.time() - START_TIME,
                        "environment": os.environ.get("ENVIRONMENT", "production")
                    }
                    
                    # Send response
                    self.wfile.write(json.dumps(health_data).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
        
        httpd = socketserver.TCPServer(("", PORT), HealthCheckHandler)
        logger.info(f"Health check server started on port {PORT}")
        
        # Run server in a separate thread
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
    
    # Record start time for uptime calculation
    START_TIME = time.time()
    
    # Start health check server
    try:
        start_health_check_server()
        logger.info("Health check server started successfully")
    except Exception as e:
        logger.error(f"Failed to start health check server: {str(e)}")
        logger.error("Continuing without health check endpoint")
    
    # Standard worker options without async_init
    worker_options = WorkerOptions(
            entrypoint_fnc=entrypoint,
            worker_type=WorkerType.ROOM
        )
    
    # Start the app after successful initialization
    cli.run_app(worker_options)