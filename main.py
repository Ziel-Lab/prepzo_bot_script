from __future__ import annotations

import asyncio
import json
import logging
import uuid
import os
import pathlib
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Optional, Tuple
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
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.openai import tts as openai_tts
from livekit.agents.llm import ChatMessage, ChatImage
from supabase import create_client, Client
from dotenv import load_dotenv
# from livekit.agents.tts import TTSService
import prompt
from openai import OpenAI
import version

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)

# Initialize Supabase client
supabase: Client = None

# Initialize OpenAI client for web search
openai_client: OpenAI = None

# OpenAI TTS configuration
OPENAI_VOICE_ID = "alloy"  

# Default interview preparation prompt
DEFAULT_INTERVIEW_PROMPT = prompt.prompt

# Helper function to get current UTC time
def get_utc_now():
    """Get current UTC time in a timezone-aware manner"""
    try:
        # Python 3.11+ approach 
        return datetime.now(datetime.UTC)
    except AttributeError:
        # Fallback for earlier Python versions
        return datetime.now(timezone.utc)

async def perform_web_search(query: str, search_context_size: str = "medium", location_context: Dict[str, Any] = None):
    """
    Perform a web search using OpenAI's web search capability.
    
    Args:
        query (str): The search query
        search_context_size (str): Size of the search context ('low', 'medium', 'high')
        location_context (Dict): Optional location data to refine search
        
    Returns:
        str: The search results text
    """
    global openai_client
    
    if not openai_client:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not set in environment variables")
            return "Unable to perform web search due to missing API key."
        openai_client = OpenAI(api_key=api_key)
    
    try:
        logger.info(f"Performing web search for: {query}")
        
        # Create a more focused query to get better search results
        # Analyze the query to determine if it's specifically about jobs, interviews, or career topics
        job_related_terms = [
            "job", "career", "interview", "resume", "CV", "hiring", "employment", 
            "recruit", "salary", "profession", "workplace", "work", "company", 
            "industry", "position", "role", "skills", "qualification", "experience"
        ]
        
        # Check if query is already job-related
        is_job_related = any(term in query.lower() for term in job_related_terms)
        
        # Formulate the enhanced query
        if is_job_related:
            # If already job-related, use as is but add "latest" or "current" to get recent information
            enhanced_query = f"{query} (latest information 2025)"
        else:
            # If not obviously job-related, add career context to make it relevant
            enhanced_query = f"{query} (in context of professional careers, jobs, and workplace)"
        
        # Add location context if available
        if location_context and location_context.get("location"):
            loc = location_context["location"]
            if loc.get("country") and loc.get("city"):
                # Add specific location to the query for more locally relevant results
                location_str = f" in {loc.get('city')}, {loc.get('country')}"
                enhanced_query += location_str
                logger.info(f"Added location context '{location_str}' to query")
        
        logger.info(f"Enhanced search query: {enhanced_query}")
        
        # Prepare search tools config
        search_tools = {
            "type": "web_search_preview",
            "search_context_size": "high",
        }
        
        # Add user location to web search if available
        if location_context and location_context.get("location"):
            loc = location_context["location"]
            if loc.get("country") and loc.get("city") and loc.get("region"):
                search_tools["user_location"] = {
                    "type": "approximate",
                    "country": loc.get("country", ""),
                    "city": loc.get("city", ""),
                    "region": loc.get("region", "")
                }
                logger.info("Added user location data to web search request")
        
        # Use high search context size for more comprehensive results
        response = openai_client.responses.create(
            model="gpt-4o",
            tools=[search_tools],
            input=enhanced_query,
        )
        
        logger.info("Web search completed successfully")
        return response.output_text
    except Exception as e:
        logger.error(f"Web search failed: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Full error details: {repr(e)}")
        return f"Sorry, I encountered an issue while searching the web: {str(e)}"

def verify_supabase_table():
    """Verify that the conversation_histories table exists and has the correct structure"""
    if not supabase:
        logger.error("Cannot verify table: Supabase client is not initialized")
        return False
        
    try:
        # Check if table exists
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
    global supabase
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_role_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    
    if not supabase_url:
        logger.error("SUPABASE_URL is not set in environment variables")
        return False
    if not supabase_role_key:
        logger.error("SUPABASE_SERVICE_ROLE_KEY is not set in environment variables")
        return False
        
    try:
        # Log the URL (with sensitive parts redacted)
        masked_url = supabase_url.replace(supabase_role_key, '[REDACTED]')
        logger.info(f"Initializing Supabase client with URL: {masked_url}")
        
        supabase = create_client(supabase_url, supabase_role_key)
        logger.info("Supabase client initialized successfully")
        
        # Verify table structure
        if not verify_supabase_table():
            logger.error("Failed to verify conversation_histories table structure")
            return False
        
        # Test query to ensure we can access the table
        test_result = supabase.table("conversation_histories").select("*").limit(1).execute()
        logger.info("Successfully tested query on conversation_sessions table")
            
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {str(e)}")
        logger.error(f"Please check your SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables")
        return False

async def store_conversation_message(
    session_id: str,
    role: str,
    content: str,
    participant_id: str,
    metadata: Dict[str, Any] = None,
    location_data: Dict[str, Any] = None,
    local_time_data: Dict[str, Any] = None
):
    """Store a single conversation message in Supabase"""
    if not supabase:
        logger.error("Cannot store message: Supabase client is not initialized")
        return
        
    try:
        data = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "participant_id": participant_id,
            "metadata": json.dumps(metadata) if metadata else None,
            "timestamp": get_utc_now().isoformat(),
            "user_location": json.dumps(location_data) if location_data else None,
            "user_local_time": json.dumps(local_time_data) if local_time_data else None
        }
        
        logger.info(f"Attempting to store message in conversation_sessions:")
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Role: {role}")
        logger.info(f"Participant ID: {participant_id}")
        logger.info(f"Content length: {len(content)}")
        logger.info(f"Metadata: {metadata}")
        if location_data:
            location_str = ", ".join(f"{k}: {v}" for k, v in location_data.items() if v)
            logger.info(f"Including location data: {location_str}")
        
        result = supabase.table("conversation_histories").insert(data).execute()
        
        if result.data:
            logger.info(f"Successfully stored message. Response data: {json.dumps(result.data, indent=2)}")
            return result
        else:
            logger.warning("Message stored but no data returned from Supabase")
            logger.warning(f"Full result object: {result}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to store message in Supabase: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {repr(e)}")
        logger.error(f"Data being stored: {json.dumps(data, indent=2)}")
        return None

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
    
    if os.environ.get('OPENAI_API_KEY'):
        api_key = os.environ.get('OPENAI_API_KEY', '')
        logger.info("OPENAI_API_KEY starts with: %s", api_key[:15] if api_key else 'EMPTY')
    
    if os.environ.get('ELEVENLABS_API_KEY'):
        api_key = os.environ.get('ELEVENLABS_API_KEY', '')
        logger.info("ELEVENLABS_API_KEY starts with: %s", api_key[:15] if api_key else 'EMPTY')

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
        
        # Try to get IP from connection info if LiveKit provides it
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

async def run_multimodal_agent(ctx: JobContext, participant: rtc.Participant):
    try:
        logger.info(f"Participant metadata raw: '{participant.metadata}'")
        logger.info(f"Participant identity: {participant.identity}")
        logger.info(f"Participant name: {participant.name}")
        
        # Immediately create session ID
        session_id = ctx.room.name
        logger.info(f"Starting new session with ID: {session_id}")
        
        # Parse metadata safely
        try:
            metadata = json.loads(participant.metadata) if participant.metadata else {}
            logger.info(f"Parsed metadata: {metadata}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse metadata: {str(e)}, using empty dict")
            metadata = {}
        
        # Immediately extract location and time data, before doing anything else
        logger.info("Extracting user location and time context...")
        user_context = {
            "location": {},
            "local_time": {}
        }
        
        # First try to get IP address - this is our primary source of location data
        client_ip = extract_client_ip(participant)
        if client_ip:
            logger.info(f"Successfully extracted client IP: {client_ip}")
            # Get geolocation data from IP
            location_data = get_ip_location(client_ip)
            if location_data:
                logger.info(f"Successfully got location data from IP: {json.dumps(location_data)}")
                user_context["location"] = location_data
                
                # If we have a timezone from geolocation, get local time immediately
                if location_data.get("timezone"):
                    logger.info(f"Getting local time from IP timezone: {location_data.get('timezone')}")
                    user_context["local_time"] = get_local_time(location_data.get("timezone"))
                    logger.info(f"Local time determined: {json.dumps(user_context['local_time'])}")
            else:
                logger.warning("Could not determine location from IP address")
        else:
            logger.warning("Could not extract client IP address")
        
        # Check if client directly provided location data in metadata (higher priority than IP)
        if metadata.get("location"):
            logger.info(f"Client provided location data: {json.dumps(metadata.get('location'))}")
            user_context["location"].update(metadata.get("location"))
        
        # Check if client directly provided timezone or local time data
        if metadata.get("timezone"):
            logger.info(f"Client provided timezone: {metadata.get('timezone')}")
            timezone = metadata.get("timezone")
            user_context["local_time"].update(get_local_time(timezone))
        
        if metadata.get("local_time"):
            logger.info(f"Client provided local time: {metadata.get('local_time')}")
            if "local_time" not in user_context:
                user_context["local_time"] = {}
            user_context["local_time"]["local_time"] = metadata.get("local_time")
            # Try to infer time of day from provided time if possible
            try:
                time_str = metadata.get("local_time")
                if ":" in time_str:  # Simple check for time format
                    hour_part = time_str.split(":")[0]
                    if "T" in hour_part:  # ISO format like 2023-01-01T14:30:00
                        hour_part = hour_part.split("T")[1]
                    hour = int(hour_part.strip())
                    
                    if 5 <= hour < 12:
                        user_context["local_time"]["time_of_day"] = "morning"
                    elif 12 <= hour < 17:
                        user_context["local_time"]["time_of_day"] = "afternoon"
                    elif 17 <= hour < 22:
                        user_context["local_time"]["time_of_day"] = "evening"
                    else:
                        user_context["local_time"]["time_of_day"] = "night"
                    
                    user_context["local_time"]["is_business_hours"] = 9 <= hour < 17
                    logger.info(f"Inferred time of day: {user_context['local_time']['time_of_day']}")
            except Exception as e:
                logger.warning(f"Could not parse time of day from provided local_time: {str(e)}")
        
        # Log what we found
        if user_context["location"]:
            location_str = ", ".join(f"{k}: {v}" for k, v in user_context["location"].items() if v)
            logger.info(f"Final user location context: {location_str}")
        else:
            logger.warning("No location context available")
        
        if user_context["local_time"]:
            time_str = ", ".join(f"{k}: {v}" for k, v in user_context["local_time"].items() if v)
            logger.info(f"Final user local time context: {time_str}")
        else:
            logger.warning("No local time context available")
        
        # Immediately store this context in a system message that will be saved to the database
        initial_context_message = {
            "role": "system",
            "content": "User context information collected at session start",
            "timestamp": get_utc_now().isoformat(),
            "metadata": {
                "type": "session_start_context",
                "user_location": user_context.get("location", {}),
                "user_local_time": user_context.get("local_time", {})
            }
        }
        
        # Keep track of conversation history
        conversation_history = []
        conversation_history.append(initial_context_message)
        
        # Immediately store the initial context to database to ensure we capture location data
        # Even if the session ends prematurely
        try:
            # Store location in a standalone message with detailed context
            detailed_content = "Session started with the following user context:\n"
            
            if user_context["location"]:
                loc = user_context["location"]
                detailed_content += f"- Location: {loc.get('city', 'Unknown')}, {loc.get('region', '')}, {loc.get('country', '')}\n"
                detailed_content += f"- Geolocation source: {loc.get('source', 'Unknown')}\n"
            else:
                detailed_content += "- Location: Could not determine user location\n"
                
            if user_context["local_time"]:
                time_data = user_context["local_time"]
                detailed_content += f"- Local time: {time_data.get('local_time', 'Unknown')}\n"
                detailed_content += f"- Timezone: {time_data.get('timezone', 'Unknown')}\n"
                detailed_content += f"- Time of day: {time_data.get('time_of_day', 'Unknown')}\n"
            else:
                detailed_content += "- Local time: Could not determine user local time\n"
            
            # Store context message in database immediately
            await store_conversation_message(
                session_id=session_id,
                role="system",
                content=detailed_content,
                participant_id="system",
                metadata={"type": "location_context", "session_start": True},
                location_data=user_context.get("location", {}),
                local_time_data=user_context.get("local_time", {})
            )
            logger.info("Successfully stored user location context in database immediately")
        except Exception as e:
            logger.error(f"Failed to store initial location context: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Full error details: {repr(e)}")
        
        # Create initial chat context with default system prompt
        logger.info(f"Using default system prompt: {DEFAULT_INTERVIEW_PROMPT[:100]}...")  # Log first 100 chars for debugging
        
        initial_ctx = llm.ChatContext().append(
            role="system",
            text=DEFAULT_INTERVIEW_PROMPT
        )
        logger.info("Created initial chat context with default system prompt")

        # Add location and time context to the system prompt if available
        location_context = ""
        if user_context["location"] or user_context["local_time"]:
            location_context = "\nUser context information:\n"
            
            if user_context["location"]:
                loc = user_context["location"]
                location_str = f"Location: {loc.get('city', '')}, {loc.get('region', '')}, {loc.get('country', '')}"
                location_context += location_str.replace(", ,", ",").replace(",,", ",") + "\n"
            
            if user_context["local_time"]:
                time = user_context["local_time"]
                location_context += f"Local time: {time.get('local_time', '')}\n"
                location_context += f"Time of day: {time.get('time_of_day', '')}\n"
                location_context += f"Day of week: {time.get('day_of_week', '')}\n"
            
            location_context += "\nYou may use this context information to personalize your responses when relevant.\n"
            initial_ctx.append(
                role="system",
                text=location_context
            )
            logger.info(f"Added user location and time context to system prompt")
        
        # Set session timeout (20 minutes)
        SESSION_TIMEOUT = 20 * 60  # 20 minutes in seconds
        session_start_time = asyncio.get_event_loop().time()
        timeout_task = None

        # Verify all required environment variables
        required_vars = ["DEEPGRAM_API_KEY", "OPENAI_API_KEY"]
        for var in required_vars:
            if not os.environ.get(var):
                raise Exception(f"Required environment variable {var} is not set")
            logger.info(f"Found {var} in environment variables")

        
        deepgram_api_key = os.environ.get("DEEPGRAM_API_KEY")
        openai_api_key = os.environ.get("OPENAI_API_KEY")


        logger.info("API keys validated successfully")

        # Initialize services with detailed error handling
        try:
            logger.info("Initializing Deepgram STT...")
            stt = deepgram.STT(
                model="nova-2-general",
                language="en-US",
                interim_results=True,
                punctuate=True,
                smart_format=True,
                sample_rate=16000,
                no_delay=True,
                endpointing_ms=25,
                filler_words=True,
                api_key=deepgram_api_key
            )
            logger.info("Deepgram STT initialized successfully")

            logger.info("Initializing OpenAI TTS...")
            try:
                tts = openai_tts.TTS(
                    model="tts-1",
                    voice=OPENAI_VOICE_ID,
                    api_key=openai_api_key
                )
                logger.info("OpenAI TTS initialized successfully")
            except Exception as tts_error:
                logger.error(f"Failed to initialize OpenAI TTS: {str(tts_error)}")
                raise

            logger.info("Initializing OpenAI LLM...")
            llm_instance = openai.LLM(
                model="ft:gpt-4o-mini-2024-07-18:improov::BFDKAqhD",
                api_key=openai_api_key
            )
            logger.info("OpenAI LLM initialized successfully with fine-tuned model")

            # Create the voice pipeline agent with the previously created initial_ctx
            logger.info("Creating VoicePipelineAgent...")
            agent = VoicePipelineAgent(
                stt=stt,
                tts=tts,
                vad=ctx.proc.userdata.get("vad"),  # Use safe dictionary access
                llm=llm_instance,
                min_endpointing_delay=0.5,
                max_endpointing_delay=5.0,
                chat_ctx=initial_ctx
            )
            logger.info("VoicePipelineAgent created successfully")

            # Add web search capability to the agent
            async def handle_web_search_requests(user_message: str):
                """
                Check if a user message requires web search and performs the search if needed.
                
                Args:
                    user_message (str): The user's message
                    
                Returns:
                    bool: True if web search was performed, False otherwise
                    str: The search results if search was performed, empty string otherwise
                """
                # List of trigger phrases that indicate a need for web search
                web_search_triggers = [
                    "search for", "find information about", "look up", 
                    "search the web", "recent information", "latest news",
                    "current trends", "what's new", "recent developments",
                    "current statistics", "latest statistics", "current market",
                    "job market", "hiring trends", "employment statistics",
                    "search", "recent", "latest", "current", "today", "this year",
                    "recent interview questions", "latest job trends"
                ]
                
                # Check if message contains trigger phrases
                if any(trigger in user_message.lower() for trigger in web_search_triggers):
                    logger.info(f"Web search trigger detected in message: {user_message}")
                    search_result = await perform_web_search(user_message, location_context=user_context)
                    return True, search_result
                
                return False, ""
            
            # Add system prompt information about web search
            web_search_system_info = """
            You have access to web search capabilities for EMERGENCY situations only. This is NOT a feature to be used regularly.
            
            Only use web search results when provided in these specific situations:
            
            1. When a user explicitly challenges the accuracy of your information with phrases like "that's not correct" or "you're wrong about that"
            2. When asked to verify very specific numeric data from 2023-2025 that you cannot confidently provide
            3. When explicitly asked for the "latest statistics" on employment, job markets, salaries, or industry trends
            
            In ALL other cases, rely on your existing knowledge to answer questions.
            
            Web search is an emergency-only feature reserved for preventing critical factual errors.
            When search results are provided to you, clearly indicate when you're incorporating this verified information.
            """
            
            # Update the initial context with web search capability
            initial_ctx.append(
                role="system",
                text=web_search_system_info
            )
            logger.info("Added web search capability to system prompt")

        except Exception as service_error:
            logger.error(f"Failed to initialize services: {str(service_error)}")
            logger.error(f"Error type: {type(service_error)}")
            logger.error(f"Full error details: {repr(service_error)}")
            raise

        # Function to store the entire conversation
        async def store_full_conversation():
            try:
                # Filter out system messages from the conversation history
                filtered_conversation = [msg for msg in conversation_history if msg.get("role") != "system"]
                
                # Create the conversation object with only the essential fields
                conversation_data = {
                    "session_id": session_id,
                    "participant_id": participant.identity,
                    "conversation": filtered_conversation,  # Store only non-system messages
                    "timestamp": get_utc_now().isoformat(),
                    "user_location": user_context.get("location", {}),  # Store location data
                    "user_local_time": user_context.get("local_time", {})  # Store time data
                }
                
                logger.info(f"Storing conversation with {len(filtered_conversation)} messages")
                logger.info(f"Last message in history: {json.dumps(filtered_conversation[-1] if filtered_conversation else None, indent=2)}")
                
                # Check if a record already exists for this session
                check_query = supabase.table("conversation_histories").select("*").eq("session_id", session_id).execute()
                
                if check_query.data:
                    logger.info(f"Updating existing conversation record for session {session_id}")
                    result = supabase.table("conversation_histories").update(conversation_data).eq("session_id", session_id).execute()
                else:
                    logger.info(f"Creating new conversation record for session {session_id}")
                    result = supabase.table("conversation_histories").insert(conversation_data).execute()
                
                if result.data:
                    logger.info(f"Successfully stored conversation. Response data: {json.dumps(result.data, indent=2)}")
                    return result
                else:
                    logger.warning("Store operation completed but returned no data")
                    logger.warning(f"Full result object: {result}")
                    return None
                    
            except Exception as e:
                logger.error(f"Failed to store conversation: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                logger.error(f"Error details: {repr(e)}")
                # Log the conversation history for debugging
                try:
                    logger.error(f"Conversation data being stored: {json.dumps(conversation_data, indent=2)}")
                except Exception as debug_err:
                    logger.error(f"Error logging conversation data: {str(debug_err)}")
                return None
        
        try:
            if not participant.metadata:
                logger.info("No metadata provided, using default configuration")
                default_metadata = {
                    "instructions": DEFAULT_INTERVIEW_PROMPT,
                    "modalities": "text_and_audio",
                    "voice": OPENAI_VOICE_ID,
                    "temperature": 0.8,
                    "max_output_tokens": 2048,
                    "turn_detection": json.dumps({
                        "type": "server_vad",
                        "threshold": 0.5,
                        "silence_duration_ms": 300,
                        "prefix_padding_ms": 200,
                    })
                }
                metadata = default_metadata
            else:
                metadata = json.loads(participant.metadata)
            logger.info(f"Using metadata: {metadata}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse metadata: {str(e)}")
            logger.error(f"Metadata type: {type(participant.metadata)}")
            logger.error(f"Metadata length: {len(participant.metadata) if participant.metadata else 0}")
            raise

        # Add system message to conversation history
        system_message = {
            "role": "system",
            "content": metadata.get("instructions", ""),
            "timestamp": get_utc_now().isoformat(),
            "metadata": {"session_start": True, "config": metadata}
        }
        conversation_history.append(system_message)
        
        # Store initial conversation
        await store_full_conversation()

        config = parse_session_config(metadata)
        logger.info(f"Starting with config: {config.to_dict()}")

        # Create initial chat context
        initial_ctx = llm.ChatContext().append(
            role="system",
            text=DEFAULT_INTERVIEW_PROMPT,
        )

        # Debug hook to print all events
        original_emit = agent.emit
        def debug_emit(event, *args, **kwargs):
            logger.info(f"EMIT: {event} with args: {args} kwargs: {kwargs}")
            # If this is an assistant message, ensure it's captured
            if event in ["message", "assistant_response", "agent_speech_committed", "agent_speech_interrupted"]:
                try:
                    msg = args[0] if args else None
                    if msg and hasattr(msg, 'role') and msg.role == "assistant":
                        msg_content = msg.content
                        if isinstance(msg_content, list):
                            msg_content = "\n".join(str(item) for item in msg_content)
                        
                        # Add to conversation history
                        assistant_message = {
                            "role": "assistant",
                            "content": msg_content,
                            "timestamp": get_utc_now().isoformat(),
                            "metadata": {"type": "response", "event": event}
                        }
                        conversation_history.append(assistant_message)
                        
                        # Store updated conversation
                        asyncio.create_task(store_full_conversation())
                        logger.info(f"Added assistant message to conversation history from {event}, total messages: {len(conversation_history)}")
                except Exception as e:
                    logger.error(f"Error in debug_emit handler for {event}: {str(e)}")
                    logger.error(f"Error type: {type(e)}")
                    logger.error(f"Full error details: {repr(e)}")
            
            return original_emit(event, *args, **kwargs)
        agent.emit = debug_emit

        # Add preparatory method to pre-generate TTS audio and reduce latency
        # This wraps the original agent to add a prepare_say method
        original_say = agent.say
        _tts_cache = {}  # Cache for prepared TTS audio
        
        async def prepare_say(text: str):
            """Pre-generate TTS audio to reduce latency when actually speaking"""
            try:
                # Skip if already in cache
                if text in _tts_cache:
                    logger.info(f"Using cached TTS audio for: {text[:50]}...")
                    return
                
                logger.info(f"Pre-generating TTS audio for: {text[:50]}...")
                # Use the TTS service directly to generate audio
                audio_data = await agent.tts.synthesize(text)
                
                # Store in cache
                _tts_cache[text] = audio_data
                logger.info(f"TTS audio prepared and cached ({len(audio_data)} bytes)")
                return audio_data
            except Exception as e:
                logger.error(f"Error preparing TTS audio: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                logger.error(f"Full error details: {repr(e)}")
                # Don't raise, just log - we'll fall back to regular synthesis
        
        async def optimized_say(text: str, allow_interruptions=True):
            """Enhanced say method that uses cached audio when available"""
            try:
                # Try to use cached audio if available
                if text in _tts_cache:
                    logger.info(f"Using cached audio for speech")
                    # Use the pre-generated audio directly if available
                    # This requires implementation details of the agent's say method
                    # If direct use isn't possible, we just let the original method run
                    # which will be faster on second attempt due to internal caching
                
                # Fall back to original say method (which will be faster if we've pre-cached)
                return await original_say(text, allow_interruptions=allow_interruptions)
            except Exception as e:
                logger.error(f"Error in optimized say: {str(e)}")
                # Fall back to original say method
                return await original_say(text, allow_interruptions=allow_interruptions)
        
        # Attach the new methods to the agent
        agent.prepare_say = prepare_say
        agent.say = optimized_say

        # Set up metrics collection
        usage_collector = metrics.UsageCollector()

        @agent.on("metrics_collected")
        def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
            metrics.log_metrics(agent_metrics)
            usage_collector.collect(agent_metrics)

        # Create a task to check session timeout and conversation end detection
        last_message_time = asyncio.get_event_loop().time()  # Define this at the outer scope
        
        async def check_session_timeout():
            try:
                nonlocal last_message_time  # Now this reference is valid
                inactive_count = 0
                
                while True:
                    current_time = asyncio.get_event_loop().time()
                    elapsed_time = current_time - session_start_time
                    time_since_last_message = current_time - last_message_time
                    
                    # Check if session timeout reached
                    if elapsed_time >= SESSION_TIMEOUT:
                        logger.info("Session timeout reached (20 minutes). Ending session...")
                        goodbye_message = "I apologize, but our session time (20 minutes) has come to an end. Thank you for the great conversation! Feel free to start a new session if you'd like to continue practicing."
                        
                        # Use say_and_store to ensure the message is in the conversation history
                        await say_and_store(goodbye_message)
                        
                        # Wait for the goodbye message to be sent
                        await asyncio.sleep(5)
                        # Close the room connection
                        await ctx.room.disconnect()
                        return
                    
                    # Detect conversation inactivity (user hasn't spoken in a while)
                    if time_since_last_message > 120:  # 2 minutes of inactivity
                        inactive_count += 1
                        if inactive_count >= 2:  # Check multiple times to confirm inactivity
                            logger.info("Conversation appears to be complete (no activity for 4+ minutes)")
                            goodbye_message = "It seems our conversation has come to a natural end. Thank you for chatting with me today! If you need more assistance in the future, feel free to start a new session. Goodbye!"
                            
                            await say_and_store(goodbye_message)
                            await asyncio.sleep(5)
                            await ctx.room.disconnect()
                            logger.info("Agent disconnected after detected conversation end")
                            return
                        
                        # First time detecting inactivity, ask if user wants to continue
                        if inactive_count == 1:
                            logger.info("Detected inactivity, checking if conversation is over")
                            prompt_message = "I notice we haven't spoken for a couple of minutes. Is there anything else I can help you with today? If not, we can end our session."
                            await say_and_store(prompt_message)
                            # Reset the timer to give the user a chance to respond
                            last_message_time = current_time
                    else:
                        # Reset counter if there's activity
                        inactive_count = 0
                    
                    # Periodically store conversation history
                    if elapsed_time % 60 < 10:  # Store every minute
                        await store_full_conversation()
                    
                    await asyncio.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Error in timeout checker: {str(e)}")
                raise
            finally:
                logger.info("Timeout checker task ended")

        # Start the timeout checker task
        timeout_task = asyncio.create_task(check_session_timeout())
        
        # Update last_message_time when user speaks
        @agent.on("user_speech_committed")
        def on_user_speech_committed(msg: llm.ChatMessage):
            try:
                nonlocal last_message_time
                last_message_time = asyncio.get_event_loop().time()
                
                logger.info(f"User speech committed: {msg.content}")
                # Convert message content list (if any, e.g., images) to a string
                if isinstance(msg.content, list):
                    msg_content = "\n".join("[image]" if isinstance(x, ChatImage) else x for x in msg.content)
                else:
                    msg_content = msg.content
                
                # Check if web search is needed for this message
                asyncio.create_task(process_user_message_with_search(msg_content))
                
                # Add to conversation history with location context
                user_message = {
                    "role": "user",
                    "content": msg_content,
                    "timestamp": get_utc_now().isoformat(),
                    "metadata": {
                        "type": "transcription", 
                        "is_final": True,
                        "user_location": user_context.get("location", {}),
                        "user_local_time": user_context.get("local_time", {})
                    }
                }
                conversation_history.append(user_message)
                
                # Check for explicit end phrases
                end_phrases = ["goodbye", "bye", "end conversation", "end session", "that's all", 
                              "thank you that's all", "we're done", "that will be all"]
                
                if any(phrase in msg_content.lower() for phrase in end_phrases):
                    logger.info("User indicated end of conversation")
                    asyncio.create_task(end_conversation())
                
                # Store updated conversation
                asyncio.create_task(store_full_conversation())
                logger.info(f"Added user message to conversation history, total messages: {len(conversation_history)}")
            except Exception as e:
                logger.error(f"Error in user speech callback: {str(e)}")
                
        # Function to end the conversation gracefully
        async def end_conversation():
            try:
                logger.info("Ending conversation gracefully")
                goodbye_message = "Thank you for chatting with me today! I hope I was able to help with your professional needs. Have a great day!"
                
                # Say goodbye and store the message
                await say_and_store(goodbye_message)
                
                # Wait for the message to be delivered
                await asyncio.sleep(5)
                
                # Disconnect from the room
                await ctx.room.disconnect()
                logger.info("Agent disconnected after conversation end")
            except Exception as e:
                logger.error(f"Error ending conversation: {str(e)}")

        # Process user message with web search capability
        async def process_user_message_with_search(msg_content: str):
            try:
                # SEVERELY constrained web search - only in true emergencies or clear hallucination risks
                # Core emergency triggers - much more limited than before
                strict_emergency_triggers = [
                    "2025 data", 
                    "latest statistics", "current statistics", 
                    "fact check this", "verify this fact",
                    "that's incorrect information", "that's not right",
                    "citation needed", "source needed"
                ]
                
                # Initial check for strict emergency triggers - explicit requests for verification
                needs_search = any(trigger in msg_content.lower() for trigger in strict_emergency_triggers)
                
                # If no emergency trigger found, check for explicit challenges to the agent's knowledge
                if not needs_search:
                    challenge_patterns = [
                        "you're wrong about", "that's not correct", "that information is outdated",
                        "that data is old", "that's false", "you're mistaken about",
                        "you need to update your information on", "you've made an error about"
                    ]
                    needs_search = any(pattern in msg_content.lower() for pattern in challenge_patterns)
                
                # Final check - only for VERY specific factual questions that require current data
                # Must contain year reference AND specific request for statistics/numbers
                if not needs_search:
                    has_year_reference = any(year in msg_content.lower() for year in ["2024", "2025"])
                    has_data_request = any(term in msg_content.lower() for term in ["statistics", "percentage", "data", "numbers", "rate", "survey"])
                    
                    needs_search = has_year_reference and has_data_request
                
                # Log the decision with detailed reasoning
                if needs_search:
                    logger.info(f"EMERGENCY web search triggered: {msg_content}")
                    logger.info("Reason: Detected critical challenge to agent knowledge or specific request for current data")
                    
                    # Add an extra check - only search for questions we're likely to hallucinate on
                    hallucination_prone_topics = [
                        "employment rate", "job market", "industry growth", "salary survey",
                        "career statistics", "hiring trends", "workforce data", "economic indicators", 
                        "technology adoption", "industry standards", "market share", "professional certification"
                    ]
                    
                    is_hallucination_prone = any(topic in msg_content.lower() for topic in hallucination_prone_topics)
                    
                    if not is_hallucination_prone:
                        logger.info("Canceling web search - topic not in hallucination-prone category")
                        return
                    
                    # Proceed with web search only in true emergency situations
                    search_result = await perform_web_search(msg_content, location_context=user_context)
                    
                    if search_result and not search_result.startswith("Sorry, I encountered an issue"):
                        logger.info(f"Emergency web search results obtained for hallucination prevention")
                        
                        # Add search results to conversation as a system message
                        system_search_message = {
                            "role": "system",
                            "content": f"EMERGENCY web search results for '{msg_content}':\n\n{search_result}",
                            "timestamp": get_utc_now().isoformat(),
                            "metadata": {"type": "web_search", "query": msg_content, "reason": "critical_hallucination_prevention"}
                        }
                        conversation_history.append(system_search_message)
                        
                        # Update agent's context with search results - emphasize emergency-only usage
                        agent.chat_ctx.append(
                            role="system",
                            text=f"EMERGENCY FACTUAL VERIFICATION: {search_result}\n\nOnly use these search results if absolutely necessary to correct critical factual errors. Otherwise, rely on your existing knowledge."
                        )
                        
                        # Store updated conversation
                        await store_full_conversation()
                        logger.info(f"Added emergency web search results to conversation history")
                    else:
                        logger.info(f"No relevant emergency web search results found or error occurred")
                else:
                    logger.info(f"No emergency web search needed - using existing model knowledge")
            except Exception as e:
                logger.error(f"Error processing message with web search: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                logger.error(f"Full error details: {repr(e)}")

        # Helper function to store assistant message
        async def store_assistant_message(msg_content: str, event_type: str):
            try:
                # Check if this exact message is already in the conversation history
                is_duplicate = False
                for existing_msg in conversation_history:
                    if (existing_msg.get('role') == 'assistant' and 
                        existing_msg.get('content') == msg_content):
                        is_duplicate = True
                        logger.info(f"Skipping duplicate assistant message from {event_type}")
                        break
                
                if not is_duplicate:
                    # Add to conversation history
                    assistant_message = {
                        "role": "assistant",
                        "content": msg_content,
                        "timestamp": get_utc_now().isoformat(),
                        "metadata": {
                            "type": "response", 
                            "event": event_type,
                            "user_location": user_context.get("location", {}),
                            "user_local_time": user_context.get("local_time", {})
                        }
                    }
                    conversation_history.append(assistant_message)
                    
                    # Store updated conversation
                    await store_full_conversation()
                    logger.info(f"Added assistant message to conversation history from {event_type}, total messages: {len(conversation_history)}")
            except Exception as e:
                logger.error(f"Error storing assistant message from {event_type}: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                logger.error(f"Full error details: {repr(e)}")

        # Try multiple events for assistant responses
        @agent.on("assistant_response")
        def on_assistant_response(msg: llm.ChatMessage):
            try:
                logger.info(f"ASSISTANT RESPONSE EVENT FIRED: {msg.content}")
                if isinstance(msg.content, list):
                    msg_content = "\n".join(str(item) for item in msg.content)
                else:
                    msg_content = msg.content
                
                # Begin pre-generating the TTS audio for this response immediately
                # This significantly reduces the lag between transcript and audio
                if hasattr(agent, 'prepare_say'):
                    asyncio.create_task(agent.prepare_say(msg_content))
                    logger.info(f"Started preparing TTS for response")
                
                # Don't immediately add to conversation history - the agent will
                # show the message when it starts speaking
                # Instead, we just prepare the audio and let the agent's speech handling
                # manage the transcript timing 
                
                # Only store messages that don't get spoken immediately
                # (agent.say will handle transcript for immediate responses)
                asyncio.create_task(store_assistant_message(msg_content, "assistant_response"))
            except Exception as e:
                logger.error(f"Error in assistant_response callback: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                logger.error(f"Full error details: {repr(e)}")

        @agent.on("message_sent")
        def on_message_sent(msg: Any):
            try:
                logger.info(f"MESSAGE SENT EVENT FIRED: {msg}")
                
                # Check if this is an assistant message
                if hasattr(msg, 'role') and msg.role == "assistant":
                    if hasattr(msg, 'content'):
                        if isinstance(msg.content, list):
                            msg_content = "\n".join(str(item) for item in msg.content)
                        else:
                            msg_content = msg.content
                        
                        asyncio.create_task(store_assistant_message(msg_content, "message_sent"))
            except Exception as e:
                logger.error(f"Error in message_sent handler: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                logger.error(f"Full error details: {repr(e)}")

        @agent.on("llm_response_chunk")
        def on_llm_response_chunk(chunk: Any):
            logger.info(f"LLM RESPONSE CHUNK EVENT FIRED: {chunk}")
            # We don't add partial chunks to the conversation, just log them
        
        @agent.on("llm_response_complete")
        def on_llm_response_complete(msg: Any):
            try:
                logger.info(f"LLM RESPONSE COMPLETE EVENT FIRED: {msg}")
                
                # Check if this is an assistant message with content
                if hasattr(msg, 'role') and msg.role == "assistant" and hasattr(msg, 'content'):
                    if isinstance(msg.content, list):
                        msg_content = "\n".join(str(item) for item in msg.content)
                    else:
                        msg_content = msg.content
                    
                    asyncio.create_task(store_assistant_message(msg_content, "llm_response_complete"))
            except Exception as e:
                logger.error(f"Error in llm_response_complete handler: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                logger.error(f"Full error details: {repr(e)}")

        # Add a message handler to log all messages for debugging
        @agent.on("message")
        def on_message(msg: Any):
            try:
                logger.info(f"MESSAGE EVENT RECEIVED: {msg}")
                if hasattr(msg, 'content'):
                    logger.info(f"Message content: {msg.content}")
                if hasattr(msg, 'role'):
                    logger.info(f"Message role: {msg.role}")
                    
                    # If this is an assistant message that we haven't captured through other events
                    if msg.role == "assistant" and hasattr(msg, 'content'):
                        # Check if we already have this message in our history
                        msg_content = msg.content
                        if isinstance(msg_content, list):
                            msg_content = "\n".join(str(item) for item in msg_content)
                        
                        asyncio.create_task(store_assistant_message(msg_content, "message"))
            except Exception as e:
                logger.error(f"Error in message handler: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                logger.error(f"Full error details: {repr(e)}")

        # Start the agent
        agent.start(ctx.room)

        # Handle participant disconnection (e.g., when user clicks "end call" button)
        @ctx.room.on("participant_disconnected")
        async def on_participant_disconnected(participant: rtc.Participant):
            logger.info(f"Participant disconnected: {participant.identity}")
            logger.info("Frontend user ended the call - cleaning up resources")
            
            # Store final conversation state
            await store_full_conversation()
            
            # Add a system message about the call being ended by user
            end_message = {
                "role": "system",
                "content": "Call ended by user via frontend",
                "timestamp": get_utc_now().isoformat(),
                "metadata": {"type": "call_end", "reason": "user_ended"}
            }
            conversation_history.append(end_message)
            await store_full_conversation()
            
            # Disconnect the room (this will happen automatically, but we make it explicit)
            try:
                if timeout_task and not timeout_task.done():
                    timeout_task.cancel()
                    logger.info("Cancelled timeout task")
                
                # Give a moment for cleanup and then disconnect
                await asyncio.sleep(1)
                await ctx.room.disconnect()
                logger.info("Room disconnected after user ended call")
            except Exception as e:
                logger.error(f"Error during room disconnect after user ended call: {str(e)}")

        # Helper function to say something and ensure it's in the conversation history
        async def say_and_store(message_text):
            try:
                # Create a message but don't add to conversation history yet
                # This delays the transcript until the speech is about to begin
                assistant_message = {
                    "role": "assistant",
                    "content": message_text,
                    "timestamp": get_utc_now().isoformat(),
                    "metadata": {
                        "type": "direct_say",
                        "user_location": user_context.get("location", {}),
                        "user_local_time": user_context.get("local_time", {})
                    }
                }
                
                logger.info(f"Preparing to speak: {message_text[:50]}...")
                
                # First, prepare the audio - this caches the audio before playback begins
                # This helps reduce the delay between transcript and audio
                preparation_task = asyncio.create_task(agent.prepare_say(message_text))
                
                # Wait for audio preparation to complete (or timeout after 5 seconds)
                try:
                    await asyncio.wait_for(preparation_task, timeout=5.0)
                    logger.info(f"Audio prepared and ready for playback")
                except asyncio.TimeoutError:
                    logger.warning(f"Audio preparation timed out, proceeding anyway")
                
                # Now add to conversation history right before speaking
                # This ensures transcription appears at roughly the same time as audio
                conversation_history.append(assistant_message)
                
                # Store in database immediately before speech begins
                await store_full_conversation()
                
                # Now begin speaking (speech and transcript should be closely synchronized)
                await agent.say(message_text, allow_interruptions=True)
                
                logger.info("Speech completed")
            except Exception as e:
                logger.error(f"Error in say_and_store: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                logger.error(f"Full error details: {repr(e)}")
                
                # Ensure the message is still stored even if speaking fails
                if assistant_message not in conversation_history:
                    conversation_history.append(assistant_message)
                    await store_full_conversation()
                logger.error("Stored message despite speech error")

        # Send initial welcome message
        initial_message = "Hi, I am Prepzo. I can help you with any professional problem you're having. I have access to the latest information through web search, so feel free to ask me about current job trends, recent interview practices, or any career-related questions."
        
        # Use the optimized say_and_store method for the welcome message
        # This ensures that the transcript and speech are synchronized from the start
        await say_and_store(initial_message)

    finally:
        # Cleanup timeout task if it exists
        if timeout_task and not timeout_task.done():
            timeout_task.cancel()
            try:
                await timeout_task
            except asyncio.CancelledError:
                pass
            logger.info("Timeout task cleaned up")

def prewarm(proc: JobProcess):
    """
    Preload the voice activity detector (VAD) from Silero.
    """
    proc.userdata["vad"] = silero.VAD.load()

if __name__ == "__main__":
    # Create a synchronous wrapper that runs the async init in a new event loop
    def sync_init_supabase():
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            init_result = loop.run_until_complete(init_supabase())
            if not init_result:
                logger.error("Failed to initialize Supabase client. Please check your environment variables.")
                logger.error("Required environment variables:")
                logger.error("- SUPABASE_URL")
                logger.error("- SUPABASE_SERVICE_ROLE_KEY")
                logger.error("- CARTESIA_API_KEY")
                logger.error("- DEEPGRAM_API_KEY")
                raise Exception("Failed to initialize Supabase client")
            logger.info("Supabase initialized successfully")
            return init_result
        except Exception as e:
            logger.error(f"Error during Supabase initialization: {str(e)}")
            return False
        finally:
            loop.close()
    
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
    
    # Run the initialization before starting the app
    if not sync_init_supabase():
        logger.error("Supabase initialization failed, exiting")
        import sys
        sys.exit(1)
    
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
            prewarm_fnc=prewarm,
            worker_type=WorkerType.ROOM
        )
    
    # Start the app after successful initialization
    cli.run_app(worker_options)