from __future__ import annotations

import asyncio
import json
import logging
import uuid
import os
import pathlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal
from datetime import datetime

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

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)

# Initialize Supabase client
supabase: Client = None

# OpenAI TTS configuration
OPENAI_VOICE_ID = "alloy"  

# Default interview preparation prompt
DEFAULT_INTERVIEW_PROMPT = prompt.prompt

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
    metadata: Dict[str, Any] = None
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
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        logger.info(f"Attempting to store message in conversation_sessions:")
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Role: {role}")
        logger.info(f"Participant ID: {participant_id}")
        logger.info(f"Content length: {len(content)}")
        logger.info(f"Metadata: {metadata}")
        
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
        
        # Parse metadata safely (but only for other configs, not for system prompt)
        try:
            metadata = json.loads(participant.metadata) if participant.metadata else {}
            logger.info(f"Parsed metadata: {metadata}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse metadata: {str(e)}, using empty dict")
            metadata = {}
        
        # Always use the default system prompt
        logger.info(f"Using default system prompt: {DEFAULT_INTERVIEW_PROMPT[:100]}...")  # Log first 100 chars for debugging
        
        initial_ctx = llm.ChatContext().append(
            role="system",
            text=DEFAULT_INTERVIEW_PROMPT
        )
        logger.info("Created initial chat context with default system prompt")

        # Generate a unique session ID
        session_id = ctx.room.name
        logger.info(f"Starting new session with ID: {session_id}")
        
        # Keep track of conversation history
        conversation_history = []
        
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
                model="gpt-4o-mini",
                api_key=openai_api_key
            )
            logger.info("OpenAI LLM initialized successfully")

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
                    "timestamp": datetime.utcnow().isoformat()
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
            "timestamp": datetime.utcnow().isoformat(),
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
                            "timestamp": datetime.utcnow().isoformat(),
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

        # Set up metrics collection
        usage_collector = metrics.UsageCollector()

        @agent.on("metrics_collected")
        def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
            metrics.log_metrics(agent_metrics)
            usage_collector.collect(agent_metrics)

        @agent.on("user_speech_committed")
        def on_user_speech_committed(msg: llm.ChatMessage):
            try:
                logger.info(f"User speech committed: {msg.content}")
                # Convert message content list (if any, e.g., images) to a string
                if isinstance(msg.content, list):
                    msg_content = "\n".join("[image]" if isinstance(x, ChatImage) else x for x in msg.content)
                else:
                    msg_content = msg.content
                
                # Add to conversation history
                user_message = {
                    "role": "user",
                    "content": msg_content,
                    "timestamp": datetime.utcnow().isoformat(),
                    "metadata": {"type": "transcription", "is_final": True}
                }
                conversation_history.append(user_message)
                
                # Store updated conversation
                asyncio.create_task(store_full_conversation())
                logger.info(f"Added user message to conversation history, total messages: {len(conversation_history)}")
            except Exception as e:
                logger.error(f"Error in user speech callback: {str(e)}")

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
                        "timestamp": datetime.utcnow().isoformat(),
                        "metadata": {"type": "response", "event": event_type}
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

        # Helper function to say something and ensure it's in the conversation history
        async def say_and_store(message_text):
            try:
                # Add to conversation history first
                assistant_message = {
                    "role": "assistant",
                    "content": message_text,
                    "timestamp": datetime.utcnow().isoformat(),
                    "metadata": {"type": "direct_say"}
                }
                conversation_history.append(assistant_message)
                
                # Store updated conversation immediately
                await store_full_conversation()
                logger.info(f"Added direct assistant message to conversation history: {message_text[:50]}...")
                logger.info(f"Total messages in history: {len(conversation_history)}")
                
                # Now say it
                await agent.say(message_text, allow_interruptions=True)
                
                # Double-check storage after a short delay
                await asyncio.sleep(0.5)
                await store_full_conversation()
                logger.info("Double-checked conversation storage after speaking")
            except Exception as e:
                logger.error(f"Error in say_and_store: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                logger.error(f"Full error details: {repr(e)}")
                # Try to store the message even if speaking fails
                try:
                    await store_full_conversation()
                except Exception as store_err:
                    logger.error(f"Failed to store conversation after error: {str(store_err)}")

        # Send initial welcome message
        initial_message = "Hi, I am Prepzo I can help you with any professional problem you are having."
        
        # Add to conversation history
        welcome_message = {
            "role": "assistant",
            "content": initial_message,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {"type": "welcome_message"}
        }
        conversation_history.append(welcome_message)
        
        # Store updated conversation
        await store_full_conversation()
        
        # Use the regular say method for the welcome message
        await agent.say(initial_message, allow_interruptions=True)

        # Create a task to check session timeout
        async def check_session_timeout():
            try:
                while True:
                    current_time = asyncio.get_event_loop().time()
                    elapsed_time = current_time - session_start_time
                    
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
    # Initialize Supabase before starting the app
    if not init_supabase ():
        logger.error("Failed to initialize Supabase client. Please check your environment variables.")
        logger.error("Required environment variables:")
        logger.error("- SUPABASE_URL")
        logger.error("- SUPABASE_SERVICE_ROLE_KEY")
        logger.error("- CARTESIA_API_KEY")
        logger.error("- DEEPGRAM_API_KEY")
        raise Exception("Failed to initialize Supabase client")
    
    logger.info("Supabase initialized successfully")
    
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            worker_type=WorkerType.ROOM
        )
    )