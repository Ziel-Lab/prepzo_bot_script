# Prepzo Bot

An AI voice agent powered by LiveKit, OpenAI, and Deepgram for professional interview preparation and career coaching.

## Features

- Real-time voice conversations with an AI assistant
- Interview preparation and career coaching
- Location-aware responses using geolocation
- Emergency web search for up-to-date information
- Automatic session timeout and conversation management
- Conversation history stored in Supabase

## Technical Components

- **Voice Agent**: Built using LiveKit's voice agent framework
- **Speech-to-Text**: Deepgram Nova-2 model for accurate transcription
- **Text-to-Speech**: OpenAI TTS for natural-sounding voice responses
- **LLM**: Fine-tuned GPT-4o model for interview coaching
- **Storage**: Supabase for conversation history and analytics
- **Deployment**: AWS EC2 via GitHub Actions and AWS CDK

## Prerequisites

- Python 3.10+
- LiveKit server access
- Supabase account and project
- OpenAI API access
- Deepgram API access

## Environment Variables

The following environment variables need to be set:

```
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_supabase_key
OPENAI_API_KEY=your_openai_key
DEEPGRAM_API_KEY=your_deepgram_key
LIVEKIT_URL=your_livekit_url
LIVEKIT_API_KEY=your_livekit_api_key
```

## Local Development Setup

1. Clone the repository
```bash
   git clone https://github.com/your-username/prepzo_bot_script.git
   cd prepzo_bot_script
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up environment variables by creating a `.env` file with the required values

5. Run the application
```bash
python main.py
```

## Deployment

This project uses GitHub Actions for CI/CD to AWS EC2. See [GITHUB_ACTIONS_SETUP.md](GITHUB_ACTIONS_SETUP.md) for detailed setup instructions.

## Architecture

The agent follows this architecture:
1. User's voice is captured and transcribed in real-time by Deepgram
2. The transcription is sent to a fine-tuned GPT-4o model
3. The model generates a response, which is converted to speech by OpenAI TTS
4. The speech is played back to the user in real-time
5. Conversation history is stored in Supabase for later analysis

## License

[MIT License](LICENSE)


