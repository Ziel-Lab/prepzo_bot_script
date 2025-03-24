# Prepzo AI Career Coach

Prepzo is an AI-powered career coaching platform that provides personalized professional guidance through natural voice conversations. The system uses advanced AI to engage in meaningful career development discussions, helping users optimize their career paths and achieve their professional goals.

## Features

- **Natural Voice Interaction**: Engage in real-time voice conversations with the AI career coach
- **Structured Career Guidance**: Receive personalized career advice and development plans
- **Multi-modal Communication**: Support for both text and audio interactions
- **Session Management**: Automatic session handling with timeout protection
- **Conversation History**: Persistent storage of coaching sessions for future reference
- **Environment Configuration**: Flexible setup for different deployment environments

## Prerequisites

- Python 3.8 or higher
- Git
- Access to the following API services:
  - LiveKit (for real-time communication)
  - Deepgram (for speech-to-text)
  - OpenAI (for LLM and text-to-speech)
  - Supabase (for data storage)

## Environment Variables

Create a `.env` file in the backend directory with the following variables:

```env
# LiveKit Configuration
LIVEKIT_URL=your_livekit_url
LIVEKIT_API_KEY=your_livekit_api_key

# API Keys
DEEPGRAM_API_KEY=your_deepgram_api_key
OPENAI_API_KEY=your_openai_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key

# Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd backend
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

## Project Structure

```
backend/
├── main.py              # Main application entry point
├── prompt.py           # AI coaching prompt configuration
├── requirements.txt    # Python dependencies
└── .env               # Environment variables (not in repo)
```

## Usage

1. Start the backend server:
```bash
python main.py
```

2. The server will initialize and wait for client connections.

3. Sessions are limited to 20 minutes by default.

## AI Coaching Approach

The AI coach (Prepzo) follows a structured approach:

1. **Trust Building**
   - Introduces itself as Prepzo
   - Requests brief professional background

2. **Focused Questioning**
   - Explores rewarding roles
   - Identifies key achievements
   - Discusses future goals

3. **Challenge Identification**
   - Identifies current professional hurdles
   - Develops targeted solutions

4. **Action Planning**
   - Offers strategic job opportunities
   - Suggests skill-building resources
   - Helps refine career narrative

## Technical Details

### Voice Processing
- Uses Deepgram for speech-to-text conversion
- OpenAI TTS for text-to-speech synthesis
- Silero VAD for voice activity detection

### Data Storage
- Supabase for conversation history storage
- Structured data model for session management

### Session Management
- 20-minute session timeout
- Automatic conversation storage
- Graceful session termination

## Development

### Adding New Features
1. Create a new branch for your feature
2. Make your changes
3. Test thoroughly
4. Submit a pull request

### Testing
```bash
# Run tests
python -m pytest

# Run with coverage
python -m pytest --cov=.
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request


