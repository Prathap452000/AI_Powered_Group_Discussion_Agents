# AI Group Discussion System

An AI-powered voice discussion platform where multiple AI agents analyze documents and engage in natural conversations, with human participation.

## Features

- 🗣️ **Multi-agent voice discussions** - Two AI agents with distinct personalities debate document content
- 🎙️ **Real-time voice synthesis** - Natural-sounding voices with different accents
- 📄 **Document analysis** - Upload PDFs, TXT, or DOCX files for discussion
- 💬 **Human participation** - Join the conversation anytime
- ⚡ **Gemini AI integration** - Powered by Google's latest AI models
- 🎚️ **Conversation controls** - Adjust speech, interrupt, or guide the discussion

## How It Works

1. Upload a document (PDF, TXT, or DOCX)
2. AI agents analyze the content and begin discussing it
3. Listen to their conversation or join in via text/voice
4. Guide the discussion with your questions and input

## Installation

### Prerequisites
- Python 3.8+
- Google Gemini API key
- FFmpeg (for audio processing)

### Setup

```bash
# Clone the repository
git clone https://github.com/Prathap452000/AI_Powered_Group_Discussion_Agents.git

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
GEMINI_API_KEY=your_api_key_here
FLASK_SECRET_KEY=your_secret_key

# Run the Flask application
python app.py

# Access the web interface at http://localhost:5000

#Folder structure

├── app.py                # Main application
├── templates/            # HTML templates
├── uploads/              # Document upload directory
├── requirements.txt      # Python dependencies
└── README.md             # This file

