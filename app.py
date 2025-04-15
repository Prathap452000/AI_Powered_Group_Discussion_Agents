import os
import tempfile
import threading
import time
import uuid
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from gtts import gTTS
import pygame
import google.generativeai as genai
import PyPDF2
from collections import deque

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24).hex())

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global state
conversations = {}
conversation_threads = {}
user_typing = {}
audio_lock = threading.Lock()
pygame_initialized = False

# Initialize pygame mixer once at startup
def initialize_pygame_mixer():
    global pygame_initialized
    try:
        pygame.mixer.init()
        pygame_initialized = True
    except Exception as e:
        app.logger.error(f"Failed to initialize pygame mixer: {e}")
        pygame_initialized = False

# Initialize pygame at startup
initialize_pygame_mixer()

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file):
    """Extract text from PDF or TXT files"""
    file_extension = file.filename.rsplit('.', 1)[1].lower()
    
    if file_extension == 'pdf':
        reader = PyPDF2.PdfReader(file)
        return "\n".join(page.extract_text() for page in reader.pages)
    elif file_extension == 'txt':
        return file.read().decode('utf-8')
    return ""

def speak_response(text, agent=None):
    """Convert text to speech using gTTS and play with pygame"""
    global pygame_initialized
    
    # Use different TLD for different agents to get variety in voices
    tld = 'com'
    if agent == "Agent 2":
        tld = 'co.uk'
    
    # Create a temporary file with a unique name
    temp_file_path = None
    try:
        # Generate speech
        tts = gTTS(text=text, lang='en', tld=tld)
        
        # Create a unique temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            temp_file_path = temp_audio_file.name
        
        # Save to temporary file
        tts.save(temp_file_path)
        
        with audio_lock:
            # Ensure pygame mixer is initialized
            if not pygame_initialized:
                initialize_pygame_mixer()
                
            if pygame_initialized:
                # Load and play the audio
                pygame.mixer.music.load(temp_file_path)
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                
                # Add a small delay to ensure file is released
                time.sleep(0.5)
            else:
                app.logger.error("Cannot play audio: pygame mixer not initialized")
                time.sleep(len(text) * 0.1)  # Simulate speaking time based on text length
                
    except Exception as e:
        app.logger.error(f"Error playing audio: {e}")
        # Simulate speaking time based on text length
        time.sleep(len(text) * 0.1)
    finally:
        # Make sure pygame.mixer.music is stopped
        if pygame_initialized:
            pygame.mixer.music.stop()
            
        # Clean up temporary file with retry
        if temp_file_path and os.path.exists(temp_file_path):
            for _ in range(5):  # Try 5 times
                try:
                    os.remove(temp_file_path)
                    break
                except PermissionError:
                    # Wait and retry if file is still in use
                    time.sleep(0.5)
                except Exception as e:
                    app.logger.error(f"Error removing temporary file: {e}")
                    break

def generate_agent_response(agent_name, document_text, conversation_history):
    """Generate response from an agent with personality"""
    try:
        # Agent personalities
        personalities = {
            "Agent 1": {
                "name": "Alex",
                "traits": "critical thinker, analytical, direct",
                "style": "challenges assumptions, asks probing questions"
            },
            "Agent 2": {
                "name": "Jordan",
                "traits": "empathetic, optimistic, creative thinker",
                "style": "builds on ideas, uses metaphors, supportive"
            }
        }
        
        persona = personalities.get(agent_name, personalities["Agent 1"])
        
        # Format conversation context (last 5 messages)
        context_messages = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
        context = "\n".join(f"{msg['speaker']}: {msg['text']}" for msg in context_messages)
        
        # Generate prompt with shorter document excerpt to reduce token usage
        prompt = f"""
        You are {persona['name']}, a {persona['traits']} participating in a professional discussion.
        Style: {persona['style']}. Keep responses under 30 words, concise and natural.
        
        Document excerpt (discussing this):
        {document_text[:1000]}[...truncated]
        
        Recent conversation:
        {context}
        
        Respond naturally as {persona['name']} in a brief, conversational way.
        """
        
        response = model.generate_content(prompt)
        return response.text.strip().replace('"', '')
        
    except Exception as e:
        app.logger.error(f"Response generation error: {str(e)}")
        return "I need a moment to gather my thoughts on that."

def get_document_topic(document_text):
    """Determine the main topic of the document"""
    try:
        # Use a small portion of the document to save tokens
        excerpt = document_text[:500] if len(document_text) > 500 else document_text
        
        prompt = """
        Based on this document excerpt, provide a brief 2-3 word topic or subject.
        Do not include quotes or prefixes like "Topic:". Just respond with the topic.
        
        Excerpt:
        {}
        """.format(excerpt)
        
        response = model.generate_content(prompt)
        topic = response.text.strip().replace('"', '')
        
        # Fallback if no topic is generated
        if not topic or len(topic) > 30:
            return "this material"
            
        return topic
        
    except Exception as e:
        app.logger.error(f"Topic extraction error: {str(e)}")
        return "this material"

def speak_and_wait(text, agent=None):
    """Speak the text and wait for completion"""
    speaking_event = threading.Event()
    
    def speak_with_event():
        speak_response(text, agent)
        speaking_event.set()
    
    # Start speaking in separate thread
    speaking_thread = threading.Thread(target=speak_with_event)
    speaking_thread.daemon = True
    speaking_thread.start()
    
    # Return the event that will be set when speaking is complete
    return speaking_event

def add_message(conversation_id, speaker, text, wait_for_speech=False):
    """Add message to conversation and auto-speak if enabled"""
    if conversation_id in conversations:
        msg = {
            'id': str(uuid.uuid4()),
            'speaker': speaker,
            'text': text,
            'timestamp': time.time()
        }
        conversations[conversation_id]['messages'].append(msg)
        
        # Auto-speak if enabled and not user's message
        if conversations[conversation_id].get('auto_speak', True) and speaker != "User":
            if wait_for_speech:
                # Speak and wait for completion
                speaking_event = speak_and_wait(text, speaker)
                return msg, speaking_event
            else:
                # Just speak (don't wait)
                threading.Thread(target=speak_response, args=(text, speaker)).start()
        
        return msg, None
    return None, None

def background_conversation(conversation_id, document_text):
    """Run the conversation between agents in the background"""
    try:
        conversation = conversations[conversation_id]
        conversation['active'] = True
        conversation['status'] = 'starting'
        
        # Get document topic for conversational context
        topic = get_document_topic(document_text)
        
        # Initial prompt from Agent 1
        starter = f"Let's discuss this document about {topic}. What do you think?"
        msg, speaking_event = add_message(conversation_id, "Agent 1", starter, wait_for_speech=True)
        
        # Wait for speaking to complete
        if speaking_event:
            speaking_event.wait()
        else:
            time.sleep(2)  # Natural pause if speaking failed
        
        conversation['status'] = 'active'
        
        # Main conversation loop (limit to reasonable number of turns)
        while conversation['active'] and len(conversation['messages']) < 50:
            # Skip if user is typing
            if user_typing.get(conversation_id, False):
                time.sleep(1)
                continue
                
            # Alternate between agents
            last_speaker = conversation['messages'][-1]['speaker'] if conversation['messages'] else None
            current_agent = "Agent 2" if last_speaker == "Agent 1" else "Agent 1"
            
            # Natural delay between messages (longer after user speaks)
            delay = 4 if last_speaker == "User" else 2
            time.sleep(delay)
            
            # Generate response
            response = generate_agent_response(
                current_agent,
                document_text,
                conversation['messages']
            )
            
            # Add message and wait for speaking to complete
            msg, speaking_event = add_message(conversation_id, current_agent, response, wait_for_speech=True)
            
            # Wait for speaking to complete before continuing
            if speaking_event:
                speaking_event.wait()
            
    except Exception as e:
        app.logger.error(f"Conversation thread error: {str(e)}")
    finally:
        conversation['status'] = 'ended'
        conversation['active'] = False

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start conversation"""
    try:
        # Check if file exists
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file part'})
            
        file = request.files['file']
        
        # Check if filename is empty
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'})
            
        if file and allowed_file(file.filename):
            # Generate unique conversation ID
            conversation_id = str(uuid.uuid4())
            session['conversation_id'] = conversation_id
            
            # Save file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Extract text from file
            document_text = extract_text_from_file(file)
            
            # Initialize conversation
            conversations[conversation_id] = {
                'id': conversation_id,
                'filename': filename,
                'document_text': document_text,
                'messages': [],
                'active': False,
                'status': 'ready',
                'auto_speak': True
            }
            
            # Start conversation thread
            thread = threading.Thread(
                target=background_conversation,
                args=(conversation_id, document_text)
            )
            thread.daemon = True
            thread.start()
            conversation_threads[conversation_id] = thread
            
            return jsonify({'success': True, 'conversation_id': conversation_id})
        else:
            return jsonify({'success': False, 'error': 'Invalid file type'})
            
    except Exception as e:
        app.logger.error(f"Upload error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/messages')
def get_messages():
    """Get messages for the current conversation"""
    conversation_id = session.get('conversation_id')
    
    if not conversation_id or conversation_id not in conversations:
        return jsonify({
            'status': 'none',
            'messages': []
        })
        
    conversation = conversations[conversation_id]
    
    return jsonify({
        'status': conversation['status'],
        'messages': conversation['messages']
    })

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle user questions"""
    conversation_id = session.get('conversation_id')
    
    if not conversation_id or conversation_id not in conversations:
        return jsonify({'success': False, 'error': 'No active conversation'})
        
    data = request.get_json()
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'success': False, 'error': 'Empty question'})
        
    # Add user message
    add_message(conversation_id, "User", question)
    
    # Flag that user is no longer typing
    user_typing[conversation_id] = False
    
    return jsonify({'success': True})

@app.route('/speak', methods=['POST'])
def speak_text():
    """Handle TTS requests"""
    data = request.get_json()
    text = data.get('text', '')
    agent = data.get('agent', None)
    
    if text:
        threading.Thread(target=speak_response, args=(text, agent)).start()
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'No text provided'})

@app.route('/typing', methods=['POST'])
def set_typing_status():
    """Set user typing status"""
    conversation_id = session.get('conversation_id')
    
    if not conversation_id:
        return jsonify({'success': False})
        
    data = request.get_json()
    is_typing = data.get('typing', False)
    
    user_typing[conversation_id] = is_typing
    
    return jsonify({'success': True})

@app.route('/toggle_auto_speak', methods=['POST'])
def toggle_auto_speak():
    """Toggle auto-speak setting"""
    conversation_id = session.get('conversation_id')
    
    if not conversation_id or conversation_id not in conversations:
        return jsonify({'success': False, 'error': 'No active conversation'})
        
    data = request.get_json()
    enabled = data.get('enabled', True)
    
    conversations[conversation_id]['auto_speak'] = enabled
    
    return jsonify({'success': True})

@app.route('/end_conversation', methods=['POST'])
def end_conversation():
    """End the current conversation"""
    conversation_id = session.get('conversation_id')
    
    if conversation_id and conversation_id in conversations:
        conversations[conversation_id]['active'] = False
        conversations[conversation_id]['status'] = 'ended'
        
    return jsonify({'success': True})

# Clean up expired conversations
def cleanup_expired_conversations():
    current_time = time.time()
    expired_ids = []
    
    for conv_id, conv in list(conversations.items()):
        # Remove conversations older than 1 hour
        if conv['messages'] and current_time - conv['messages'][-1]['timestamp'] > 3600:
            expired_ids.append(conv_id)
            
    for conv_id in expired_ids:
        if conv_id in conversations:
            del conversations[conv_id]
        if conv_id in conversation_threads:
            del conversation_threads[conv_id]
        if conv_id in user_typing:
            del user_typing[conv_id]

# Start cleanup thread
def cleanup_thread_function():
    while True:
        time.sleep(300)  # Check every 5 minutes
        cleanup_expired_conversations()

# Start background cleanup thread
cleanup_thread = threading.Thread(target=cleanup_thread_function)
cleanup_thread.daemon = True
cleanup_thread.start()

if __name__ == '__main__':
    # Ensure the uploads directory exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.run(debug=True, threaded=True)