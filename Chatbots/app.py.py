from flask import Flask, render_template, jsonify
import os
import tempfile
import base64
import time
import threading
import queue
import re
import speech_recognition as sr
from gtts import gTTS
from flask_socketio import SocketIO
import google.generativeai as genai

app = Flask(__name__)
socketio = SocketIO(app)

# Gemini API Configuration
GEMINI_API_KEY = "AIzaSyCliqGdNtcYJY0f638LOfext7L-Hy4kxXw"  # Replace with your actual Gemini API key
GEMINI_MODEL = "gemini-1.5-pro"  # Adjust according to available models

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Queue for speech processing; items will be tuples of (token, sentence)
tts_queue = queue.Queue()

# Global token to keep track of the current response
current_token = 0


# Function to Fetch Response from Gemini API
def get_grievance_response(user_input):
    # Comprehensive prompt for grievance handling
    system_prompt = '''You are an empathetic and professional Grievance Filing Assistant. Your purpose is to help users document their grievances accurately and thoroughly for official filing.

Follow these guidelines when processing grievance reports:

1. INFORMATION COLLECTION:
   - Identify the nature of the grievance (workplace, consumer, healthcare, housing, etc.)
   - Gather essential details: who, what, when, where, why, and how
   - Determine if there were any witnesses or supporting evidence
   - Note any prior attempts to resolve the issue

2. REPORT STRUCTURE:
   - Begin with a clear summary of the incident/issue
   - Include chronological details with specific dates and times
   - Document names and titles of all relevant parties
   - Note any applicable policies, regulations, or laws that were violated
   - Detail the impact (emotional, financial, physical, etc.) on the complainant
   - Include any attempted resolutions and their outcomes
   - End with the complainant's desired resolution

3. TONE AND APPROACH:
   - Maintain a professional, factual tone
   - Be empathetic while remaining objective
   - Use clear, specific language without emotional qualifiers
   - Avoid making legal determinations or promising specific outcomes
   - Highlight key facts that support the grievance claim

4. NEXT STEPS:
   - Suggest documentation or evidence the user should gather
   - Explain the typical timeline for processing similar grievances
   - Outline what to expect in the grievance process
   - Recommend appropriate follow-up actions

If the user doesn't provide enough information, ask follow-up questions to ensure the report is complete. Focus especially on specific details, dates, locations, and the names/positions of people involved.

Present the final grievance report in a structured format that would be suitable for official submission, while suggesting any additional information that might strengthen their case.'''

    try:
        # Initialize the Gemini model
        model = genai.GenerativeModel(GEMINI_MODEL)

        # Create a chat session
        chat = model.start_chat(history=[])

        # Send the system prompt and user input to the model
        response = chat.send_message(f"{system_prompt}\n\nUser input: {user_input}")

        return response.text
    except Exception as e:
        return f"Error: Unable to fetch response. {str(e)}"


def stream_response(user_input, token):
    global current_token
    if token != current_token:
        return  # Exit if this response is no longer current

    socketio.emit('thinking_status', {'status': True})

    try:
        full_response = get_grievance_response(user_input)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', full_response) if s.strip()]

        print(f"Processing response with {len(sentences)} sentences")

        accumulated_text = ""
        for sentence in sentences:
            if token != current_token:
                print("Token changed, stopping response streaming")
                break  # Stop processing if token has changed

            accumulated_text += (" " if accumulated_text else "") + sentence
            socketio.emit('response_stream', {'text': accumulated_text, 'is_final': False})

            print(f"Adding sentence to TTS queue: '{sentence}'")
            tts_queue.put((token, sentence))

            socketio.sleep(0.3)  # Reduced delay for quicker response

        if token == current_token:
            socketio.emit('response_stream', {'text': accumulated_text, 'is_final': True})

    except Exception as e:
        print(f"Error in stream_response: {str(e)}")
        socketio.emit('error_message', {'message': f'Error generating response: {str(e)}'})
    finally:
        if token == current_token:
            socketio.emit('thinking_status', {'status': False})


# Function to handle speech recognition with improved error handling
def recognize_speech():
    global current_token, tts_queue
    recognizer = sr.Recognizer()

    try:
        print("Starting speech recognition")
        socketio.emit('listening_status', {'status': True})

        with sr.Microphone() as source:
            print("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Listening for speech...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            print("Audio captured")

        socketio.emit('listening_status', {'status': False})

        print("Recognizing speech...")
        user_input = recognizer.recognize_google(audio)
        print(f"Speech recognized: '{user_input}'")

        socketio.emit('speech_recognized', {'text': user_input})

        # **Cancel previous processing before starting a new one**
        current_token += 1
        print(f"Cancelling previous processing, new token: {current_token}")

        with tts_queue.mutex:
            queue_size = len(tts_queue.queue)
            tts_queue.queue.clear()
            print(f"Cleared TTS queue ({queue_size} items removed)")

        socketio.emit('stop_audio')
        print("Sent stop_audio signal to client")

        # **Start response streaming for speech input**
        thread = threading.Thread(target=stream_response, args=(user_input, current_token))
        thread.daemon = True
        thread.start()

    except sr.UnknownValueError:
        print("Speech recognition could not understand audio")
        socketio.emit('error_message', {'message': "Sorry, I couldn't understand. Please try again."})
    except sr.RequestError as e:
        print(f"Speech recognition service error: {e}")
        socketio.emit('error_message', {'message': f"Error in speech recognition service: {str(e)}"})
    except Exception as e:
        print(f"Speech recognition error: {e}")
        import traceback
        traceback.print_exc()
        socketio.emit('error_message', {'message': f"An error occurred during speech recognition: {str(e)}"})


# Function to generate speech audio optimized for sentence-by-sentence processing
def text_to_speech(text):
    try:
        if not text or len(text.strip()) < 2:
            print("Empty text received, skipping TTS")
            return

        print(f"Converting to speech: '{text}'")

        # Create temp file in a way that ensures it's accessible
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_filename = temp_file.name
            print(f"Temp file created: {temp_filename}")

        # Generate and save audio
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(temp_filename)
        print("Audio saved to temp file")

        # Add a small delay to ensure file is fully written
        time.sleep(0.2)

        # Read the audio file
        with open(temp_filename, 'rb') as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
            print(f"Audio file read, size: {len(audio_data)} chars")

        # Clean up
        try:
            os.unlink(temp_filename)
            print("Temp file deleted")
        except Exception as e:
            print(f"Warning: Could not delete temp file: {str(e)}")

        # Send the audio data to the client
        print("Sending audio data to client")
        socketio.emit('play_audio', {'audio_data': audio_data})
        print("Audio data sent")

    except Exception as e:
        print(f"TTS Error: {str(e)}")
        socketio.emit('error_message', {'message': f'Error generating speech: {str(e)}'})


# TTS worker thread: processes items from the queue if they belong to the current message
def tts_worker():
    global current_token
    print("TTS worker thread started")
    while True:
        try:
            print("Waiting for sentence in TTS queue...")
            token, text = tts_queue.get()
            print(f"Got sentence from queue (token {token}): '{text}'")

            if token != current_token:
                print(f"Skipping TTS for outdated token {token} (current token is {current_token})")
                tts_queue.task_done()
                continue

            text_to_speech(text)
            tts_queue.task_done()
        except Exception as e:
            print(f"Error in TTS worker: {str(e)}")
            try:
                tts_queue.task_done()
            except:
                pass


# Start TTS worker thread
print("Starting TTS worker thread")
tts_thread = threading.Thread(target=tts_worker)
tts_thread.daemon = True
tts_thread.start()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/test_tts')
def test_tts():
    print("Testing TTS functionality")
    text_to_speech("This is a test of the text to speech system.")
    return "Testing TTS functionality. Check console for logs."


@app.route('/test_mic')
def test_mic():
    print("Testing microphone setup")
    try:
        mic_list = sr.Microphone.list_microphone_names()
        return jsonify({
            'status': 'success',
            'microphones': mic_list,
            'count': len(mic_list)
        })
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': error_trace
        })


@socketio.on('send_message')
def handle_message(data):
    global current_token, tts_queue
    user_input = data['message'].strip()
    if not user_input:
        return

    print(f"Received new message: '{user_input}'")

    # Cancel previous processing
    current_token += 1
    print(f"Cancelling previous processing, new token: {current_token}")

    with tts_queue.mutex:
        queue_size = len(tts_queue.queue)
        tts_queue.queue.clear()
        print(f"Cleared TTS queue ({queue_size} items removed)")

    # Notify client to stop audio immediately
    socketio.emit('stop_audio')
    print("Sent stop_audio signal to client")

    # Start new processing thread
    print(f"Starting new processing thread for token {current_token}")
    thread = threading.Thread(target=stream_response, args=(user_input, current_token))
    thread.daemon = True
    thread.start()


@socketio.on('start_voice_input')
def handle_voice_input():
    print("Starting voice input")
    thread = threading.Thread(target=recognize_speech)
    thread.daemon = True
    thread.start()


@socketio.on('connect')
def handle_connect():
    print("Client connected")


@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")


if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)

    # HTML template with customized UI for grievance filing
    html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Grievance Filing Assistant</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        body { background-color: #f8f9fa; font-family: 'Arial', sans-serif; }
        .chat-container { max-width: 800px; margin: 0 auto; padding: 20px; background-color: white; border-radius: 15px; box-shadow: 0 0 20px rgba(0, 0, 0, 0.1); display: flex; flex-direction: column; height: 85vh; }
        .chat-header { background: linear-gradient(135deg, #4285F4, #0F9D58); color: white; padding: 15px; border-radius: 10px 10px 0 0; font-size: 1.5rem; text-align: center; margin-bottom: 15px; }
        .chat-messages { flex-grow: 1; overflow-y: auto; padding: 10px; margin-bottom: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
        .message { margin-bottom: 15px; padding: 10px 15px; border-radius: 10px; max-width: 75%; word-wrap: break-word; }
        .user-message { background-color: #e8f0fe; color: #174ea6; margin-left: auto; border-bottom-right-radius: 0; }
        .bot-message { background-color: #f1f8e9; color: #33691e; margin-right: auto; border-bottom-left-radius: 0; }
        .system-message { background-color: #f5f5f5; color: #757575; text-align: center; margin: 10px auto; font-style: italic; max-width: 50%; }
        .thinking { display: flex; align-items: center; margin-bottom: 15px; }
        .thinking-dots { display: flex; margin-left: 10px; }
        .thinking-dot { height: 10px; width: 10px; margin: 0 3px; background-color: #0F9D58; border-radius: 50%; animation: pulse 1.5s infinite; }
        .thinking-dot:nth-child(2) { animation-delay: 0.2s; }
        .thinking-dot:nth-child(3) { animation-delay: 0.4s; }
        .input-container { display: flex; margin-top: 10px; }
        .message-input { flex-grow: 1; padding: 12px; border: 1px solid #ccc; border-radius: 25px; font-size: 1rem; outline: none; transition: border 0.3s; }
        .message-input:focus { border-color: #4285F4; }
        .send-button, .voice-button { padding: 10px 15px; margin-left: 10px; border: none; border-radius: 25px; cursor: pointer; outline: none; transition: background-color 0.3s; }
        .send-button { background-color: #4285F4; color: white; }
        .voice-button { background-color: #0F9D58; color: white; }
        .send-button:hover { background-color: #3367d6; }
        .voice-button:hover { background-color: #0b8043; }
        .voice-button.listening { animation: pulse 1.5s infinite; background-color: #EA4335; }
        @keyframes pulse { 0% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.1); opacity: 0.7; } 100% { transform: scale(1); opacity: 1; } }
        .status-bar { margin-top: 15px; padding: 5px 10px; background-color: #f5f5f5; border-radius: 5px; font-size: 0.85rem; color: #757575; }
        @media (max-width: 576px) { .chat-container { height: 95vh; border-radius: 0; box-shadow: none; } .message { max-width: 85%; } }
        .debug-panel { padding: 10px; background-color: #f0f0f0; border-radius: 5px; margin-top: 10px; font-family: monospace; font-size: 0.8rem; max-height: 100px; overflow-y: auto; display: none; }
        .mic-status { background-color: #f5f5f5; padding: 10px; margin-top: 10px; border-radius: 5px; }
        .quick-buttons { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; margin-bottom: 10px; }
        .quick-button { background-color: #e8f0fe; color: #4285F4; border: 1px solid #c6dafc; border-radius: 15px; padding: 8px 12px; font-size: 0.9rem; cursor: pointer; transition: all 0.2s; }
        .quick-button:hover { background-color: #c6dafc; }
    </style>
</head>
<body>
    <div class="container mt-4">
        <div class="chat-container">
            <div class="chat-header">
                <i class="fas fa-comments me-2"></i> Grievance Filing Assistant
            </div>
            <div class="chat-messages" id="chatMessages">
                <div class="message bot-message">
                    Hello! I'm your grievance filing assistant. I'll help you document your concerns professionally and thoroughly. Please describe your situation, and I'll guide you through the process. What type of grievance would you like to file today?
                </div>
            </div>
            <div class="quick-buttons" id="quickButtons">
                <button class="quick-button">File a workplace grievance</button>
                <button class="quick-button">Report a consumer complaint</button>
                <button class="quick-button">File a healthcare concern</button>
                <button class="quick-button">What information do I need?</button>
            </div>
            <div class="input-container">
                <input type="text" class="message-input" id="messageInput" placeholder="Describe your grievance or ask a question...">
                <button class="send-button" id="sendButton">
                    <i class="fas fa-paper-plane"></i>
                </button>
                <button class="voice-button" id="voiceButton">
                    <i class="fas fa-microphone"></i>
                </button>
            </div>
            <div class="status-bar" id="statusBar">Ready</div>
            <div class="debug-panel" id="debugPanel"></div>
            <div class="mic-status" id="micStatus" style="display: none;"></div>
        </div>
    </div>
    <script src="https://cdn.socket.io/4.4.1/socket.io.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const socket = io();
            const chatMessages = document.getElementById('chatMessages');
            const messageInput = document.getElementById('messageInput');
            const sendButton = document.getElementById('sendButton');
            const voiceButton = document.getElementById('voiceButton');
            const statusBar = document.getElementById('statusBar');
            const debugPanel = document.getElementById('debugPanel');
            const micStatus = document.getElementById('micStatus');
            const quickButtons = document.getElementById('quickButtons');

            // Uncomment to enable debug panel
            // debugPanel.style.display = 'block';

            let isThinking = false;
            let isListening = false;
            let currentBotMessage = null;
            let currentThinking = null;
            let audioQueue = [];
            let isPlayingAudio = false;
            let currentAudio = null;
            let audioContext = null;

            // Debug log function
            function debugLog(message) {
                console.log(message);
                const timestamp = new Date().toISOString().substr(11, 8);
                debugPanel.innerHTML += `<div>[${timestamp}] ${message}</div>`;
                debugPanel.scrollTop = debugPanel.scrollHeight;
            }

            // Initialize audio context with user interaction
            function initAudioContext() {
                if (!audioContext) {
                    try {
                        audioContext = new (window.AudioContext || window.webkitAudioContext)();
                        debugLog("Audio context initialized");
                    } catch (e) {
                        debugLog("Error initializing audio context: " + e);
                    }
                }
            }

            // Check microphone availability
            async function checkMicrophone() {
                try {
                    // Check if microphone is available
                    const devices = await navigator.mediaDevices.enumerateDevices();
                    const hasMic = devices.some(device => device.kind === 'audioinput');

                    if (!hasMic) {
                        debugLog("No microphone detected!");
                        micStatus.textContent = "No microphone detected. Voice input will not work.";
                        micStatus.style.display = "block";
                        micStatus.style.color = "#f44336";
                        voiceButton.disabled = true;
                        voiceButton.title = "No microphone available";
                        return false;
                    }

                    // Check if we have permission
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                        stream.getTracks().forEach(track => track.stop());
                        micStatus.textContent = "Microphone available and permission granted.";
                        micStatus.style.color = "#4CAF50";
                        debugLog("Microphone permission granted");
                        return true;
                    } catch (err) {
                        debugLog("Microphone permission denied: " + err);
                        micStatus.textContent = "Microphone permission denied. Voice input will not work.";
                        micStatus.style.color = "#FF9800";
                        voiceButton.title = "Microphone permission required";
                        return false;
                    }
                } catch (err) {
                    debugLog("Error checking microphone: " + err);
                    micStatus.textContent = " Cannot check microphone status. Voice input may not work.";
                    micStatus.style.color = "#FF9800";
                    return false;
                } finally {
                    micStatus.style.display = "block";
                }
            }

            // Test server-side microphone detection
            function testServerMicrophone() {
                fetch('/test_mic')
                    .then(response => response.json())
                    .then(data => {
                        if (data.status === 'success') {
                            debugLog(`Server detected ${data.count} microphones: ${data.microphones.join(', ')}`);
                            micStatus.innerHTML += `<br>Server-side: ${data.count} microphone(s) detected.`;
                        } else {
                            debugLog(`Server-side microphone error: ${data.message}`);
                            micStatus.innerHTML += `<br>Server-side error: ${data.message}`;
                        }
                    })
                    .catch(error => {
                        debugLog("Error testing server microphone: " + error);
                    });
            }

            function sendMessage(message = null) {
                const messageText = message || messageInput.value.trim();
                if (messageText && !isThinking && !isListening) {
                    initAudioContext(); // Initialize audio context with user interaction

                    const userMessageElement = document.createElement('div');
                    userMessageElement.className = 'message user-message';
                    userMessageElement.textContent = messageText;
                    chatMessages.appendChild(userMessageElement);
                    messageInput.value = '';
                    chatMessages.scrollTop = chatMessages.scrollHeight;

                    debugLog("Sending message: " + messageText);
                    socket.emit('send_message', { message: messageText });
                }
            }

            function showThinking() {
                const thinkingElement = document.createElement('div');
                thinkingElement.className = 'thinking';
                thinkingElement.innerHTML = `
                    <div class="message bot-message" style="margin-bottom: 0; padding-right: 20px;">
                        <div class="thinking-dots">
                            <div class="thinking-dot"></div>
                            <div class="thinking-dot"></div>
                            <div class="thinking-dot"></div>
                        </div>
                    </div>
                `;
                chatMessages.appendChild(thinkingElement);
                chatMessages.scrollTop = chatMessages.scrollHeight;
                return thinkingElement;
            }

            // Set up quick buttons
            document.querySelectorAll('.quick-button').forEach(button => {
                button.addEventListener('click', function() {
                    sendMessage(this.textContent);
                });
            });

            // Run microphone checks on startup
            checkMicrophone().then(micAvailable => {
                if (micAvailable) {
                    testServerMicrophone();
                }
            });

            sendButton.addEventListener('click', () => sendMessage());

            messageInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });

            voiceButton.addEventListener('click', function() {
                if (!isListening && !isThinking) {
                    initAudioContext(); // Initialize audio context with user interaction

                    // Check microphone permission again when button is clicked
                    navigator.mediaDevices.getUserMedia({ audio: true })
                        .then(stream => {
                            stream.getTracks().forEach(track => track.stop());
                            debugLog("Starting voice input");
                            socket.emit('start_voice_input');
                        })
                        .catch(err => {
                            debugLog("Microphone permission denied when activating: " + err);
                            const errorElement = document.createElement('div');
                            errorElement.className = 'message system-message';
                            errorElement.textContent = 'Microphone access denied. Please allow microphone access in your browser settings.';
                            chatMessages.appendChild(errorElement);
                            chatMessages.scrollTop = chatMessages.scrollHeight;
                        });
                }
            });

            socket.on('connect', function() {
                debugLog("Connected to server");
            });

            socket.on('disconnect', function() {
                debugLog("Disconnected from server");
            });

            socket.on('thinking_status', function(data) {
                isThinking = data.status;
                debugLog("Thinking status: " + data.status);
                if (isThinking) {
                    statusBar.textContent = 'Thinking...';
                    currentThinking = showThinking();
                } else {
                    statusBar.textContent = 'Ready';
                }
            });

            socket.on('listening_status', function(data) {
                isListening = data.status;
                debugLog("Listening status: " + data.status);
                if (isListening) {
                    voiceButton.classList.add('listening');
                    const listeningElement = document.createElement('div');
                    listeningElement.className = 'message system-message';
                    listeningElement.id = 'listeningMessage';
                    listeningElement.textContent = 'Listening...';
                    chatMessages.appendChild(listeningElement);
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                    statusBar.textContent = 'Listening...';
                } else {
                    voiceButton.classList.remove('listening');
                    const listeningElement = document.getElementById('listeningMessage');
                    if (listeningElement) { listeningElement.remove(); }
                    statusBar.textContent = 'Ready';
                }
            });

            socket.on('speech_recognized', function(data) {
                debugLog("Speech recognized: " + data.text);
                const userMessageElement = document.createElement('div');
                userMessageElement.className = 'message user-message';
                userMessageElement.textContent = data.text;
                chatMessages.appendChild(userMessageElement);
                messageInput.value = data.text;
                chatMessages.scrollTop = chatMessages.scrollHeight;
            });

            socket.on('response_stream', function(data) {
                debugLog("Response stream: " + (data.is_final ? "final" : "partial"));
                if (currentThinking) {
                    currentThinking.remove();
                    currentThinking = null;
                }
                if (!currentBotMessage) {
                    currentBotMessage = document.createElement('div');
                    currentBotMessage.className = 'message bot-message';
                    chatMessages.appendChild(currentBotMessage);
                }
                currentBotMessage.textContent = data.text;
                chatMessages.scrollTop = chatMessages.scrollHeight;
                if (data.is_final) { 
                    debugLog("Response complete");
                    currentBotMessage = null; 
                }
            });

            socket.on('error_message', function(data) {
                debugLog("Error: " + data.message);
                const errorElement = document.createElement('div');
                errorElement.className = 'message system-message';
                errorElement.textContent = data.message;
                chatMessages.appendChild(errorElement);
                chatMessages.scrollTop = chatMessages.scrollHeight;
                statusBar.textContent = 'Ready';
            });

            // Audio playback handling with improved logging
            socket.on('play_audio', function(data) {
                debugLog("Received audio data: " + data.audio_data.substring(0, 20) + "... (" + data.audio_data.length + " chars)");

                const audioSrc = 'data:audio/mp3;base64,' + data.audio_data;
                audioQueue.push(audioSrc);

                debugLog("Added to audio queue. Queue length: " + audioQueue.length);

                if (!isPlayingAudio) {
                    playNextInQueue();
                }
            });

            // Stop audio event handler: clears the queue and stops current playback
            socket.on('stop_audio', function() {
                debugLog("Received stop_audio signal");
                const queueLength = audioQueue.length;
                audioQueue = [];

                if (currentAudio) {
                    debugLog("Stopping current audio playback");
                    currentAudio.pause();
                    currentAudio.currentTime = 0;
                    currentAudio = null;
                }

                debugLog(`Audio stopped, cleared queue (${queueLength} items)`);
                isPlayingAudio = false;
            });

            function playNextInQueue() {
                if (audioQueue.length === 0) {
                    debugLog("Audio queue empty, playback complete");
                    isPlayingAudio = false;
                    return;
                }

                isPlayingAudio = true;
                const nextAudioSrc = audioQueue.shift();

                debugLog("Playing next audio in queue, remaining: " + audioQueue.length);

                try {
                        currentAudio = new Audio(nextAudioSrc);

                        currentAudio.onended = function() {
                            debugLog("Audio playback ended");
                            currentAudio = null;
                            // Small delay before playing next segment for more natural speech rhythm
                            setTimeout(playNextInQueue, 150);
                        };

                        currentAudio.onerror = function(e) {
                            debugLog("Audio playback error: " + e);
                            currentAudio = null;
                            playNextInQueue();
                        };

                        currentAudio.play().catch(function(error) {
                            debugLog("Error playing audio: " + error);
                            currentAudio = null;
                            playNextInQueue();
                        });
                } catch (error) {
                    debugLog("Error creating audio element: " + error);
                    setTimeout(playNextInQueue, 100);
                }
            }

            // Test TTS function
            function testTTS() {
                fetch('/test_tts')
                    .then(response => response.text())
                    .then(data => {
                        debugLog("TTS test result: " + data);
                    })
                    .catch(error => {
                        debugLog("Error testing TTS: " + error);
                    });
            }

            // Optional: Run a TTS test when page loads
            // setTimeout(testTTS, 2000);
        });
    </script>
</body>
</html>
'''

    # Write the HTML template to a file
    with open('templates/index.html', 'w') as f:
        f.write(html_content)

    print("Starting the application")
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)