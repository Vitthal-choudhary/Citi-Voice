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
def get_farming_response(user_input):
    # Comprehensive prompt for farming assistance
    system_prompt = '''You are an experienced agricultural expert and Smart Farming Assistant. Your purpose is to help farmers with their agricultural queries, provide crop guidance, and share market insights.

Follow these guidelines when responding to farming queries:

1. CROP DISEASE DETECTION:
   - When farmers describe symptoms in their crops, identify possible diseases or pest issues
   - Recommend appropriate treatments, both organic and chemical options
   - Suggest preventive measures for future plantings
   - Mention any environmental factors that might be causing the issue

2. WEATHER-BASED GUIDANCE:
   - Provide irrigation recommendations based on weather conditions
   - Suggest optimal harvesting times considering weather forecasts
   - Recommend crop protection measures for extreme weather
   - Advise on soil management practices for current weather conditions

3. MARKET INSIGHTS:
   - Inform about current market trends for agricultural products
   - Suggest optimal timing for selling produce
   - Mention any value-addition possibilities to increase profit margins
   - Note any upcoming seasonal price fluctuations

4. FARMING RECOMMENDATIONS:
   - Provide specific fertilizer, seed, and cultivation technique recommendations
   - Suggest sustainable and organic farming methods when applicable
   - Recommend crop rotations and companion planting for soil health
   - Offer guidance on modern agricultural technologies and their application

5. GOVERNMENT SCHEMES:
   - Inform about relevant subsidies, grants, and government programs for farmers
   - Mention application deadlines and eligibility criteria
   - Provide information about agricultural loans and financial assistance
   - Share details about training programs and agricultural extension services

Always tailor your responses to the farmer's specific geographical region and crop type when they mention it. Use simple, clear language and practical advice that can be implemented with resources typically available to farmers.

If the farmer doesn't provide enough information, ask follow-up questions to better understand their specific situation, especially regarding crop type, region, current agricultural practices, and exact symptoms or issues they're facing.'''

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
        full_response = get_farming_response(user_input)
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
        time.sleep(0.4)

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

    # HTML template with customized UI for farming assistance
    html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Farming Assistant</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        body { background-color: #f8f9fa; font-family: 'Arial', sans-serif; }
        .chat-container { max-width: 800px; margin: 0 auto; padding: 20px; background-color: white; border-radius: 15px; box-shadow: 0 0 20px rgba(0, 0, 0, 0.1); display: flex; flex-direction: column; height: 85vh; }
        .chat-header { background: linear-gradient(135deg, #3CB371, #228B22); color: white; padding: 15px; border-radius: 10px 10px 0 0; font-size: 1.5rem; text-align: center; margin-bottom: 15px; }
        .chat-messages { flex-grow: 1; overflow-y: auto; padding: 10px; margin-bottom: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
        .message { margin-bottom: 15px; padding: 10px 15px; border-radius: 10px; max-width: 75%; word-wrap: break-word; }
        .user-message { background-color: #e8f5e9; color: #1b5e20; margin-left: auto; border-bottom-right-radius: 0; }
        .bot-message { background-color: #f1f8e9; color: #33691e; margin-right: auto; border-bottom-left-radius: 0; }
        .system-message { background-color: #f5f5f5; color: #757575; text-align: center; margin: 10px auto; font-style: italic; max-width: 50%; }
        .thinking { display: flex; align-items: center; margin-bottom: 15px; }
        .thinking-dots { display: flex; margin-left: 10px; }
        .thinking-dot { height: 10px; width: 10px; margin: 0 3px; background-color: #388E3C; border-radius: 50%; animation: pulse 1.5s infinite; }
        .thinking-dot:nth-child(2) { animation-delay: 0.2s; }
        .thinking-dot:nth-child(3) { animation-delay: 0.4s; }
        .input-container { display: flex; margin-top: 10px; }
        .message-input { flex-grow: 1; padding: 12px; border: 1px solid #ccc; border-radius: 25px; font-size: 1rem; outline: none; transition: border 0.3s; }
        .message-input:focus { border-color: #4CAF50; }
        .send-button, .voice-button { padding: 10px 15px; margin-left: 10px; border: none; border-radius: 25px; cursor: pointer; outline: none; transition: background-color 0.3s; }
        .send-button { background-color: #4CAF50; color: white; }
        .voice-button { background-color: #388E3C; color: white; }
        .send-button:hover { background-color: #3d8b40; }
        .voice-button:hover { background-color: #2e7d32; }
        .voice-button.listening { animation: pulse 1.5s infinite; background-color: #f44336; }
        @keyframes pulse { 0% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.1); opacity: 0.7; } 100% { transform: scale(1); opacity: 1; } }
        .status-bar { margin-top: 15px; padding: 5px 10px; background-color: #f5f5f5; border-radius: 5px; font-size: 0.85rem; color: #757575; }
        @media (max-width: 576px) { .chat-container { height: 95vh; border-radius: 0; box-shadow: none; } .message { max-width: 85%; } }
        .debug-panel { padding: 10px; background-color: #f0f0f0; border-radius: 5px; margin-top: 10px; font-family: monospace; font-size: 0.8rem; max-height: 100px; overflow-y: auto; display: none; }
        .mic-status { background-color: #f5f5f5; padding: 10px; margin-top: 10px; border-radius: 5px; }
        .quick-buttons { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; margin-bottom: 10px; }
        .quick-button { background-color: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; border-radius: 15px; padding: 8px 12px; font-size: 0.9rem; cursor: pointer; transition: all 0.2s; }
        .quick-button:hover { background-color: #c8e6c9; }
        .features-container { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px; }
        .feature-card { background-color: #f1f8e9; border-left: 4px solid #4CAF50; padding: 8px 12px; border-radius: 5px; flex: 1 1 calc(33% - 10px); min-width: 200px; cursor: pointer; transition: all 0.2s; }
        .feature-card:hover { background-color: #e8f5e9; transform: translateY(-2px); }
        .feature-card i { color: #388E3C; margin-right: 8px; }
        .weather-widget { background-color: #e1f5fe; padding: 10px; border-radius: 5px; margin-bottom: 15px; display: flex; align-items: center; justify-content: space-between; }
        .weather-widget i { font-size: 2rem; color: #0288d1; }
    </style>
</head>
<body>
    <div class="container mt-4">
        <div class="chat-container">
            <div class="chat-header">
                <i class="fas fa-leaf me-2"></i> Smart Farming Assistant
            </div>
            <div class="features-container">
                <div class="feature-card" onclick="sendMessage('Help me identify a crop disease')">
                    <i class="fas fa-bug"></i> Crop Disease Detection
                </div>
                <div class="feature-card" onclick="sendMessage('Weather-based farming tips')">
                    <i class="fas fa-cloud-sun-rain"></i> Weather Guidance
                </div>
                <div class="feature-card" onclick="sendMessage('Current market prices for crops')">
                    <i class="fas fa-chart-line"></i> Market Price Alerts
                </div>
                <div class="feature-card" onclick="sendMessage('Government schemes for farmers')">
                    <i class="fas fa-university"></i> Govt Schemes
                </div>
                <div class="feature-card" onclick="sendMessage('Best farming practices')">
                    <i class="fas fa-seedling"></i> Farming Tips
                </div>
                <div class="feature-card" onclick="sendMessage('Soil health management')">
                    <i class="fas fa-mountain"></i> Soil Health
                </div>
            </div>
            <div class="chat-messages" id="chatMessages">
                <div class="message bot-message">
                    Namaste! I'm your Smart Farming Assistant. I can help you with crop disease detection, weather-based guidance, market prices, government schemes, and best farming practices. How can I assist you with your farm today?
                </div>
            </div>
            <div class="quick-buttons" id="quickButtons">
                <button class="quick-button">My tomato plants have yellow leaves</button>
                <button class="quick-button">When should I harvest wheat?</button>
                <button class="quick-button">What's the best fertilizer for rice?</button>
                <button class="quick-button">Any subsidies for drip irrigation?</button>
            </div>
            <div class="input-container">
                <input type="text" class="message-input" id="messageInput" placeholder="Type your farming question here...">
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

            // Make feature cards clickable
            document.querySelectorAll('.feature-card').forEach(card => {
                card.addEventListener('click', function() {
                    const topic = this.textContent.trim();
                    debugLog("Feature card clicked: " + topic);
                    // You can customize the messages for each feature
                });
            });

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

            // Function to be accessible from HTML
            window.sendMessage = sendMessage;

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
                    voiceButton.classList.remove('listening');
                    isThinking = false;
                    isListening = false;
                });

                // Audio playback
                function playAudio(audioData) {
                    return new Promise((resolve, reject) => {
                        try {
                            const audio = new Audio('data:audio/mp3;base64,' + audioData);
                            currentAudio = audio;

                            audio.onended = function() {
                                debugLog("Audio playback completed");
                                currentAudio = null;
                                resolve();
                            };

                            audio.onerror = function(e) {
                                debugLog("Audio playback error: " + e);
                                currentAudio = null;
                                reject(e);
                            };

                            debugLog("Starting audio playback");
                            audio.play().catch(error => {
                                debugLog("Audio play error: " + error);
                                reject(error);
                            });
                        } catch (e) {
                            debugLog("Error creating audio: " + e);
                            reject(e);
                        }
                    });
                }

                async function processAudioQueue() {
                    if (isPlayingAudio || audioQueue.length === 0) return;

                    isPlayingAudio = true;
                    debugLog(`Processing audio queue (${audioQueue.length} items)`);

                    try {
                        while (audioQueue.length > 0) {
                            const audioData = audioQueue.shift();
                            await playAudio(audioData);
                        }
                    } catch (error) {
                        debugLog("Error in audio queue processing: " + error);
                    } finally {
                        isPlayingAudio = false;
                    }
                }

                socket.on('play_audio', function(data) {
                    debugLog("Received audio data");
                    audioQueue.push(data.audio_data);

                    if (!isPlayingAudio) {
                        processAudioQueue();
                    }
                });

                socket.on('stop_audio', function() {
                    debugLog("Received stop audio command");

                    // Clear the audio queue
                    audioQueue = [];

                    // Stop current audio if playing
                    if (currentAudio) {
                        debugLog("Stopping current audio");
                        currentAudio.pause();
                        currentAudio.currentTime = 0;
                        currentAudio = null;
                    }

                    isPlayingAudio = false;
                });

                // Test TTS on load (uncomment to enable)
                // fetch('/test_tts').then(response => {
                //     debugLog("TTS test initiated");
                // });
            });
        </script>
    </body>
    </html>
    '''

    # Write the HTML template
    with open('templates/index.html', 'w') as f:
        f.write(html_content)

    print("Template created successfully. Starting server...")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)