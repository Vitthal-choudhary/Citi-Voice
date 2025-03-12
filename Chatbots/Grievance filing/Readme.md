# Legal_voice_bot
![LegalVoiceBot UI](assets/bot-ui.jpg)


Legal Voice Bot is an AI-powered legal assistant chatbot that supports both voice and text input. It processes legal queries, summarizes legal documents, and provides responses in both text and voice using Mistral AI’s API.  

## Features  
- **Voice Input & Output**: Accepts queries via voice and responds in speech.  
- **Text-Based Interaction**: Users can also input text queries and receive text responses.  
- **Legal Document Analysis**: Summarizes uploaded legal document images (if integrated with OCR).  
- **Uses Mistral AI API**: Provides legal assistance through an AI model.  
- **Real-time Communication**: Uses Flask-SocketIO for real-time interaction.  

## Installation  

### Prerequisites  
- Python 3.x  
- Virtual environment (optional but recommended)  

### Setup  

1. Clone this repository:  
   ```bash
   git clone https://github.com/yourusername/LegalVoiceBot.git
   cd LegalVoiceBot
   lone this repository:  
   ```bash
   git clone https://github.com/yourusername/LegalVoiceBot.git
   cd LegalVoiceBot
2.Create and activate a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

3.Install dependencies:
 
pip install -r requirements.txt
Create a .env file in the project root and add your API key:
LEGAL_API_KEY=your_actual_api_key_here

4.Run the application:
python app.py

5.Access the chatbot via:
http://localhost:5000
