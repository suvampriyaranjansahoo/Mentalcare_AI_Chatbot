# app.py
import streamlit as st
from transformers import pipeline
from gtts import gTTS
from io import BytesIO
import pandas as pd
from difflib import get_close_matches
import random
import time
import speech_recognition as sr

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="MENTALCARE AI 💬", page_icon="💬", layout="wide")

# -----------------------------
# Session State Initialization
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "mood_memory" not in st.session_state:
    st.session_state.mood_memory = []

# -----------------------------
# Load CSV FAQ Data
# -----------------------------
data = pd.read_csv(r"C:\Users\suvha\Documents\PROJECTS\MENTALCARE_AI\data\expanded_data.csv")  # your CSV with user_input & bot_response

def get_response_from_csv(user_input):
    choices = data['user_input'].tolist()
    match = get_close_matches(user_input, choices, n=1, cutoff=0.5)
    if match:
        return data[data['user_input'] == match[0]]['bot_response'].values[0]
    return None

# -----------------------------
# Hugging Face AI Model
# -----------------------------
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="gpt2")  # replace with your preferred model

hf_model = load_model()

# -----------------------------
# Sentiment Analyzer
# -----------------------------
sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    return sentiment_analyzer(text)[0]['label']

# -----------------------------
# Voice Setup
# -----------------------------
recognizer = sr.Recognizer()

def speak(text):
    tts = gTTS(text=text, lang='en')
    audio_file = BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)
    st.audio(audio_file, format='audio/mp3')

def listen():
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except:
        return ""

# -----------------------------
# Generate AI Response
# -----------------------------
motivational_quotes = [
    "You are stronger than you think. 🌟",
    "Every day is a new beginning. 🌱",
    "Believe in yourself. 💪 You’ve got this!",
    "Small steps every day lead to big changes. ✨",
    "Even storms pass. Keep going. 💛"
]

comforting_responses = [
    "I hear you. 💛 Can you tell me a little more?",
    "It’s okay to feel this way. I’m here with you.",
    "That sounds tough. 🌧️ Let's take a deep breath together.",
    "Remember, you’re not alone. 💌"
]

positive_responses = [
    "That’s wonderful! Keep up the positive energy! ✨",
    "Great! It feels good to hear that. 🌈",
    "Awesome! You’re doing great! 💪"
]

def generate_response(user_input):
    # 1️⃣ Check CSV FAQ
    csv_response = get_response_from_csv(user_input)
    if csv_response:
        return csv_response

    # 2️⃣ Sentiment & mood memory
    sentiment = analyze_sentiment(user_input)
    st.session_state.mood_memory.append(sentiment)
    if len(st.session_state.mood_memory) > 10:
        st.session_state.mood_memory.pop(0)
    recent_negative = st.session_state.mood_memory.count("NEGATIVE") >= 1

    # 3️⃣ Hugging Face AI fallback
    ai_response = hf_model(user_input, max_length=100, do_sample=True)[0]['generated_text']
    ai_response = ai_response.strip()

    # 4️⃣ Decide final response
    if recent_negative:
        return random.choice(comforting_responses) + " " + random.choice(motivational_quotes)
    elif ai_response:
        return ai_response
    else:
        return random.choice(positive_responses)

# -----------------------------
# CSS Styling
# -----------------------------
st.markdown("""
<style>
.user-bubble {background-color:#87ceeb;color:white;padding:10px;border-radius:15px;margin:5px;display:inline-block;max-width:80%;}
.ai-bubble {background-color:#e6e6fa;color:black;padding:10px;border-radius:15px;margin:5px;display:inline-block;max-width:80%;}
.chat-box {max-height:500px; overflow-y:auto; padding:10px; border:1px solid #ccc; border-radius:15px; background:white;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("MENTALCARE AI 💬")
st.subheader("Your empathetic AI companion")

chat_box = st.container()
input_col, button_col = st.columns([4,1])

use_voice = st.sidebar.checkbox("Voice Input")

with input_col:
    if not use_voice:
        user_input = st.text_input("Type your message:", key="user_input")
    else:
        if st.button("🎤 Speak"):
            user_input = listen()
            st.text_input("You said:", value=user_input, key="voice_input")
with button_col:
    send_button = st.button("Send")

def display_chat():
    chat_box.empty()
    with chat_box:
        for speaker, message in st.session_state.history:
            if speaker == "You":
                st.markdown(f"<div class='user-bubble'><b>{speaker}:</b> {message}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='ai-bubble'><b>{speaker}:</b> {message}</div>", unsafe_allow_html=True)

# -----------------------------
# Handle Sending
# -----------------------------
if send_button and user_input:
    st.session_state.history.append(("You", user_input))
    display_chat()
    placeholder = st.empty()
    placeholder.markdown("<i>MENTALCARE AI is typing... ⏳</i>", unsafe_allow_html=True)
    time.sleep(1.5)
    placeholder.empty()
    ai_response = generate_response(user_input)
    st.session_state.history.append(("MENTALCARE AI", ai_response))
    display_chat()
    speak(ai_response)

st.markdown("---")
st.markdown("💡 Tip: Keep chatting! MENTALCARE AI provides multi-step motivational support.")
