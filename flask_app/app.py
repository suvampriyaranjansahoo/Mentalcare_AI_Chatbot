from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import pandas as pd
import random
from difflib import get_close_matches

app = Flask(__name__)

# Load CSV
data = pd.read_csv(r"C:\Users\suvha\Documents\PROJECTS\MENTALCARE_AI\data\expanded_data.csv")

def get_response_from_csv(user_input):
    choices = data['user_input'].str.lower().tolist()
    match = get_close_matches(user_input.lower(), choices, n=1, cutoff=0.5)
    if match:
        return data[data['user_input'].str.lower() == match[0]]['bot_response'].values[0]
    return None

# Hugging Face sentiment analyzer
sentiment_analyzer = pipeline("sentiment-analysis")
mood_memory = []

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

def analyze_sentiment(text):
    return sentiment_analyzer(text)[0]['label']

def generate_response(user_input):
    csv_response = get_response_from_csv(user_input)
    if csv_response:
        return csv_response

    sentiment = analyze_sentiment(user_input)
    mood_memory.append(sentiment)
    if len(mood_memory) > 10:
        mood_memory.pop(0)

    recent_negative = mood_memory.count("NEGATIVE") >= 1

    # Multi-step motivational flow
    if recent_negative:
        motivational_chain = random.choices(comforting_responses, k=2) + random.choices(motivational_quotes, k=2)
        return " ".join(motivational_chain)
    else:
        return random.choice(positive_responses)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data_input = request.get_json()
    user_input = data_input.get("message", "")
    if not user_input.strip():
        return jsonify({"response": "Please type something!"})
    ai_response = generate_response(user_input)
    return jsonify({"response": ai_response})

if __name__ == "__main__":
    app.run(debug=True)
