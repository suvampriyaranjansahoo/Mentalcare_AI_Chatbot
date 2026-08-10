import os
import sys

from flask import Flask, render_template, request, jsonify, session

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import get_pipeline

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

pipeline = None  # lazy-loaded on first request to keep app startup fast


def _get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = get_pipeline()
    return pipeline


@app.route("/")
def index():
    if "session_id" not in session:
        import uuid
        session["session_id"] = str(uuid.uuid4())
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data_input = request.get_json(silent=True)
    if data_input is None:
        return jsonify({"error": "Malformed JSON body."}), 400

    user_input = data_input.get("message", "")
    session_id = session.get("session_id", "anonymous")

    result = _get_pipeline().handle_message(user_input, session_id=session_id)
    return jsonify({"response": result["response"]})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=debug_mode, port=port)
