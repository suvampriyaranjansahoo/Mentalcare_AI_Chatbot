"""Streamlit deployment entry point for Streamlit Community Cloud."""
from __future__ import annotations
import os
from uuid import uuid4
import streamlit as st
from app.config import Settings
from app.database import ConversationRepository
from app.nlp.factory import build_emotion_classifier
from app.safety import SafetyService
from app.services import ChatService
from app.llm import OllamaClient

@st.cache_resource
def get_service() -> ChatService:
    if "FLASK_SECRET_KEY" in st.secrets: os.environ["FLASK_SECRET_KEY"] = st.secrets["FLASK_SECRET_KEY"]
    # Streamlit Community Cloud cannot reach a locally running Ollama server.
    # It uses the app's safe template responses unless explicitly configured otherwise.
    os.environ.setdefault("LLM_PROVIDER", "disabled")
    settings = Settings.from_env()
    llm = OllamaClient(settings.ollama_base_url, settings.ollama_model, settings.ollama_timeout_seconds)
    return ChatService(settings, ConversationRepository(settings.database_path), build_emotion_classifier(settings), SafetyService(), llm)

def confidence_meter(confidence: float) -> None:
    st.progress(int(confidence * 100), text=f"Model confidence: {confidence:.0%}")

st.set_page_config(page_title="MentalCare AI", page_icon="chat", layout="centered")
st.title("MentalCare AI")
st.warning("Wellness portfolio demo only: not therapy, diagnosis, or emergency support. In immediate danger, contact local emergency services.")
if "session_id" not in st.session_state: st.session_state.session_id = str(uuid4())
if "messages" not in st.session_state: st.session_state.messages = []
for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.write(item["content"])
        if item["role"] == "assistant":
            meta = item["metadata"]
            st.caption(f"Emotion: {meta['emotion']} | Risk: {meta['risk_level']}")
            confidence_meter(meta["confidence"])
if prompt := st.chat_input("Share what is on your mind"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.write(prompt)
    try:
        result = get_service().chat(prompt, st.session_state.session_id)
        st.session_state.messages.append({"role": "assistant", "content": result["response"], "metadata": result})
        with st.chat_message("assistant"):
            st.write(result["response"])
            st.caption(f"Emotion: {result['emotion']} | Risk: {result['risk_level']}")
            confidence_meter(result["confidence"])
        left, right = st.columns(2)
        if left.button("Helpful", key=f"up-{result['message_id']}"): get_service().repository.add_feedback(result["message_id"], True)
        if right.button("Not helpful", key=f"down-{result['message_id']}"): get_service().repository.add_feedback(result["message_id"], False)
    except Exception:
        st.error("I could not process that message. Please try again.")
