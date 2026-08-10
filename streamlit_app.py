"""Streamlit deployment entry point for Streamlit Community Cloud."""
from __future__ import annotations

import os
from uuid import uuid4

import streamlit as st

from app.config import Settings
from app.database import ConversationRepository
from app.nlp import EmotionClassifier
from app.safety import SafetyService
from app.services import ChatService


@st.cache_resource
def get_service() -> ChatService:
    if "FLASK_SECRET_KEY" in st.secrets:
        os.environ["FLASK_SECRET_KEY"] = st.secrets["FLASK_SECRET_KEY"]
    settings = Settings.from_env()
    return ChatService(settings, ConversationRepository(settings.database_path), EmotionClassifier(settings.model_path), SafetyService())


st.set_page_config(page_title="MentalCare AI", page_icon="💬", layout="centered")
st.title("MentalCare AI")
st.warning("Portfolio wellness demo only — not therapy, diagnosis, or emergency support. In immediate danger, contact local emergency services.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.write(item["content"])
        if item["role"] == "assistant" and item.get("metadata"):
            metadata = item["metadata"]
            st.caption(f"Emotion: {metadata['emotion']} · Confidence: {metadata['confidence']} · Risk: {metadata['risk_level']}")

if prompt := st.chat_input("Share what is on your mind"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    try:
        result = get_service().chat(prompt, st.session_state.session_id)
        st.session_state.messages.append({"role": "assistant", "content": result["response"], "metadata": result})
        with st.chat_message("assistant"):
            st.write(result["response"])
            st.caption(f"Emotion: {result['emotion']} · Confidence: {result['confidence']} · Risk: {result['risk_level']}")
        left, right = st.columns(2)
        if left.button("👍 Helpful", key=f"up-{result['message_id']}"):
            get_service().repository.add_feedback(result["message_id"], True)
        if right.button("👎 Not helpful", key=f"down-{result['message_id']}"):
            get_service().repository.add_feedback(result["message_id"], False)
    except Exception:
        st.error("I could not process that message. Please try again.")

with st.expander("Session mood analytics"):
    records = get_service().repository.session_messages(st.session_state.session_id)
    if records:
        st.bar_chart({"emotion": [record["predicted_emotion"] for record in records], "confidence": [record["confidence"] for record in records]}, x="emotion")
    else:
        st.caption("Analytics will appear after your first message.")
