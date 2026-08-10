from __future__ import annotations

from flask import Flask, jsonify, render_template, request, session

from app.analytics import session_analytics
from app.config import Settings
from app.database import ConversationRepository
from app.nlp.factory import build_emotion_classifier
from app.safety import SafetyService
from app.services import ChatService
from app.llm import OllamaClient


def _error(message: str, status: int, code: str = "validation_error"):
    return jsonify({"error": {"code": code, "message": message}}), status


def create_app(settings: Settings | None = None) -> Flask:
    settings = settings or Settings.from_env()
    app = Flask(__name__, template_folder="../../flask_app/templates", static_folder="../../flask_app/static")
    app.config.update(SECRET_KEY=settings.secret_key, SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")
    repository = ConversationRepository(settings.database_path)
    llm = OllamaClient(settings.ollama_base_url, settings.ollama_model, settings.ollama_timeout_seconds)
    service = ChatService(settings, repository, build_emotion_classifier(settings), SafetyService(), llm)
    app.extensions["chat_service"] = service
    app.extensions["repository"] = repository
    app.extensions["settings"] = settings

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        return response

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "service": "mentalcare-ai"})

    @app.get("/model-info")
    def model_info():
        classifier = service.classifier
        return jsonify({"model_version": settings.model_version, "labels": classifier.labels, "confidence_note": "SVM margin-derived uncertainty heuristic; not calibrated clinical probability."})

    def parse_body():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise TypeError("A JSON object is required")
        return payload

    @app.post("/chat")
    @app.post("/predict")
    def chat():
        try:
            payload = parse_body()
            result = service.chat(payload.get("message"), payload.get("session_id") or session.get("session_id"))
            session["session_id"] = result["session_id"]
            return jsonify(result), 200
        except TypeError as exc:
            return _error(str(exc), 400, "malformed_request")
        except ValueError as exc:
            return _error(str(exc), 422)
        except Exception:
            app.logger.exception("chat request failed")
            return _error("Unable to process the request.", 500, "internal_error")

    @app.get("/sessions/<session_id>")
    def session_detail(session_id: str):
        messages = repository.session_messages(session_id)
        return jsonify({"messages": messages, "analytics": session_analytics(messages)})

    @app.post("/feedback")
    def feedback():
        try:
            payload = parse_body()
            message_id, helpful = payload.get("message_id"), payload.get("helpful")
            if not isinstance(message_id, int) or not isinstance(helpful, bool):
                return _error("message_id (integer) and helpful (boolean) are required", 422)
            if not repository.add_feedback(message_id, helpful):
                return _error("message_id was not found", 404, "not_found")
            return jsonify({"status": "recorded"}), 201
        except TypeError as exc:
            return _error(str(exc), 400, "malformed_request")

    @app.delete("/sessions/<session_id>")
    def delete_session(session_id: str):
        # Explicit deletion endpoint kept intentionally narrow: this implementation
        # removes only messages belonging to the requested opaque session id.
        with repository.connect() as conn:
            cursor = conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        return jsonify({"deleted_messages": cursor.rowcount})

    return app
