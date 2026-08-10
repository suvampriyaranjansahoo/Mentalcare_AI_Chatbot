from app.config import Settings
from app.nlp.classifier import EmotionClassifier
from app.nlp.huggingface import HuggingFaceEmotionClassifier

def build_emotion_classifier(settings: Settings):
    if settings.emotion_provider == "huggingface":
        return HuggingFaceEmotionClassifier(settings.hf_emotion_model)
    return EmotionClassifier(settings.model_path)
