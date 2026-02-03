from collections import Counter
from datetime import datetime
from typing import List, Dict, Any

from config.settings import Config


class MedicalChatbot:
    """
    Eye disease medical chatbot using semantic search.
    """

    def __init__(self, semantic_search):
        self.search = semantic_search
        self.history = []
        self.start_time = datetime.now()

    def get_answer(self, question: str, k: int = Config.TOP_K_RESULTS) -> Dict[str, Any]:
        # Emergency detection
        if self.detect_emergency(question):
            return {
                "answer": "⚠️ Emergency symptoms detected. Please seek urgent medical attention immediately.",
                "disease": "emergency",
                "confidence": 1.0,
                "related_diseases": []
            }

        results = self.search.query(question, k)

        if not results:
            return {
                "answer": "No relevant medical information found. Please consult a doctor for diagnosis.",
                "disease": "unknown",
                "confidence": 0.0,
                "related_diseases": []
            }

        disease_counts = Counter(r["disease"] for r in results)
        main_disease = disease_counts.most_common(1)[0][0]

        answer = " ".join(r["text"] for r in results)
        answer = answer[:Config.MAX_RESPONSE_LENGTH]

        confidence = sum(r["score"] for r in results) / len(results)

        self.history.append((question, main_disease))

        return {
            "answer": answer,
            "disease": main_disease,
            "confidence": round(confidence, 3),
            "related_diseases": list(disease_counts.keys())
        }

    def clear_conversation(self):
        self.history = []

    def get_conversation_summary(self) -> Dict[str, Any]:
        duration = (datetime.now() - self.start_time).seconds / 60
        diseases = [d for _, d in self.history]

        return {
            "total_questions": len(self.history),
            "duration_minutes": round(duration, 2),
            "recent_diseases": list(dict.fromkeys(diseases))
        }

    @staticmethod
    def detect_emergency(question: str) -> bool:
        q = question.lower()
        return any(keyword in q for keyword in Config.EMERGENCY_KEYWORDS)


class SymptomAnalyzer:
    """
    A simple symptom analyzer for emergency detection and symptom extraction.
    """

    @staticmethod
    def extract_symptoms(text: str) -> List[str]:
        text = text.lower()
        symptoms = []
        for keyword in Config.EMERGENCY_KEYWORDS:
            if keyword in text:
                symptoms.append(keyword)
        return symptoms

    @staticmethod
    def is_emergency(text: str) -> bool:
        return len(SymptomAnalyzer.extract_symptoms(text)) > 0
