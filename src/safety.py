class SafetyProtocol:
    EMERGENCY_KEYWORDS = [
        "sudden vision loss",
        "loss of vision",
        "flashing lights",
        "many floaters",
        "eye injury",
        "chemical exposure",
        "severe eye pain",
        "trauma"
    ]

    def evaluate(self, text: str):
        text = text.lower()
        emergency = any(k in text for k in self.EMERGENCY_KEYWORDS)

        return {
            "emergency": {
                "is_emergency": emergency,
                "message": "🚨 This may be an eye emergency.",
                "recommendations": [
                    "Seek immediate medical attention",
                    "Visit an emergency department",
                    "Avoid self-medication"
                ] if emergency else []
            }
        }

    def get_when_to_see_doctor(self):
        return """
        ### 🚨 See an eye doctor immediately if you experience:
        - Sudden loss of vision
        - Severe eye pain
        - Eye injury or trauma
        - Flashes of light or many floaters
        - Red eye with pain and vision loss
        """
