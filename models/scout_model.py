from groq import Groq
from config import GROQ_API_KEY, SCOUT_MODEL

class ScoutModel:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = SCOUT_MODEL

    def process(self, text: str) -> str:
        """Process text using Scout model."""
        message = self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": text}],
            max_tokens=1024,
        )
        return message.content[0].text