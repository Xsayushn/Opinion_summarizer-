"""
Multi-provider LLM Client supporting Gemini, Groq, Ollama, and offline Mock fallback.
"""

from typing import Optional, Dict, Any
import os
import requests
import json
from src.config import (
    LLM_PROVIDER, GEMINI_API_KEY, GROQ_API_KEY,
    OLLAMA_BASE_URL, OLLAMA_MODEL, GEMINI_MODEL, GROQ_MODEL
)


class LLMClient:
    """
    Unified LLM caller with multi-provider dispatch and automatic fallback.
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or LLM_PROVIDER).lower()

    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        """Dispatches generation to configured LLM provider."""
        if self.provider == "gemini" and (GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")):
            return self._call_gemini(prompt, temperature)
        elif self.provider == "groq" and (GROQ_API_KEY or os.getenv("GROQ_API_KEY")):
            return self._call_groq(prompt, temperature)
        elif self.provider == "ollama":
            return self._call_ollama(prompt, temperature)
        else:
            return self._call_mock(prompt)

    def _call_gemini(self, prompt: str, temperature: float) -> str:
        try:
            from google import genai
            from google.genai import types
            api_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
            client = genai.Client(api_key=api_key)
            config = types.GenerateContentConfig(temperature=temperature)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=config
            )
            return response.text or ""
        except Exception as e:
            # Fallback to mock on error
            return f"### Summary (Gemini API unavailable: {e})\n" + self._call_mock(prompt)

    def _call_groq(self, prompt: str, temperature: float) -> str:
        try:
            api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY")
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature
            }
            res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=30)
            if res.status_code == 200:
                data = res.json()
                return data["choices"][0]["message"]["content"]
            else:
                return self._call_mock(prompt)
        except Exception:
            return self._call_mock(prompt)

    def _call_ollama(self, prompt: str, temperature: float) -> str:
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature}
            }
            res = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=45)
            if res.status_code == 200:
                return res.json().get("response", "")
            else:
                return self._call_mock(prompt)
        except Exception:
            return self._call_mock(prompt)

    def _call_mock(self, prompt: str) -> str:
        """
        Deterministic Mock Generator for offline testing and verification.
        Generates structured dialectical summaries with citations [E1], [E2]...
        """
        return """### Consensus & Broad Agreement
Customers widely agree that the primary features perform reliably under normal conditions [E1, E2]. Overall customer satisfaction remains solidly positive across everyday interactions [E3].

### Contentions & Divergent Views
However, notable disagreements emerge under demanding conditions [E4]. A significant minority of users reported unexpected issues or delays, creating a distinct contrast with positive accounts [E5].

### Key Aspect Breakdown
- **Primary Observations**: Most users highlight consistent quality [E1], while negative experiences were concentrated in isolated high-stress scenarios [E4].
"""
