from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

MODEL_CONFIG = {
    "opus-4.6": {
        "provider": "anthropic",
        "display_name": "opus-4.6",
        "index": 0,
        "llm_factory": lambda temp: ChatAnthropic(
            temperature=temp,
            model_name="claude-opus-4-6",
            max_tokens=16384,
            timeout=120,
            max_retries=3,
            thinking={"type": "adaptive"},
        ),
    },
    "gemini-3.1": {
        "provider": "google",
        "display_name": "gemini-3.1 pro preview",
        "index": 1,
        "generation_config": {
            "thinking_config": {
                "thinking_level": "high",
                "include_thoughts": True,
            }
        },
        "llm_factory": lambda temp: ChatGoogleGenerativeAI(
            model="gemini-3.1-pro-preview",
            temperature=temp,
            output_version="v1",
        ),
    },
    "gpt-5.4": {
        "provider": "openai",
        "display_name": "gpt-5.4",
        "index": 2,
        "llm_factory": lambda temp: ChatOpenAI(
            model="gpt-5.4",
            temperature=temp,
            max_tokens=16384,
            reasoning={"effort": "medium", "summary": "auto"},
        ),
    },
}
