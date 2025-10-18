from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# モデル設定の一元管理
MODEL_CONFIG = {
    "claude-sonnet-4.5": {
        "provider": "anthropic",
        "display_name": "Claude Sonnet 4.5",
        "index": 0,
        "llm_factory": lambda temp: ChatAnthropic(
            temperature=1.0,
            model_name="claude-sonnet-4-5-20250929",
            max_tokens=16384,
            timeout=120,
            max_retries=3,
            thinking={"type": "enabled","budget_tokens": 8192}
        )
    },
    "gemini-2.5-pro": {
        "provider": "google",
        "display_name": "Gemini 2.5 Pro",
        "index": 1,
        "llm_factory": lambda temp: ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=1.0,
            thinking_budget=16000,
            include_thoughts=True
        )
    },
    "gpt-5": {
        "provider": "openai",
        "display_name": "GPT 5",
        "index": 2,
        "llm_factory": lambda temp: ChatOpenAI(
            model="gpt-5-chat-latest",
            temperature=1.0,
        )
    }
}