from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# モデル設定の一元管理
MODEL_CONFIG = {
    "claude-opus-4.5": {
        "provider": "anthropic",
        "display_name": "claude-opus-4.5",
        "index": 0,
        "llm_factory": lambda temp: ChatAnthropic(
            temperature=1.0,
            model_name="claude-opus-4-5-20251101",
            max_tokens=16384,
            timeout=120,
            max_retries=3,
            thinking={"type": "enabled","budget_tokens": 8192}
        )
    },
    "gemini-3.0-pro": {
        "provider": "google",
        "display_name": "gemini-3.0-pro",
        "index": 1,
        "llm_factory": lambda temp: ChatGoogleGenerativeAI(
            model="gemini-3-pro-preview",
            temperature=1.0,
            thinking_budget=16384,
            include_thoughts=True,
            output_version="v1"
        )
    },
    "gpt-5.2": {
        "provider": "openai",
        "display_name": "gpt-5.2",
        "index": 2,
        "llm_factory": lambda temp: ChatOpenAI(
            model="gpt-5.2",
            temperature=1.0,
            max_tokens=16384,
            reasoning={"effort": "medium","summary": "auto"}
        )
    }
}