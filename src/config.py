"""
Task 1: Project Foundation & Multi-Provider Configuration Management.
Loads environment variables, centralizes system hyperparameters, and auto-detects
LLM providers: NVIDIA NIM, OpenRouter, Groq, Google Gemini, and OpenAI.
"""
import os
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv(override=True)


class Config:
    # Provider API Keys from .env
    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    DEFAULT_API_KEY: str = (
        NVIDIA_API_KEY
        or OPENROUTER_API_KEY
        or GROQ_API_KEY
        or GEMINI_API_KEY
        or OPENAI_API_KEY
    )

    # Provider Default LLM Models
    NVIDIA_LLM_MODEL: str = os.getenv(
        "NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b"
    )
    OPENROUTER_LLM_MODEL: str = os.getenv(
        "OPENROUTER_MODEL", "openai/gpt-4o-mini"
    )
    GROQ_LLM_MODEL: str = os.getenv(
        "GROQ_MODEL", "llama-3.3-70b-versatile"
    )
    GEMINI_LLM_MODEL: str = os.getenv(
        "GEMINI_MODEL", "gemini-2.5-flash"
    )
    LLM_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Provider Default Embedding Models
    NVIDIA_EMBEDDING_MODEL: str = os.getenv(
        "NVIDIA_EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b"
    )
    EMBEDDING_MODEL: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "auto")

    # OpenAI-Compatible Base URLs for Multi-Provider Support
    PROVIDER_BASE_URLS: Dict[str, str] = {
        "openrouter": "https://openrouter.ai/api/v1",
        "groq": "https://api.groq.com/openai/v1",
        "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "nvidia": "https://integrate.api.nvidia.com/v1",
        "openai": "https://api.openai.com/v1",
    }

    PROVIDER_LABELS: Dict[str, str] = {
        "nvidia": "NVIDIA NIM",
        "openrouter": "OpenRouter",
        "groq": "Groq Cloud",
        "gemini": "Google Gemini",
        "openai": "OpenAI",
    }

    PROVIDER_ENV_KEYS: Dict[str, str] = {
        "nvidia": "NVIDIA_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "groq": "GROQ_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
    }

    PROVIDER_MODELS: Dict[str, List[str]] = {
        "nvidia": [
            "nvidia/nemotron-3.5-lightning-30b-a3b",
            "nvidia/nemotron-nano-3-30b-a3b",
            "nvidia/llama-3.1-nemotron-70b-instruct",
            "nvidia/nemotron-3-super-120b-a12b",
            "google/gemma-4-31b-it",
            "deepseek-ai/deepseek-v4.1-flash",
            "mistralai/mistral-large-2-instruct",
        ],
        "openrouter": [
            "openai/gpt-4o-mini",
            "meta-llama/llama-3.3-70b-instruct",
            "google/gemini-2.5-flash",
            "deepseek/deepseek-chat",
            "anthropic/claude-3.5-haiku",
            "qwen/qwen-2.5-72b-instruct",
        ],
        "groq": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        "gemini": [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ],
        "openai": [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ],
    }

    # Vector Store Settings (matches vector_db/ folder requirement in Google Doc)
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./vector_db")
    COLLECTION_NAME: str = os.getenv(
        "COLLECTION_NAME", "enterprise_knowledge_base"
    )

    # Chunking Hyperparameters (Task 4)
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    # Retrieval Hyperparameters (Task 6)
    TOP_K: int = 5
    ALLOWED_EXTENSIONS: set = {".pdf", ".txt", ".csv", ".xlsx", ".xls"}
    MAX_FILE_SIZE_MB: int = 25

    @staticmethod
    def detect_provider(api_key: str, override_provider: str = "auto") -> str:
        """
        Auto-detects the LLM provider from the API key prefix, or uses explicit override.
        - 'nvapi-...' -> 'nvidia'
        - 'sk-or-...' -> 'openrouter'
        - 'gsk_...'   -> 'groq'
        - 'AIza...'   -> 'gemini'
        - 'sk-...'    -> 'openai'
        """
        if override_provider and override_provider.lower() not in ("auto", ""):
            return override_provider.lower()

        if not api_key:
            return "openai"

        key = api_key.strip()
        if key.startswith("nvapi-"):
            return "nvidia"
        if key.startswith("sk-or-"):
            return "openrouter"
        if key.startswith("gsk_"):
            return "groq"
        if key.startswith("AIza"):
            return "gemini"
        return "openai"

    @staticmethod
    def save_key_to_env(api_key: str, provider: str = "auto") -> None:
        """Saves the active API key into the local .env file so evaluators have it configured."""
        if not api_key or not api_key.strip():
            return
        key = api_key.strip()
        det = Config.detect_provider(key, override_provider=provider)
        env_var = Config.PROVIDER_ENV_KEYS.get(det, "OPENAI_API_KEY")
        env_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), ".env"
        )
        lines = []
        found = False
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith(f"{env_var}="):
                        lines.append(f'{env_var}="{key}"\n')
                        found = True
                    else:
                        lines.append(line)
        if not found:
            lines.append(f'{env_var}="{key}"\n')
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
