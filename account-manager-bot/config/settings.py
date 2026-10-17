"""
Centralised Settings — loaded from environment variables / .env file.
NEVER commit secrets to source control.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(Path(__file__).parent.parent / ".env")


class Settings:

    # ── LLM Backend ──────────────────────────────────────────────────────────
    OPENAI_API_KEY:     str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY:  str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_MODEL:       str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE:  float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    MAX_TOKENS:         int = int(os.getenv("MAX_TOKENS", "1024"))

    # ── Embeddings ────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ── Document Processing ───────────────────────────────────────────────────
    CHUNK_SIZE:    int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # ── Retrieval ─────────────────────────────────────────────────────────────
    RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", "5"))

    # ── Memory ────────────────────────────────────────────────────────────────
    MEMORY_WINDOW: int = int(os.getenv("MEMORY_WINDOW", "6"))

    # ── Storage ───────────────────────────────────────────────────────────────
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "./data/vectorstore")
    DOCS_PATH:         str = os.getenv("DOCS_PATH", "./docs")

    # ── App ───────────────────────────────────────────────────────────────────
    APP_TITLE:   str  = os.getenv("APP_TITLE",   "Account Manager Bot")
    APP_VERSION: str  = os.getenv("APP_VERSION", "1.0.0")
    ENVIRONMENT: str  = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL:   str  = os.getenv("LOG_LEVEL",   "INFO")
    VERBOSE:     bool = os.getenv("VERBOSE", "false").lower() == "true"

    # ── Security ──────────────────────────────────────────────────────────────
    ALLOWED_EXTENSIONS: list = [".pdf"]
    MAX_FILE_SIZE_MB:   int  = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    ENABLE_AUTH:        bool = os.getenv("ENABLE_AUTH", "false").lower() == "true"

    # ── Gradio ────────────────────────────────────────────────────────────────
    GRADIO_SERVER_PORT: int  = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    GRADIO_SHARE:       bool = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    GRADIO_AUTH_USER:   str  = os.getenv("GRADIO_AUTH_USER", "")
    GRADIO_AUTH_PASS:   str  = os.getenv("GRADIO_AUTH_PASS", "")

    def to_dict(self) -> dict:
        """Return safe (non-secret) settings for logging."""
        return {
            "OPENAI_MODEL":      self.OPENAI_MODEL,
            "EMBEDDING_MODEL":   self.EMBEDDING_MODEL,
            "CHUNK_SIZE":        self.CHUNK_SIZE,
            "CHUNK_OVERLAP":     self.CHUNK_OVERLAP,
            "RETRIEVER_K":       self.RETRIEVER_K,
            "MEMORY_WINDOW":     self.MEMORY_WINDOW,
            "VECTOR_STORE_PATH": self.VECTOR_STORE_PATH,
            "ENVIRONMENT":       self.ENVIRONMENT,
            "LOG_LEVEL":         self.LOG_LEVEL,
        }
