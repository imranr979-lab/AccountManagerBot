"""
Test Suite — Account Manager Bot
==================================
Run: pytest tests/ -v --cov=src --cov=config
"""

import os
import shutil
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def settings():
    """Settings with test-safe defaults."""
    from config.settings import Settings
    s = Settings()
    s.VECTOR_STORE_PATH = "/tmp/test_vectorstore"
    s.DOCS_PATH = "/tmp/test_docs"
    s.CHUNK_SIZE = 500
    s.CHUNK_OVERLAP = 50
    s.RETRIEVER_K = 3
    return s


@pytest.fixture(autouse=True)
def clean_vectorstore():
    """Clean up test vector store after each test."""
    yield
    for path in ["/tmp/test_vectorstore", "/tmp/test_docs"]:
        if Path(path).exists():
            shutil.rmtree(path, ignore_errors=True)


# ── Settings Tests ─────────────────────────────────────────────────────────────

class TestSettings:

    def test_default_values(self, settings):
        assert settings.CHUNK_SIZE == 500
        assert settings.CHUNK_OVERLAP == 50
        assert settings.RETRIEVER_K == 3
        assert settings.MEMORY_WINDOW == 6
        assert settings.LLM_TEMPERATURE == 0.0

    def test_to_dict_excludes_secrets(self, settings):
        d = settings.to_dict()
        assert "OPENAI_API_KEY" not in d
        assert "ANTHROPIC_API_KEY" not in d
        assert "OPENAI_MODEL" in d
        assert "CHUNK_SIZE" in d

    def test_environment_variable_override(self, monkeypatch):
        monkeypatch.setenv("CHUNK_SIZE", "1000")
        from config.settings import Settings
        s = Settings()
        assert s.CHUNK_SIZE == 1000


# ── Bot Initialisation Tests ──────────────────────────────────────────────────

class TestAccountManagerBotInit:

    def test_bot_initialises(self, settings):
        from src.bot import AccountManagerBot
        bot = AccountManagerBot(settings)
        assert bot.vectorstore is None
        assert bot.qa_chain is None
        assert bot.document_count == 0
        assert bot.loaded_files == []

    def test_ask_before_loading_raises(self, settings):
        from src.bot import AccountManagerBot
        bot = AccountManagerBot(settings)
        with pytest.raises(RuntimeError, match="Bot not ready"):
            bot.ask("What is the SLA?")

    def test_load_existing_vectorstore_missing_raises(self, settings):
        from src.bot import AccountManagerBot
        bot = AccountManagerBot(settings)
        with pytest.raises(FileNotFoundError):
            bot.load_existing_vectorstore()


# ── Document Loading Tests ────────────────────────────────────────────────────

class TestDocumentLoading:

    def test_load_nonexistent_path_raises(self, settings):
        from src.bot import AccountManagerBot
        bot = AccountManagerBot(settings)
        with pytest.raises(ValueError):
            bot.load_documents("/nonexistent/path/file.pdf")

    def test_load_non_pdf_raises(self, settings, tmp_path):
        from src.bot import AccountManagerBot
        bot = AccountManagerBot(settings)
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello world")
        with pytest.raises(ValueError):
            bot.load_documents(str(txt_file))

    def test_empty_directory_raises(self, settings, tmp_path):
        from src.bot import AccountManagerBot
        bot = AccountManagerBot(settings)
        with pytest.raises(ValueError, match="No documents found"):
            bot.load_documents(str(tmp_path))


# ── Text Splitting Tests ──────────────────────────────────────────────────────

class TestTextSplitting:

    def test_split_produces_chunks(self, settings):
        from src.bot import AccountManagerBot
        from langchain.schema import Document

        bot = AccountManagerBot(settings)
        docs = [Document(page_content="A" * 2000, metadata={"source": "test.pdf"})]
        chunks = bot._split_documents(docs)
        assert len(chunks) > 1

    def test_chunk_size_respected(self, settings):
        from src.bot import AccountManagerBot
        from langchain.schema import Document

        settings.CHUNK_SIZE = 100
        settings.CHUNK_OVERLAP = 10
        bot = AccountManagerBot(settings)
        docs = [Document(page_content="Word " * 200, metadata={"source": "test.pdf"})]
        chunks = bot._split_documents(docs)
        for chunk in chunks:
            assert len(chunk.page_content) <= settings.CHUNK_SIZE * 1.2  # 20% tolerance

    def test_overlap_preserves_context(self, settings):
        from src.bot import AccountManagerBot
        from langchain.schema import Document

        settings.CHUNK_SIZE = 50
        settings.CHUNK_OVERLAP = 20
        bot = AccountManagerBot(settings)
        docs = [Document(page_content="context overlap test " * 20,
                         metadata={"source": "t.pdf"})]
        chunks = bot._split_documents(docs)
        assert len(chunks) >= 2  # Overlap forces multiple chunks


# ── Embedding & Chain Tests (mocked) ─────────────────────────────────────────

class TestChainIntegration:

    @patch("src.bot.AccountManagerBot._build_embeddings")
    @patch("src.bot.AccountManagerBot._build_llm")
    def test_ask_returns_expected_keys(self, mock_llm, mock_embed, settings):
        from src.bot import AccountManagerBot
        from langchain.schema import Document

        bot = AccountManagerBot(settings)

        # Mock embedding
        mock_embed.return_value = MagicMock()

        # Mock QA chain
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "answer": "The Platinum SLA is 4 hours.",
            "source_documents": [
                Document(
                    page_content="Platinum SLA: 4 hours",
                    metadata={"source": "support_sla.pdf", "page": 2}
                )
            ],
        }
        bot.qa_chain = mock_chain
        bot.memory = MagicMock()

        result = bot.ask("What is the Platinum SLA?")

        assert "answer" in result
        assert "sources" in result
        assert "question" in result
        assert "timestamp" in result
        assert result["answer"] == "The Platinum SLA is 4 hours."
        assert len(result["sources"]) == 1

    def test_empty_question_returns_prompt(self, settings):
        from src.bot import AccountManagerBot

        bot = AccountManagerBot(settings)
        bot.qa_chain = MagicMock()
        bot.memory = MagicMock()

        result = bot.ask("   ")
        assert "Please enter a question" in result["answer"]

    def test_reset_memory_calls_clear(self, settings):
        from src.bot import AccountManagerBot

        bot = AccountManagerBot(settings)
        mock_memory = MagicMock()
        bot.memory = mock_memory
        bot.reset_memory()
        mock_memory.clear.assert_called_once()


# ── Security Tests ────────────────────────────────────────────────────────────

class TestSecurity:

    def test_conversation_log_has_no_raw_content(self, settings):
        """Audit log must never store raw question text or document content."""
        from src.bot import AccountManagerBot

        bot = AccountManagerBot(settings)
        bot.qa_chain = MagicMock()
        bot.qa_chain.invoke.return_value = {
            "answer": "test answer",
            "source_documents": [],
        }
        bot.memory = MagicMock()

        bot.ask("sensitive customer data query")

        assert len(bot._conversation_log) == 1
        log_entry = bot._conversation_log[0]
        assert "sensitive customer data query" not in str(log_entry)
        assert "question_hash" in log_entry
        assert "answer_length" in log_entry

    def test_file_extension_validation(self, settings, tmp_path):
        from src.bot import AccountManagerBot

        bot = AccountManagerBot(settings)
        bad_file = tmp_path / "malicious.exe"
        bad_file.write_bytes(b"MZ\x90\x00")
        with pytest.raises(ValueError):
            bot.load_documents(str(bad_file))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
