"""
Account Manager Bot — Core LangChain QA Engine
================================================
Combines:
  • PDF document loading          (PyPDFLoader)
  • Text splitting                (RecursiveCharacterTextSplitter)
  • Embeddings                    (HuggingFaceEmbeddings / OpenAIEmbeddings)
  • Vector store                  (Chroma)
  • Retrieval-Augmented Generation(RetrievalQA chain)
  • Memory                        (ConversationBufferMemory)

Usage:
    from src.bot import AccountManagerBot
    bot = AccountManagerBot()
    bot.load_documents("docs/")
    answer = bot.ask("What is the Platinum SLA?")
"""

import os
import logging
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate

from config.settings import Settings

logger = logging.getLogger(__name__)


# ── Prompts ───────────────────────────────────────────────────────────────────

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
Given the following conversation and a follow-up question, rephrase the
follow-up question to be a standalone question in its original language.

Chat History:
{chat_history}

Follow-Up Question: {question}
Standalone question:""")

QA_PROMPT = PromptTemplate.from_template("""
You are an expert Account Manager Assistant for NexaCorp. Your role is to
provide precise, helpful answers to sales, support, and account management
questions based ONLY on the provided context documents.

Rules:
- Answer based solely on the provided context. Do not use outside knowledge.
- If the answer is not in the context, say: "I don't have that information
  in the loaded documents. Please contact your Account Executive."
- Be concise but complete. Use bullet points for lists.
- When quoting specific values (prices, SLAs, percentages), be exact.
- Always maintain a professional, helpful tone.

Context:
{context}

Question: {question}

Answer:""")


class AccountManagerBot:
    """
    Core question-answering bot using LangChain RAG pipeline.

    Architecture:
        PDF Documents → PyPDFLoader → RecursiveCharacterTextSplitter
        → Embeddings → ChromaDB vector store → Retriever
        → ConversationalRetrievalChain → Response
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.vectorstore = None
        self.qa_chain = None
        self.memory = None
        self.loaded_files: List[str] = []
        self._conversation_log: List[Dict[str, Any]] = []
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, self.settings.LOG_LEVEL),
            format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        )

    # ── Document Loading ──────────────────────────────────────────────────────

    def load_documents(self, path: str) -> int:
        """
        Load PDF documents from a file path or directory.

        Args:
            path: Path to a single PDF or directory of PDFs.

        Returns:
            Number of document chunks created.
        """
        path_obj = Path(path)
        documents: List[Document] = []

        if path_obj.is_file() and path_obj.suffix.lower() == '.pdf':
            logger.info(f"Loading single PDF: {path}")
            loader = PyPDFLoader(str(path_obj))
            documents = loader.load()
            self.loaded_files = [str(path_obj)]

        elif path_obj.is_dir():
            logger.info(f"Loading PDFs from directory: {path}")
            loader = DirectoryLoader(
                str(path_obj),
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True,
            )
            documents = loader.load()
            self.loaded_files = [
                str(p) for p in path_obj.glob("**/*.pdf")
            ]
        else:
            raise ValueError(f"Path must be a PDF file or directory: {path}")

        if not documents:
            raise ValueError(f"No documents found at: {path}")

        logger.info(f"Loaded {len(documents)} raw pages from "
                    f"{len(self.loaded_files)} file(s)")

        # Add metadata tags for traceability
        for doc in documents:
            doc.metadata['loaded_at'] = datetime.now(timezone.utc).isoformat()
            doc.metadata['source_hash'] = hashlib.md5(
                doc.page_content.encode()).hexdigest()[:8]

        return self._index_documents(documents)

    # ── Text Splitting ────────────────────────────────────────────────────────

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks using RecursiveCharacterTextSplitter.

        Strategy: split on paragraph → sentence → word boundaries,
        with overlap to preserve context across chunk boundaries.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks "
                    f"(size={self.settings.CHUNK_SIZE}, "
                    f"overlap={self.settings.CHUNK_OVERLAP})")
        return chunks

    # ── Embeddings & Vector Store ─────────────────────────────────────────────

    def _build_embeddings(self):
        """
        Build embedding model.
        Uses HuggingFace sentence-transformers locally (no API key needed).
        Falls back to OpenAI embeddings if OPENAI_API_KEY is set.
        """
        if self.settings.OPENAI_API_KEY:
            from langchain_openai import OpenAIEmbeddings
            logger.info("Using OpenAI embeddings")
            return OpenAIEmbeddings(
                openai_api_key=self.settings.OPENAI_API_KEY,
                model="text-embedding-3-small",
            )
        else:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            logger.info("Using HuggingFace embeddings (local)")
            return HuggingFaceEmbeddings(
                model_name=self.settings.EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

    def _index_documents(self, documents: List[Document]) -> int:
        """Split, embed, and store documents in the vector store."""
        chunks = self._split_documents(documents)
        embeddings = self._build_embeddings()

        # Build ChromaDB vector store (persisted to disk)
        persist_dir = self.settings.VECTOR_STORE_PATH
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        from langchain_community.vectorstores import Chroma
        logger.info(f"Building Chroma vector store at: {persist_dir}")

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name="account_manager_docs",
        )
        logger.info(f"Vector store built with {len(chunks)} chunks")

        self._build_chain()
        return len(chunks)

    def load_existing_vectorstore(self):
        """Load a previously persisted vector store (skip re-indexing)."""
        from langchain_community.vectorstores import Chroma
        embeddings = self._build_embeddings()
        persist_dir = self.settings.VECTOR_STORE_PATH

        if not Path(persist_dir).exists():
            raise FileNotFoundError(
                f"No persisted vector store found at: {persist_dir}. "
                "Call load_documents() first."
            )
        self.vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="account_manager_docs",
        )
        logger.info("Loaded existing vector store")
        self._build_chain()

    # ── QA Chain ─────────────────────────────────────────────────────────────

    def _build_llm(self):
        """Build the LLM backend."""
        if self.settings.OPENAI_API_KEY:
            from langchain_openai import ChatOpenAI
            logger.info(f"Using OpenAI LLM: {self.settings.OPENAI_MODEL}")
            return ChatOpenAI(
                openai_api_key=self.settings.OPENAI_API_KEY,
                model_name=self.settings.OPENAI_MODEL,
                temperature=self.settings.LLM_TEMPERATURE,
                max_tokens=self.settings.MAX_TOKENS,
                streaming=True,
            )
        else:
            # Fallback: Anthropic Claude via API
            try:
                from langchain_anthropic import ChatAnthropic
                if self.settings.ANTHROPIC_API_KEY:
                    logger.info("Using Anthropic Claude LLM")
                    return ChatAnthropic(
                        anthropic_api_key=self.settings.ANTHROPIC_API_KEY,
                        model_name="claude-sonnet-4-6",
                        temperature=self.settings.LLM_TEMPERATURE,
                        max_tokens=self.settings.MAX_TOKENS,
                    )
            except ImportError:
                pass
            raise EnvironmentError(
                "No LLM API key found. Set OPENAI_API_KEY or "
                "ANTHROPIC_API_KEY in your .env file."
            )

    def _build_chain(self):
        """Assemble the ConversationalRetrievalChain."""
        if not self.vectorstore:
            raise RuntimeError("Vector store not initialised. Call load_documents() first.")

        llm = self._build_llm()

        # Memory: keep last N turns in context window
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            k=self.settings.MEMORY_WINDOW,
        )

        # Retriever: MMR for diversity, fetch top-K chunks
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.settings.RETRIEVER_K,
                "fetch_k": self.settings.RETRIEVER_K * 3,
                "lambda_mult": 0.7,  # 0=diversity, 1=relevance
            },
        )

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=self.memory,
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True,
            verbose=self.settings.VERBOSE,
        )
        logger.info("ConversationalRetrievalChain built successfully")

    # ── Public Query Interface ────────────────────────────────────────────────

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and receive an answer grounded in loaded documents.

        Args:
            question: Natural language question string.

        Returns:
            Dict with keys: answer, sources, question, timestamp
        """
        if not self.qa_chain:
            raise RuntimeError(
                "Bot not ready. Call load_documents() or "
                "load_existing_vectorstore() first."
            )

        if not question.strip():
            return {
                "answer": "Please enter a question.",
                "sources": [],
                "question": question,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        logger.info(f"Processing question: {question[:80]}...")

        result = self.qa_chain.invoke({"question": question})

        # Extract unique source documents
        sources = []
        seen = set()
        for doc in result.get("source_documents", []):
            src = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            key = f"{src}:p{page}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "file": Path(src).name if src != "Unknown" else src,
                    "page": page,
                    "excerpt": doc.page_content[:200].strip() + "...",
                })

        response = {
            "answer": result.get("answer", "No answer generated."),
            "sources": sources,
            "question": question,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Audit log (never contains raw document text — only metadata)
        self._conversation_log.append({
            "timestamp": response["timestamp"],
            "question_hash": hashlib.sha256(question.encode()).hexdigest()[:16],
            "source_count": len(sources),
            "answer_length": len(response["answer"]),
        })

        return response

    def reset_memory(self):
        """Clear conversation history (useful for a new session)."""
        if self.memory:
            self.memory.clear()
            logger.info("Conversation memory cleared")

    @property
    def document_count(self) -> int:
        """Return number of chunks in the vector store."""
        if self.vectorstore:
            return self.vectorstore._collection.count()
        return 0

    @property
    def loaded_document_names(self) -> List[str]:
        return [Path(f).name for f in self.loaded_files]
