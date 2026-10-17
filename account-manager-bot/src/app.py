"""
Account Manager Bot — Gradio Web Interface
==========================================
Provides a conversational chat UI with:
  • File upload panel for PDF documents
  • Chat interface with conversation history
  • Source citation display
  • Session management (clear memory)
  • Document status panel
"""

import os
import logging
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr

from src.bot import AccountManagerBot
from config.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

# Global bot instance (reused across sessions in single-user deployment)
# For multi-user production, use per-session state via gr.State()
_bot: Optional[AccountManagerBot] = None


def get_bot() -> AccountManagerBot:
    """Return the global bot instance, creating it if needed."""
    global _bot
    if _bot is None:
        _bot = AccountManagerBot(settings)
        # Auto-load docs if DOCS_PATH exists
        docs_path = Path(settings.DOCS_PATH)
        if docs_path.exists() and any(docs_path.glob("*.pdf")):
            try:
                count = _bot.load_documents(str(docs_path))
                logger.info(f"Auto-loaded {count} chunks from {settings.DOCS_PATH}")
            except Exception as e:
                logger.warning(f"Auto-load failed: {e}")
    return _bot


# ── Gradio callback functions ─────────────────────────────────────────────────

def upload_and_index(files: List[tempfile.TemporaryFile]) -> Tuple[str, str]:
    """Handle PDF uploads and trigger document indexing."""
    if not files:
        return "⚠️ No files uploaded.", ""

    bot = get_bot()
    upload_dir = Path(settings.DOCS_PATH)
    upload_dir.mkdir(parents=True, exist_ok=True)

    uploaded_names = []
    for f in files:
        src = Path(f.name)
        if src.suffix.lower() != ".pdf":
            return f"❌ Only PDF files are supported. Got: {src.name}", ""

        # File size check
        size_mb = src.stat().st_size / (1024 * 1024)
        if size_mb > settings.MAX_FILE_SIZE_MB:
            return (
                f"❌ File too large: {src.name} ({size_mb:.1f} MB). "
                f"Maximum: {settings.MAX_FILE_SIZE_MB} MB"
            ), ""

        dest = upload_dir / src.name
        shutil.copy2(src, dest)
        uploaded_names.append(src.name)

    try:
        chunk_count = bot.load_documents(str(upload_dir))
        doc_list = "\n".join(f"• {n}" for n in bot.loaded_document_names)
        status = (
            f"✅ Indexed {len(uploaded_names)} new document(s) into "
            f"{chunk_count} searchable chunks.\n\n"
            f"**Loaded documents:**\n{doc_list}"
        )
        return status, f"📚 {chunk_count} chunks indexed"
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        return f"❌ Indexing failed: {str(e)}", ""


def chat(
    message: str,
    history: List[Tuple[str, str]],
    show_sources: bool,
) -> Tuple[List[Tuple[str, str]], str]:
    """Process a chat message and return updated history + sources."""
    bot = get_bot()

    if not message.strip():
        return history, ""

    if bot.document_count == 0:
        history.append((message, (
            "⚠️ No documents loaded yet. Please upload PDF files using the "
            "panel on the left before asking questions."
        )))
        return history, ""

    try:
        result = bot.ask(message)
        answer = result["answer"]

        # Format source citations
        sources_text = ""
        if show_sources and result["sources"]:
            lines = ["\n\n---\n📎 **Sources:**"]
            for i, src in enumerate(result["sources"], 1):
                lines.append(
                    f"{i}. **{src['file']}** (page {src['page']}) — "
                    f"*{src['excerpt'][:100]}...*"
                )
            sources_text = "\n".join(lines)

        full_answer = answer + sources_text
        history.append((message, full_answer))

        # Build sources panel content
        sources_panel = ""
        if result["sources"]:
            srcs = [f"**{s['file']}** p.{s['page']}" for s in result["sources"]]
            sources_panel = "📎 Sources: " + " | ".join(srcs)

        return history, sources_panel

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        history.append((message, f"❌ Error processing your question: {str(e)}"))
        return history, ""


def clear_chat():
    """Clear the chat history and conversation memory."""
    bot = get_bot()
    bot.reset_memory()
    return [], "💬 Chat cleared. Memory reset."


def get_document_status() -> str:
    """Return a status string for the loaded documents panel."""
    bot = get_bot()
    if bot.document_count == 0:
        return "📂 No documents loaded. Upload PDFs to get started."

    doc_list = "\n".join(f"• {n}" for n in bot.loaded_document_names) or "• (pre-loaded)"
    return (
        f"✅ **{bot.document_count}** chunks indexed\n\n"
        f"**Loaded documents:**\n{doc_list}"
    )


# ── Gradio Interface Builder ──────────────────────────────────────────────────

def build_interface() -> gr.Blocks:
    """Build and return the Gradio Blocks interface."""

    css = """
    .gradio-container { max-width: 1200px !important; font-family: 'Segoe UI', sans-serif; }
    .chat-box { border-radius: 12px; }
    footer { display: none !important; }
    .source-panel { font-size: 0.82rem; color: #555; padding: 6px 10px;
                    background: #f0f4f8; border-radius: 6px; }
    """

    with gr.Blocks(
        title=settings.APP_TITLE,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css=css,
    ) as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center; padding:20px 0 10px;">
          <h1 style="margin:0; font-size:2rem; color:#1A3A5C;">
            📋 Account Manager Bot
          </h1>
          <p style="color:#556677; margin-top:6px; font-size:1rem;">
            Intelligent Q&A powered by LangChain · Answers grounded in your company documents
          </p>
        </div>
        """)

        with gr.Row():

            # ── LEFT PANEL: Document Management ───────────────────────────────
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### 📂 Document Management")

                upload_btn = gr.File(
                    label="Upload PDF Documents",
                    file_types=[".pdf"],
                    file_count="multiple",
                    height=120,
                )

                index_btn = gr.Button(
                    "🔍 Index Documents",
                    variant="primary",
                    size="sm",
                )

                upload_status = gr.Markdown(
                    value=get_document_status(),
                    label="Status",
                )

                chunk_badge = gr.Textbox(
                    label="Index Status",
                    interactive=False,
                    value="",
                )

                gr.Markdown("---")
                gr.Markdown("### ⚙️ Options")

                show_sources = gr.Checkbox(
                    label="Show source citations",
                    value=True,
                )

                clear_btn = gr.Button(
                    "🗑️ Clear Chat & Memory",
                    variant="secondary",
                    size="sm",
                )

                gr.Markdown("---")
                gr.Markdown("""
**Tips:**
- Upload your company PDFs above
- Ask about pricing, SLAs, contacts
- Follow-up questions are remembered
- Sources show which document answered
                """)

            # ── RIGHT PANEL: Chat Interface ────────────────────────────────────
            with gr.Column(scale=3):
                gr.Markdown("### 💬 Ask Your Account Manager Bot")

                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=480,
                    show_copy_button=True,
                    elem_classes=["chat-box"],
                    avatar_images=(None, "🤖"),
                    bubble_full_width=False,
                )

                sources_display = gr.Markdown(
                    value="",
                    elem_classes=["source-panel"],
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="e.g. What is the Platinum account SLA for Severity 1 issues?",
                        label="Your Question",
                        lines=2,
                        scale=5,
                        show_label=False,
                    )
                    send_btn = gr.Button(
                        "Send ➤",
                        variant="primary",
                        scale=1,
                        min_width=80,
                    )

                # Suggested example questions
                gr.Markdown("**Try these example questions:**")
                with gr.Row():
                    ex1 = gr.Button("What is the SLA for Platinum accounts?",
                                    size="sm", variant="secondary")
                    ex2 = gr.Button("What discount can an AE approve without approval?",
                                    size="sm", variant="secondary")
                    ex3 = gr.Button("What are the pricing tiers for NexaCorp?",
                                    size="sm", variant="secondary")

        # ── Event Wiring ──────────────────────────────────────────────────────

        index_btn.click(
            fn=upload_and_index,
            inputs=[upload_btn],
            outputs=[upload_status, chunk_badge],
        )

        send_btn.click(
            fn=chat,
            inputs=[msg_input, chatbot, show_sources],
            outputs=[chatbot, sources_display],
            queue=True,
        ).then(fn=lambda: "", outputs=msg_input)

        msg_input.submit(
            fn=chat,
            inputs=[msg_input, chatbot, show_sources],
            outputs=[chatbot, sources_display],
            queue=True,
        ).then(fn=lambda: "", outputs=msg_input)

        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, sources_display],
        )

        # Example question buttons
        for btn, q in [(ex1, "What is the SLA for Platinum accounts?"),
                       (ex2, "What discount can an AE approve without manager approval?"),
                       (ex3, "What are the pricing tiers for the NexaCorp Core Platform?")]:
            btn.click(
                fn=lambda question=q: question,
                outputs=msg_input,
            )

        gr.HTML("""
        <div style="text-align:center; padding:12px; color:#888; font-size:.78rem;">
          Account Manager Bot v{} · Answers grounded in loaded documents only ·
          Not a substitute for official company policy
        </div>
        """.format(settings.APP_VERSION))

    return demo


# ── Entry point ────────────────────────────────────────────────────────────────

def launch():
    """Launch the Gradio application."""
    demo = build_interface()

    # Optional authentication
    auth = None
    if settings.ENABLE_AUTH and settings.GRADIO_AUTH_USER:
        auth = (settings.GRADIO_AUTH_USER, settings.GRADIO_AUTH_PASS)

    demo.launch(
        server_name="0.0.0.0",
        server_port=settings.GRADIO_SERVER_PORT,
        share=settings.GRADIO_SHARE,
        auth=auth,
        show_error=True,
        quiet=False,
    )


if __name__ == "__main__":
    launch()
