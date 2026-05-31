"""
Chainlit frontend for the Agentic RAG Personal Chat System.

Connects to the FastAPI backend via httpx and streams token-by-token
responses using Server-Sent Events. Retrieved source documents are
displayed in the side panel.
"""

import os

import chainlit as cl
import httpx
from chainlit.input_widget import Select

from src.agentic_rag_personal_chat_system.frontend.src.main_utils import (
    handle_event_type,
    handle_streamed_response,
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


@cl.on_chat_start
async def on_chat_start():
    settings = await cl.ChatSettings(
        [
            Select(
                id="mode",
                label="Assistant Mode",
                values=["auto", "personal", "technical"],
                initial_value="auto",
                description=(
                    "auto = smart routing  |  "
                    "personal = your documents  |  "
                    "technical = coding & tech"
                ),
            )
        ]
    ).send()
    cl.user_session.set("mode", settings.get("mode", "auto"))

    await cl.Message(
        content=(
            "👋 **Welcome to your Personal AI Assistant!**\n\n"
            "I'm an intelligent multi-agent RAG (Retrieval-Augmented Generation) system "
            "designed to give you accurate, context-aware answers — drawing from both your "
            "**personal documents** and deep **technical knowledge**.\n\n"
            "---\n\n"
            "### 🤖 My two specialist assistants\n\n"
            "| Assistant | What it knows |\n"
            "|-----------|---------------|\n"
            "| 🔧 **Technical** | Programming, debugging, algorithms, data science, software architecture |\n"
            "| 📄 **Personal** | Your documents (leases, notes, reports), general knowledge, everyday questions |\n\n"
            "---\n\n"
            "### ⚙️ How to get started\n\n"
            "1. **Pick a mode** — use the settings icon (⚙️) above to choose `auto`, `personal`, or `technical`\n"
            "2. **Ask anything** — type your question below and I'll find the most relevant answer\n"
            "3. **Check your sources** — cited documents appear in the **side panel** on the right\n\n"
            "---\n\n"
            "> 💡 **Tip:** Leave the mode on **auto** and I'll intelligently route your question "
            "to the right assistant every time.\n\n"
            "_Go ahead — what would you like to know?_ 🚀"
        )
    ).send()


@cl.on_settings_update
async def on_settings_update(settings):
    cl.user_session.set("mode", settings["mode"])
    await cl.Message(
        content=f"✅ Mode switched to **{settings['mode']}**.",
        author="System",
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    mode = cl.user_session.get("mode", "auto")

    # Placeholder message — tokens will stream into it
    msg = cl.Message(content="")
    await msg.send()

    sources = []
    agent_type = "unknown"

    try:
        payload = {
            "query": message.content,
            "context": {},
            "mode": mode,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{BACKEND_URL}/api/query/stream",
                json=payload,
            ) as response:
                if response.status_code != 200:
                    msg.content = (
                        "**Something went wrong on my end.**\n\n"
                        "The backend returned an unexpected response. "
                        "Please try again in a moment."
                    )
                    await msg.update()
                    return

                async for line in response.aiter_lines():
                    etype, event = handle_streamed_response(line)
                    await handle_event_type(etype, msg, event)

        # Attach source documents as side-panel elements
        elements = []
        for i, src in enumerate(sources):
            source_text = src.get("text") or "No content available."
            source_name = src.get("source") or f"Document {i + 1}"
            score = src.get("score") or src.get("relevance")

            label = source_name
            if score is not None:
                label += f"  (score: {score:.2f})"

            elements.append(
                cl.Text(
                    name=label,
                    content=source_text,
                    display="side",
                )
            )

        if elements:
            msg.elements = elements

        await msg.update()

        # Footer note showing which assistant answered
        if agent_type != "unknown":
            icon = "🔧" if agent_type == "technical" else "📄"
            await cl.Message(
                content=f"{icon} *Answered by the **{agent_type}** assistant.*",
                author="System",
            ).send()

    except httpx.ConnectError:
        msg.content = (
            "🔌 **Could not reach the backend.**\n\n"
            f"Make sure the backend service is running at `{BACKEND_URL}` and try again."
        )
        await msg.update()
    except httpx.TimeoutException:
        msg.content = (
            "⏱️ **Request timed out.**\n\n"
            "The model is taking longer than expected — it may still be loading. "
            "Please try again in a few moments."
        )
        await msg.update()
    except Exception:
        msg.content = (
            "⚠️ **An unexpected error occurred.**\n\n"
            "Please try again. If the problem persists, check the service logs."
        )
        await msg.update()
