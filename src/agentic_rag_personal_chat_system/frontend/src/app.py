"""Chainlit frontend for the Agentic RAG Personal Chat System."""

import os

import chainlit as cl
import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")


@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.Message(
        content="👋 Hello! I'm your Agentic RAG assistant. Ask me anything.",
        author="Assistant",
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    thinking = cl.Message(content="", author="Assistant")
    await thinking.send()

    try:
        resp = requests.post(
            f"{BACKEND_URL}/chat",
            json={"query": message.content, "mode": "auto"},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("response", "No response received.")
        agent_type = data.get("agent_type", "")
        if agent_type:
            answer = f"{answer}\n\n*Agent: {agent_type}*"
    except requests.exceptions.ConnectionError:
        answer = "⚠️ Cannot reach the backend. Is it running?"
    except Exception as e:
        answer = f"⚠️ Error: {e}"

    thinking.content = answer
    await thinking.update()
