import json
from typing import Any, Optional, Tuple

from chainlit.message import Message


async def handle_event_type(etype: Any, msg: Message, event: Any) -> None:
    if etype is None and event is None:
        return
    if etype == "token":
        await msg.stream_token(event.get("content", ""))
    # elif etype == "agent_type":
    #     agent_type = event.get("content", "unknown")
    # elif etype == "sources":
    #     sources = event.get("sources", [])
    elif etype == "error":
        await msg.stream_token(f"\n\n{event.get('content', 'An error occurred.')}")


def handle_streamed_response(line: str) -> Optional[Tuple[Any, Any]]:
    if line.startswith("data: "):
        raw = line[6:].strip()
        if raw == "[DONE]":
            return None, None

        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            return None, None

        etype = event.get("type")
        return etype, event
