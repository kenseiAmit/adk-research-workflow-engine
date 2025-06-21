"""Shared helpers for executing ADK agents through the official ADK Runner.

Every workflow stage is driven by the ADK runtime (Runner + session service) so the
framework's composite agents control iteration and stage ordering, rather than
hand-written Python control flow.
"""

import json
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional

from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

APP_NAME = "ai_research_assistant"
USER_ID = "researcher"


class _AppNameMismatchFilter(logging.Filter):
    """Drop ADK's cosmetic app-name notice; agents are imported, not CLI-loaded."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "App name mismatch" not in record.getMessage()


logging.getLogger("google_adk.google.adk.runners").addFilter(_AppNameMismatchFilter())


async def run_adk_agent(
    agent: BaseAgent,
    prompt: str,
    on_event: Optional[Callable[[Event], None]] = None,
) -> Dict[str, Any]:
    """Execute an ADK agent (or composed workflow) via the ADK Runner.

    Args:
        agent: Any ADK agent - LlmAgent, LoopAgent, ParallelAgent or SequentialAgent
        prompt: The user message that seeds the invocation
        on_event: Optional callback invoked for each streamed ADK event

    Returns:
        Dictionary with the streamed ``events`` and the final session ``state``
    """
    session_service = InMemorySessionService()
    session_id = f"session-{uuid.uuid4().hex[:8]}"

    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )

    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

    events: List[Event] = []
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=Content(role="user", parts=[Part(text=prompt)]),
    ):
        events.append(event)
        if on_event is not None:
            on_event(event)

    session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )

    return {"events": events, "state": dict(session.state)}


def event_text(event: Event) -> Optional[str]:
    """Return the text payload of an ADK event, if it carries one."""
    if event.content and event.content.parts:
        return event.content.parts[0].text
    return None


def parse_agent_json(raw: Any, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Parse an agent's JSON output, falling back to a default shape on failure."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return dict(fallback)
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return dict(fallback)
    return parsed if isinstance(parsed, dict) else dict(fallback)
