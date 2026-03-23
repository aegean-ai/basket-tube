"""Broadcast event bus for SSE pipeline progress.

Supports multiple concurrent subscribers via an append-only event log
with per-subscriber cursors. Events are never removed — late subscribers
replay from any position.
"""

import asyncio
from typing import AsyncIterator


class EventBus:
    """Broadcast event bus supporting multiple concurrent SSE consumers."""

    def __init__(self) -> None:
        self._events: list[dict] = []
        self._notify = asyncio.Condition()

    @property
    def size(self) -> int:
        return len(self._events)

    async def emit(self, event: dict) -> None:
        """Append an event and wake all waiting subscribers."""
        self._events.append(event)
        async with self._notify:
            self._notify.notify_all()

    async def subscribe(self, cursor: int = 0) -> AsyncIterator[dict]:
        """Yield events starting from *cursor*, waiting for new ones."""
        while True:
            while cursor < len(self._events):
                yield self._events[cursor]
                cursor += 1
            async with self._notify:
                await self._notify.wait()
