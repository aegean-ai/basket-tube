import asyncio
import pytest
from api.src.services.event_bus import EventBus


@pytest.mark.asyncio
async def test_emit_and_subscribe():
    bus = EventBus()
    await bus.emit({"type": "a"})
    await bus.emit({"type": "b"})

    events = []
    async for evt in bus.subscribe(cursor=0):
        events.append(evt)
        if len(events) == 2:
            break
    assert events == [{"type": "a"}, {"type": "b"}]


@pytest.mark.asyncio
async def test_multiple_subscribers_receive_all_events():
    bus = EventBus()
    await bus.emit({"type": "first"})

    results_a = []
    results_b = []

    async def consume(results, cursor=0):
        async for evt in bus.subscribe(cursor=cursor):
            results.append(evt)
            if len(results) == 2:
                break

    task_a = asyncio.create_task(consume(results_a))
    task_b = asyncio.create_task(consume(results_b))
    await asyncio.sleep(0.01)
    await bus.emit({"type": "second"})
    await asyncio.gather(task_a, task_b)

    assert results_a == [{"type": "first"}, {"type": "second"}]
    assert results_b == [{"type": "first"}, {"type": "second"}]


@pytest.mark.asyncio
async def test_subscribe_waits_for_new_events():
    bus = EventBus()
    received = []

    async def consumer():
        async for evt in bus.subscribe():
            received.append(evt)
            if evt.get("type") == "done":
                break

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.01)
    assert received == []

    await bus.emit({"type": "hello"})
    await asyncio.sleep(0.01)
    assert received == [{"type": "hello"}]

    await bus.emit({"type": "done"})
    await task
    assert len(received) == 2


@pytest.mark.asyncio
async def test_replay_from_cursor():
    bus = EventBus()
    await bus.emit({"type": "a"})
    await bus.emit({"type": "b"})
    await bus.emit({"type": "c"})

    events = []
    async for evt in bus.subscribe(cursor=1):
        events.append(evt)
        if len(events) == 2:
            break
    assert events == [{"type": "b"}, {"type": "c"}]


@pytest.mark.asyncio
async def test_size_property():
    bus = EventBus()
    assert bus.size == 0
    await bus.emit({"type": "a"})
    assert bus.size == 1
    await bus.emit({"type": "b"})
    assert bus.size == 2
