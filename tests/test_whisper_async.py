import pytest
from api.src.services.whisper_service import transcribe


@pytest.mark.asyncio
async def test_transcribe_is_async():
    """Verify transcribe is a coroutine function."""
    import inspect
    assert inspect.iscoroutinefunction(transcribe)
