from __future__ import annotations

import pytest
from app.config import get_settings, verify_internal_token
from app.errors import InternalAuthError


@pytest.mark.asyncio
async def test_internal_token_is_required() -> None:
    expected = get_settings().internal_token.get_secret_value()
    await verify_internal_token(expected)
    with pytest.raises(InternalAuthError):
        await verify_internal_token("wrong-token")
