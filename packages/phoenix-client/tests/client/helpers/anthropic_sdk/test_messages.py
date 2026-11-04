from __future__ import annotations

from typing import Any

import pytest

from phoenix.client.helpers.sdk.anthropic.messages import (
    create_prompt_version_from_anthropic,
    to_chat_messages_and_kwargs,
)

_FUNCTION_TOOL: dict[str, Any] = {
    "name": "get_weather",
    "input_schema": {"type": "object", "properties": {}},
}
_CUSTOM_TOOL: dict[str, Any] = {"type": "custom", **_FUNCTION_TOOL}
_SERVER_TOOL: dict[str, Any] = {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}


def _round_trip(tools: list[dict[str, Any]]) -> dict[str, Any]:
    obj: Any = {
        "model": "claude-x",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": tools,
        "tool_choice": {"type": "any"},
    }
    _, kwargs = to_chat_messages_and_kwargs(create_prompt_version_from_anthropic(obj))
    return dict(kwargs)


@pytest.mark.parametrize(
    "tools",
    [
        pytest.param([_SERVER_TOOL], id="server-only"),
        pytest.param([_FUNCTION_TOOL, _SERVER_TOOL], id="function-and-server"),
        pytest.param([_CUSTOM_TOOL], id="custom"),
    ],
)
def test_tools_with_type_key_are_kept(tools: list[dict[str, Any]]) -> None:
    kwargs = _round_trip(tools)
    assert [t["name"] for t in kwargs["tools"]] == [t["name"] for t in tools]
    assert kwargs["tool_choice"] == {"type": "any"}
