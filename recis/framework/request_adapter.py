from typing import Any, Callable, Optional


RequestAdapter = Callable[[dict[str, Any]], dict[str, Any]]


def adapt_request_payload(
    payload: dict[str, Any],
    request_adapter: Optional[RequestAdapter] = None,
) -> dict[str, Any]:
    if request_adapter is None:
        return payload
    return request_adapter(payload)
