"""mini-SWE-agent Model implementation backed by the Conditional IS service."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from hashlib import blake2b
from typing import Any


DEFAULT_OBSERVATION_TEMPLATE = (
    "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
    "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
)


class ConditionalISModel:
    """Thin mini-SWE-agent adapter; generation remains in the persistent service."""

    def __init__(
        self,
        *,
        model_name: str = "conditional-is/Qwen3-Coder-30B-A3B-Instruct",
        endpoint: str = "http://127.0.0.1:8123",
        timeout_seconds: float = 3600.0,
        retries: int = 2,
        seed: int = 20260908,
        observation_template: str = DEFAULT_OBSERVATION_TEMPLATE,
        format_error_template: str = "{{ error }}",
        **_: Any,
    ) -> None:
        if timeout_seconds <= 0 or retries < 0 or seed < 0:
            raise ValueError("invalid Conditional IS client retry, timeout, or seed")
        self.config = {
            "model_name": model_name,
            "endpoint": endpoint.rstrip("/"),
            "timeout_seconds": timeout_seconds,
            "retries": retries,
            "seed": seed,
            "observation_template": observation_template,
            "format_error_template": format_error_template,
        }

    def _identity(self, messages: list[dict[str, Any]]) -> tuple[str, int]:
        canonical = json.dumps(
            messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest = blake2b(canonical, digest_size=16).hexdigest()
        seed_material = blake2b(
            canonical,
            digest_size=8,
            key=int(self.config["seed"]).to_bytes(16, "little"),
        ).digest()
        seed = int.from_bytes(seed_material, "little") & ((1 << 63) - 1)
        return f"mini-swe-agent:{digest}", seed

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.config['endpoint']}/v1/query",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(
            request, timeout=float(self.config["timeout_seconds"])
        ) as response:
            decoded = json.loads(response.read().decode("utf-8"))
        if not isinstance(decoded, dict):
            raise RuntimeError("Conditional IS service returned a non-object response")
        return decoded

    def query(self, messages: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        request_id, seed = self._identity(messages)
        payload = {"messages": messages, "request_id": request_id, "seed": seed}
        response: dict[str, Any] | None = None
        for attempt in range(int(self.config["retries"]) + 1):
            try:
                response = self._post(payload)
                break
            except (urllib.error.URLError, TimeoutError):
                if attempt >= int(self.config["retries"]):
                    raise
                time.sleep(0.5 * (2**attempt))
        assert response is not None
        message = response.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("Conditional IS service response omitted 'message'")
        if not message.get("extra", {}).get("actions"):
            from minisweagent.exceptions import FormatError

            finish_reason = (
                message.get("extra", {})
                .get("conditional_is", {})
                .get("finish_reason", "unknown")
            )
            raise FormatError(
                {
                    "role": "user",
                    "content": (
                        "No bash tool call was found in the selected Conditional IS "
                        f"completion (finish_reason={finish_reason}). Respond with one bash tool call."
                    ),
                    "extra": {
                        "interrupt_type": "FormatError",
                        "response": response,
                        "cost": 0.0,
                    },
                }
            )
        return message

    def format_message(self, **kwargs: Any) -> dict[str, Any]:
        return dict(kwargs)

    def format_observation_messages(
        self,
        message: dict[str, Any],
        outputs: list[dict[str, Any]],
        template_vars: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        from minisweagent.models.utils.actions_toolcall import (
            format_toolcall_observation_messages,
        )

        return format_toolcall_observation_messages(
            actions=message.get("extra", {}).get("actions", []),
            outputs=outputs,
            observation_template=str(self.config["observation_template"]),
            template_vars=template_vars,
        )

    def get_template_vars(self, **_: Any) -> dict[str, Any]:
        return dict(self.config)

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "config": {
                    "model": dict(self.config),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }


__all__ = ["ConditionalISModel"]
