"""Process-wide logging setup: plain text for local runs, JSON lines for log aggregators."""

from __future__ import annotations

import json
import logging
import sys

_NOISY = ("httpx", "httpcore", "huggingface_hub", "fsspec", "urllib3")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", fmt: str = "text") -> None:
    """Install one stdout handler on the root logger unless something already configured it."""
    root = logging.getLogger()
    if root.handlers:  # e.g. pytest's capture handler, or an embedding app's own config
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter() if fmt == "json" else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(level.upper())
    for name in _NOISY:  # request URLs / retries; our own logs carry what matters
        logging.getLogger(name).setLevel(logging.WARNING)
