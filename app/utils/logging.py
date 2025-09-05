from __future__ import annotations

import json
import logging
import os
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        """Format log record as a JSON line with standard and extra fields."""
        data: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
        }
        # Include extras if present
        for key in ("extra",):
            if hasattr(record, key):
                value = getattr(record, key)
                if isinstance(value, dict):
                    data.update(value)
        # Also include custom attributes attached to record via logger.*(extra={...})
        # The logging module injects them directly on the record; include simple JSON-able ones.
        for k, v in record.__dict__.items():
            if k in {"name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs", "relativeCreated", "thread", "threadName", "processName", "process", "message", "asctime"}:
                continue
            if k.startswith("_" ):
                continue
            try:
                json.dumps(v)
            except Exception:
                continue
            data[k] = v
        return json.dumps(data, ensure_ascii=False)


def configure_json_logging(level: str | int | None = None) -> None:  # noqa: D401
    """Configure root logging to JSON format for the whole process."""
    root = logging.getLogger()
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    root.setLevel(level)  # type: ignore[arg-type]
    # Clear existing handlers to avoid duplicate logs in uvicorn
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)

