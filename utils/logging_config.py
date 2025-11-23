# logging_config.py
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

LOG_PATH = Path("./logs")
LOG_PATH.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_PATH / "system.jsonl"


class JSONFormatter(logging.Formatter):
    """
    Emit a single-line JSON object for each log record.
    Fields: timestamp, level, logger, message, extra (if present).
    """
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.utcfromtimestamp(record.created).isoformat() + "Z"
        payload: Dict[str, Any] = {
            "timestamp": ts,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include extra attributes (exclude builtin ones) as 'meta'
        extras = {}
        for k, v in record.__dict__.items():
            if k in ("name", "msg", "args", "levelname", "levelno", "pathname",
                     "filename", "module", "exc_info", "exc_text", "stack_info",
                     "lineno", "funcName", "created", "msecs", "relativeCreated",
                     "thread", "threadName", "processName", "process"):
                continue
            # Avoid serializing unserializable objects
            try:
                json.dumps({k: v})
                extras[k] = v
            except Exception:
                extras[k] = str(v)
        if extras:
            payload["meta"] = extras

        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str = "multiagent") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setFormatter(JSONFormatter())
        logger.addHandler(fh)

        # Optional: also stream minimal JSON to stdout for local debugging
        sh = logging.StreamHandler()
        sh.setFormatter(JSONFormatter())
        logger.addHandler(sh)
    return logger