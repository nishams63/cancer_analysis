import logging
import re
import sys
from typing import Any

# Secret redaction patterns
P1 = re.compile(r"nvapi-[A-Za-z0-9_-]{20,}", re.IGNORECASE)
P2 = re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{20,}", re.IGNORECASE)
P3 = re.compile(r"""(api[_-]?key|secret|password)\s*[:=]\s*([^\s,;]+)""", re.IGNORECASE)

SECRET_PATTERNS = [P1, P2, P3]

def redact_secrets(text: Any) -> str:
    if not isinstance(text, str):
        text = str(text)
    for pat in SECRET_PATTERNS:
        text = pat.sub("[REDACTED_SECRET]", text)
    return text

class SecretRedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        orig = super().format(record)
        return redact_secrets(orig)

def get_logger(name: str = "Stage5Integration", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        fmt = SecretRedactingFormatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger
