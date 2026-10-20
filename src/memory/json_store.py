
from __future__ import annotations

import json

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonlRunStore:
    """
    Minimal JSONL-based run store.

    Each call to append_run writes one JSON object per line. 

    This is useful for lightweight traceability without requiring a database.
    """

    def __init__(self, path: str | Path = "logs/runs.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append_run(self, record: dict[str, Any]) -> None:
        payload = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **record}

        with self.path.open("a", encoding = "utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii = False) + "\n")
