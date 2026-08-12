#!/usr/bin/env python3
"""Continue the Codex slim-core queue while auto mode is armed."""

import datetime
import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[2]
AUTO = ROOT / ".codex" / "slim-core-auto.on"
QUEUE = ROOT / ".codex" / "slim-core-queue.md"
LOG = ROOT / ".codex" / "slim-core-auto.log"


def emit(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False))


def log(message: str) -> None:
    timestamp = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()
    try:
        with LOG.open("a", encoding="utf-8") as stream:
            stream.write(f"{timestamp} {message}\n")
    except OSError:
        pass


try:
    event = json.load(sys.stdin)
except (json.JSONDecodeError, OSError):
    emit({})
    raise SystemExit(0)

if not AUTO.is_file() or not QUEUE.is_file():
    emit({})
    raise SystemExit(0)

if event.get("stop_hook_active"):
    log(f"skip active continuation turn_id={event.get('turn_id', '')}")
    emit({})
    raise SystemExit(0)

pending = sum(line.startswith("- [ ]") for line in QUEUE.read_text(encoding="utf-8").splitlines())
if pending == 0:
    AUTO.unlink(missing_ok=True)
    log("queue empty; disarmed auto")
    emit({})
    raise SystemExit(0)

reason = (
    "Use $slim-core-script and process only the next unchecked file in "
    ".codex/slim-core-queue.md. Mark it complete only after mandatory checks, "
    "report the result, and stop without opening another file."
)
log(f"continue pending={pending} turn_id={event.get('turn_id', '')}")
emit({"decision": "block", "reason": reason})
