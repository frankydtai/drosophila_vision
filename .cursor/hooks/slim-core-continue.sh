#!/usr/bin/env bash
# Auto-continue slim-core queue when `.cursor/slim-core-auto.on` exists.
# File contents = bound conversation_id (one line). Other chats get {}.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
AUTO="$ROOT/.cursor/slim-core-auto.on"
QUEUE="$ROOT/.cursor/skills/slim-core-script/slim-core-queue.md"
LOG="$ROOT/.cursor/slim-core-auto.log"

input="$(cat)"
noop() { echo '{}'; exit 0; }
log() { echo "$(date -Iseconds) $*" >>"$LOG" 2>/dev/null || true; }

[[ -f "$AUTO" ]] || noop
[[ -f "$QUEUE" ]] || noop

bound="$(tr -d '[:space:]' <"$AUTO")"
[[ -n "$bound" ]] || noop

parsed="$(printf '%s' "$input" | python3 -c '
import json,sys
d=json.load(sys.stdin)
status=d.get("status") or ""
cid=d.get("conversation_id") or d.get("session_id") or ""
loop=d.get("loop_count", "")
# transcript basename often equals conversation_id; keep for debug
tp=d.get("transcript_path") or ""
print(f"{status}\t{cid}\t{loop}\t{tp}")
' 2>/dev/null || true)"
IFS=$'\t' read -r status cid loop_count transcript_path <<<"$parsed"

match=0
[[ "$cid" == "$bound" ]] && match=1
# Also accept binding by agent-transcript folder uuid if path contains it.
if [[ "$match" -eq 0 && -n "$bound" && "$transcript_path" == *"$bound"* ]]; then
  match=1
fi

log "stop status=$status loop_count=$loop_count cid=$cid bound=$bound match=$match"

[[ "$status" == "completed" ]] || noop
[[ "$match" -eq 1 ]] || noop

pending="$(grep -cE '^- \[ \]' "$QUEUE" || true)"
if [[ "${pending:-0}" -lt 1 ]]; then
  rm -f "$AUTO"
  log "queue empty; disarmed auto"
  noop
fi

msg='用 slim-core-script skill，只處理 .cursor/skills/slim-core-script/slim-core-queue.md 裡下一個未完成的檔案：內聯單次使用 local；禁止新造名詞（僅 lexicon / 檔內既有名）。做完勾選後結束本輪（不要開下一檔；auto hook 會繼續）。'
python3 -c 'import json,sys; print(json.dumps({"followup_message": sys.argv[1]}))' "$msg"
log "emit followup pending=$pending loop_count=$loop_count cid=$cid"
exit 0
