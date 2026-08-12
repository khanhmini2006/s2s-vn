#!/usr/bin/env bash
# Test chạy local (serve + talk in-process) với một config file.
#
# Cách dùng:
#   ./scripts/test_local.sh                          # local với config-local.json (mặc định)
#   ./scripts/test_local.sh config-gemini.json       # local với config-gemini.json
#   ./scripts/test_local.sh config-local.json 8765   # + port
#
# Cần mic/loa (sounddevice) — chạy trong terminal thật.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

[[ -x "$HOME/miniconda3/bin/python" ]] && export PATH="$HOME/miniconda3/bin:$PATH"
if ! command -v s2s-vn >/dev/null 2>&1; then
  echo "❌ thiếu s2s-vn — pip install -e .[realtime]" >&2
  exit 1
fi

CONFIG="${1:-config-local.json}"
PORT="${2:-8765}"

if [[ ! -f "$CONFIG" ]]; then
  echo "❌ Không thấy file config: $CONFIG" >&2
  echo "   Có: $(ls config-*.json 2>/dev/null | tr '\n' ' ')" >&2
  exit 1
fi

echo "=== Test s2s-vn local — config: $CONFIG — port: $PORT ==="
echo "    (server tự lên loopback, rồi mở mic nói chuyện — Ctrl+C để thoát)"
s2s-vn local --config "$CONFIG" --port "$PORT"
