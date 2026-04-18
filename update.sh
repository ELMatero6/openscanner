#!/usr/bin/env bash
# openscanner update + launch wrapper.
#
# Behaviour:
#   1. If .settings has auto_update: false, skip git pull entirely.
#   2. Otherwise, try a fast-forward-only pull with a short timeout.
#      Network down / merge conflict / any error -> stay on cached version.
#   3. Always exec pi_scanner.py at the end, even if pull failed.
#
# Run by systemd; do NOT add interactive prompts here.

set -uo pipefail

cd "$(dirname "$0")"
SETTINGS=".settings"
PYTHON="$(command -v python3)"
LOG="/tmp/openscanner-update.log"

# Disable update? Look for "auto_update": false in .settings (no jq dep).
if [[ -f "$SETTINGS" ]] && grep -q '"auto_update"[[:space:]]*:[[:space:]]*false' "$SETTINGS"; then
    echo "[update] auto_update disabled in $SETTINGS - skipping pull"
else
    echo "[update] checking for updates..."
    # 10s timeout total. --ff-only = refuse to merge unrelated histories.
    if timeout 10 git fetch --quiet origin 2>>"$LOG"; then
        BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
        if timeout 10 git pull --quiet --ff-only origin "$BRANCH" 2>>"$LOG"; then
            echo "[update] up to date on $BRANCH @ $(git rev-parse --short HEAD)"
        else
            echo "[update] pull failed - running cached version"
        fi
    else
        echo "[update] fetch failed (offline?) - running cached version"
    fi
fi

# Default to KMSDRM if nothing else is set. systemd unit also exports these,
# but having them here lets `bash update.sh` work standalone for testing.
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-kmsdrm}"
export SDL_FBDEV="${SDL_FBDEV:-/dev/fb0}"
export PYGAME_HIDE_SUPPORT_PROMPT=1

echo "[update] launching pi_scanner.py (SDL_VIDEODRIVER=$SDL_VIDEODRIVER)..."
exec "$PYTHON" pi_scanner.py
