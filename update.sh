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

echo "[update] killing any file manager windows that could steal focus..."
# pcmanfm / nautilus / nemo pop up on USB insert and cover our fullscreen
# window, which makes the scanner look frozen. Best-effort cleanup here;
# we re-kill them whenever a popup slips through via xdotool during run.
for prog in pcmanfm pcmanfm-desktop nautilus nemo thunar; do
    pkill -f "$prog" 2>/dev/null || true
done

echo "[update] launching pi_scanner.py..."
exec "$PYTHON" pi_scanner.py
