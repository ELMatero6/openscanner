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

# Trust this repo even if ownership is mixed. Git 2.35+ rejects repos
# owned by a different uid than the caller unless they're on the
# safe.directory list. This is the single most common reason the
# updater silently fails under systemd. Idempotent.
git config --global --add safe.directory "$PWD" 2>/dev/null || true

# Disable update? Look for "auto_update": false in .settings (no jq dep).
if [[ -f "$SETTINGS" ]] && grep -q '"auto_update"[[:space:]]*:[[:space:]]*false' "$SETTINGS"; then
    echo "[update] auto_update disabled in $SETTINGS - skipping pull"
else
    echo "[update] checking for updates..."
    # stderr -> stdout so systemd journal captures the real git message
    # (not just our "pull failed" summary). See: journalctl -u openscanner.
    if timeout 10 git fetch origin 2>&1; then
        BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
        BEHIND="$(git rev-list --count "HEAD..origin/$BRANCH" 2>/dev/null || echo 0)"
        if [[ "$BEHIND" == "0" ]]; then
            echo "[update] already at origin/$BRANCH @ $(git rev-parse --short HEAD)"
        elif timeout 10 git pull --ff-only origin "$BRANCH" 2>&1; then
            echo "[update] pulled $BEHIND commit(s) - now at $(git rev-parse --short HEAD)"
        else
            # --ff-only refuses if the working tree is dirty. Show what's dirty
            # so the user can see it in `journalctl -u openscanner`.
            DIRTY="$(git status --porcelain 2>/dev/null | head -20)"
            if [[ -n "$DIRTY" ]]; then
                echo "[update] pull blocked by local changes:"
                echo "$DIRTY" | sed 's/^/[update]   /'
                echo "[update] run 'sudo git -C $PWD reset --hard origin/$BRANCH' to force update"
            else
                echo "[update] pull failed (not a dirty-tree issue) - running cached version"
            fi
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
