"""Persistent user settings (atomic JSON file)."""

import json
import os
import tempfile

DEFAULTS = {
    "dist_mode":      "SMALL",
    "bg_on":          False,
    "auto_update":    True,
    "screen_rotate":  0,
    "last_save_dest": "local",
    "viewer_subsample": 60000,
    "log_level":      "INFO",
}


def load(path):
    if not os.path.exists(path):
        return dict(DEFAULTS)
    try:
        with open(path) as f:
            data = json.load(f)
        out = dict(DEFAULTS)
        out.update({k: v for k, v in data.items() if k in DEFAULTS})
        return out
    except Exception as e:
        print(f"[SETTINGS] read failed ({e}), using defaults")
        return dict(DEFAULTS)


def save(path, data):
    """Atomic write: tmp file in same dir + rename."""
    try:
        d = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp = tempfile.mkstemp(prefix=".settings.", dir=d)
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        print(f"[SETTINGS] write failed: {e}")
