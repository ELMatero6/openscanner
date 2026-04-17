"""Centralised logging setup.

Writes to both stdout (so systemd journal still captures everything)
and a rotating file in SAVE_DIR. The file lives beside the captures
so it ships automatically with every zip/USB export.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

_FMT = logging.Formatter(
    "%(asctime)s %(levelname)-5s %(name)-22s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_configured = False


def init(save_dir, level="INFO"):
    """Set up root logger once. Safe to call multiple times; only first wins."""
    global _configured
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "openscanner.log")

    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    if _configured:
        return log_path

    for h in list(root.handlers):
        root.removeHandler(h)

    console = logging.StreamHandler()
    console.setFormatter(_FMT)
    root.addHandler(console)

    fh = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3)
    fh.setFormatter(_FMT)
    root.addHandler(fh)

    _configured = True
    return log_path


def get(name):
    return logging.getLogger(name)
