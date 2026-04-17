"""Entrypoint - delegates to scanner.main.

The actual code lives in scanner/. This file exists so that
`python3 pi_scanner.py` and the systemd unit both keep working
without reaching into a package path.
"""

from scanner.main import run

if __name__ == "__main__":
    run()
