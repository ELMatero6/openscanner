#!/usr/bin/env bash
# openscanner installer - sudo required.
#
# Pulls the repo into /opt/openscanner, installs python deps,
# installs a systemd unit so the scanner starts on boot.
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/elmatero6/openscanner/main/install.sh | sudo bash
# or
#   sudo ./install.sh
#
# Re-running is safe (idempotent).

set -euo pipefail

REPO_URL="${OPENSCANNER_REPO:-https://github.com/elmatero6/openscanner.git}"
BRANCH="${OPENSCANNER_BRANCH:-main}"
INSTALL_DIR="/opt/openscanner"
SERVICE="/etc/systemd/system/openscanner.service"
RUN_USER="${SUDO_USER:-pi}"

if [[ $EUID -ne 0 ]]; then
    echo "[install] must be run as root (sudo)"
    exit 1
fi

echo "[install] target user: $RUN_USER"
echo "[install] target dir:  $INSTALL_DIR"
echo "[install] branch:      $BRANCH"

echo "[install] apt: installing system packages..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    git python3 python3-pip python3-numpy \
    python3-opencv python3-rpi.gpio \
    libatlas-base-dev libgl1 libglib2.0-0

# Optional: opencv-contrib for WLS filter. Only install if not already present.
if ! python3 -c "import cv2; cv2.ximgproc.createDisparityWLSFilter" >/dev/null 2>&1; then
    echo "[install] pip: installing opencv-contrib-python (for WLS)..."
    pip3 install --break-system-packages opencv-contrib-python || \
        echo "[install] WARN: opencv-contrib install failed - WLS filter will be disabled"
fi

if [[ -d "$INSTALL_DIR/.git" ]]; then
    echo "[install] repo exists, fetching latest..."
    sudo -u "$RUN_USER" git -C "$INSTALL_DIR" fetch --quiet origin
    sudo -u "$RUN_USER" git -C "$INSTALL_DIR" checkout --quiet "$BRANCH"
    sudo -u "$RUN_USER" git -C "$INSTALL_DIR" reset --quiet --hard "origin/$BRANCH"
else
    echo "[install] cloning $REPO_URL into $INSTALL_DIR..."
    mkdir -p "$INSTALL_DIR"
    chown "$RUN_USER:$RUN_USER" "$INSTALL_DIR"
    sudo -u "$RUN_USER" git clone --quiet --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
fi

chmod +x "$INSTALL_DIR/update.sh" "$INSTALL_DIR/install.sh"

echo "[install] writing systemd unit at $SERVICE..."
cat > "$SERVICE" <<EOF
[Unit]
Description=openscanner stereo 3D scanner
After=multi-user.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$INSTALL_DIR
ExecStart=/usr/bin/env bash $INSTALL_DIR/update.sh
Restart=on-failure
RestartSec=5
# Allow access to framebuffer + X server if running under desktop
Environment=DISPLAY=:0
Environment=XAUTHORITY=/home/$RUN_USER/.Xauthority

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable openscanner.service

# Allow the scanner user to shut the system down without a password.
SUDOERS=/etc/sudoers.d/openscanner
if [[ ! -f "$SUDOERS" ]]; then
    echo "$RUN_USER ALL=(root) NOPASSWD: /sbin/shutdown" > "$SUDOERS"
    chmod 440 "$SUDOERS"
    echo "[install] sudoers: $RUN_USER may shutdown without password"
fi

echo
echo "[install] done!"
echo "  - run now:    sudo systemctl start openscanner"
echo "  - logs:       journalctl -u openscanner -f"
echo "  - disable:    sudo systemctl disable openscanner"
echo "  - update.sh runs at every start; toggle auto_update in .settings to opt out"
