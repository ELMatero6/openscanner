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

# Required - fail if any of these are missing.
apt-get install -y --no-install-recommends \
    git python3 python3-pip python3-numpy \
    python3-opencv python3-rpi.gpio \
    libgl1 \
    dosfstools udisks2 xvfb

# Optional - tolerate missing ones (names drift across Pi OS releases):
#   libglib2.0-0 is now libglib2.0-0t64 on Bookworm/Trixie (apt auto-selects)
#   libatlas-base-dev is gone on Trixie (not needed; modern numpy ships its own BLAS)
for pkg in libglib2.0-0 libglib2.0-0t64 libatlas3-base libopenblas0; do
    apt-get install -y --no-install-recommends "$pkg" 2>/dev/null || true
done

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
    sudo -u "$RUN_user" git -C "$INSTALL_DIR" reset --quiet --hard "origin/$BRANCH"
else
    echo "[install] cloning $REPO_URL into $INSTALL_DIR..."
    mkdir -p "$INSTALL_DIR"
    chown "$RUN_user:$RUN_user" "$INSTALL_DIR"
    sudo -u "$RUN_user" git clone --quiet --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
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
Group=$RUN_USER
WorkingDirectory=$INSTALL_DIR
ExecStart=/usr/bin/env bash $INSTALL_DIR/update.sh
Restart=on-failure
RestartSec=5
# Removed GUI environment variables to support headless operation

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable openscanner.service

# Allow the scanner user to shut down + format USB drives without a password.
SUDOERS=/etc/sudoers.d/openscanner
cat > "$SUDOERS" <<EOF
$RUN_user ALL=(root) NOPASSWD: /sbin/shutdown, /usr/sbin/mkfs.vfat, /sbin/mkfs.vfat, /bin/umount, /usr/bin/eject
EOF
chmod 440 "$SUDOERS"
echo "[install] sudoers: $RUN_user may shutdown / mkfs.vfat / umount / eject without password"

# Stop the desktop file-manager from popping up a window every time a USB
# drive is plugged in - that popup steals focus from the scanner's
# fullscreen window and makes it look frozen.
# Profiles: LXDE-pi (Bookworm), LXDE (Bullseye), default (labwc/Wayland)
for profile in LXDE-pi LXDE default; do
    PCMANFM_DIR="/home/$RUN_user/.config/pcmanfm/$profile"
    mkdir -p "$PCMANFM_DIR"
    cat > "$PCMANFM_DIR/pcmanfm.conf" <<EOF
[config]
bm_open_method=0

[volume]
mount_on_startup=0
mount_removable=0
autorun=0

[ui]
always_show_tabs=0
EOF
done
chown -R "$RUN_user:$RUN_user" "/home/$RUN_user/.config/pcmanfm"
echo "[install] pcmanfm: auto-mount popup disabled for LXDE-pi/LXDE/default profiles"

# Block pcmanfm --desktop from autostarting (it's what shows the popup)
AUTOSTART_DIR="/home/$RUN_user/.config/autostart"
mkdir -p "$AUTOSTART_DIR"
cat > "$AUTOSTART_DIR/pcmanfm.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=pcmanfm (disabled)
Exec=true
Hidden=true
X-GNOME-Autostart-enabled=false
EOF
chown -R "$RUN_user:$RUN_user" "$AUTOSTART_DIR"
echo "[install] pcmanfm desktop autostart blocked via $AUTOSTART_DIR"

echo
echo "[install] done!"
echo "  - run now:    sudo systemctl start openscanner"
echo "  - logs:       journalctl -u openscanner -f"
echo "  - disable:    sudo systemctl disable openscanner"
echo "  - update.sh runs at every start; toggle auto_update in .settings to opt out"
