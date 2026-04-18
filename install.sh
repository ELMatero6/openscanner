#!/usr/bin/env bash
# openscanner installer - sudo required.
#
# Targets Raspberry Pi OS Lite (Bookworm/Trixie). Runs without a desktop:
# pygame uses SDL2's KMSDRM backend to draw straight to the framebuffer,
# so there's no X/Wayland dependency. If you happen to be on Pi OS with a
# desktop SDL will pick up that compositor instead and it still works.
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

# Required packages. python3-pygame brings SDL2; libcamera/v4l for the
# stereo camera; dosfstools+udisks2 for USB export.
apt-get install -y --no-install-recommends \
    git python3 python3-pip python3-numpy \
    python3-opencv python3-rpi.gpio python3-pygame \
    libsdl2-2.0-0 libsdl2-image-2.0-0 \
    libgl1 v4l-utils \
    dosfstools udisks2

# Optional - tolerate missing ones (names drift across Pi OS releases).
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
    chown -R "$RUN_USER:$RUN_USER" "$INSTALL_DIR"
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

# KMSDRM needs the user in 'video' (framebuffer) and 'render' (GPU) groups,
# and 'input' for the touchscreen. Idempotent.
for grp in video render input tty; do
    if getent group "$grp" >/dev/null; then
        usermod -a -G "$grp" "$RUN_USER"
    fi
done
echo "[install] $RUN_USER added to video/render/input/tty groups"

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
# pygame + SDL2: draw straight to KMS, no X/Wayland needed
Environment=SDL_VIDEODRIVER=kmsdrm
Environment=SDL_FBDEV=/dev/fb0
Environment=PYGAME_HIDE_SUPPORT_PROMPT=1
# Needed so KMSDRM can grab the active TTY
TTYPath=/dev/tty1
StandardInput=tty
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable openscanner.service

# Allow the scanner user to shut down + format USB drives without a password.
SUDOERS=/etc/sudoers.d/openscanner
cat > "$SUDOERS" <<EOF
$RUN_USER ALL=(root) NOPASSWD: /sbin/shutdown, /usr/sbin/mkfs.vfat, /sbin/mkfs.vfat, /bin/umount, /usr/bin/eject
EOF
chmod 440 "$SUDOERS"
echo "[install] sudoers: $RUN_USER may shutdown / mkfs.vfat / umount / eject without password"

# Block USB auto-mount popups at the udev level. Our export.py mounts the
# drive explicitly via udisksctl, so we don't need any session helper to do
# it for us. This kills the source of the popup, not just the popup itself.
UDEV_RULE=/etc/udev/rules.d/99-openscanner-no-automount.rules
cat > "$UDEV_RULE" <<'EOF'
# openscanner: stop udisks/desktop from auto-mounting removable USB drives.
# We mount on demand from the app via udisksctl.
SUBSYSTEM=="block", ENV{ID_BUS}=="usb", ENV{UDISKS_AUTO}="0", ENV{UDISKS_IGNORE}="0"
EOF
udevadm control --reload-rules || true
echo "[install] udev: USB auto-mount disabled (manual mount still works)"

echo
echo "[install] done!"
echo "  - run now:    sudo systemctl start openscanner"
echo "  - logs:       journalctl -u openscanner -f"
echo "  - disable:    sudo systemctl disable openscanner"
echo "  - update.sh runs at every start; toggle auto_update in .settings to opt out"
echo
echo "  Pi OS Lite users: reboot once after install so the new groups"
echo "  (video/render/input) take effect for the service user."
