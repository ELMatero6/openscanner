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
  BRANCH="${OPENSCANNER_BRANCH:-ma in}"
  INSTALL_DIR="/opt/openscanner"
  SERVICE="/etc/systemd/system/openscanner.service"
  RUN_USER="${SUDO_USER:-pi}"

  if [[ $EUID -ne 0 ]]; then
      echo "[install] ERROR: must be run as root (sudo)"
      exit 1
  fi

  echo "[install] Starting setup..."
  echo "[install] Target User: $RUN_USER"
  echo "[install] Target Dir:  $INSTALL_DIR"
  echo "[install] Branch:      $BRANCH"

  # 1. System Packages Installation
  echo "[install] apt: updating system packages and installing dependencies..."
  apt-get update -qq
  apt-get install -y --no-install-recommends \
      git python3 python3-pip python3-numpy \
      python3-opencv python3-rpi.gpio \
      libgl1 \
      dosfstools udisks2

  # Handle potential missing/renamed GUI libraries gracefully
  for pkg in libglib2.0-0 libglib2.0-0t64 libatlas3-base libopenblas0; do
      apt-get install -y --no-install-recommends "$pkg" 2>/dev/null || true
  done

  # 2. Python Dependencies
  echo "[install] pip: installing opencv-contrib-python (for WLS)..."
  # Use --break-system-packages for modern Debian/Pi OS versions
  pip3 install --break-system-packages opencv-contrib-python || \
      echo "[install] WARN: opencv-contrib install failed - WLS filter will be disabled"

  # 3. Repository Setup
  if [[ -d "$INSTALL_DIR/.git" ]]; then
      echo "[install] repo exists, fetching latest..."
      # Ensure the owner of the directory is the intended running user
      chown -R "$RUN_USER:$RUN_USER" "$INSTALL_DIR"
      sudo -u "$RUN_USER" git -C "$INSTALL_DIR" fetch --quiet origin
      sudo -u "$RUN_USER" git -C "$INSTALL_DIR" checkout --quiet "$BRANCH"
      sudo -u "$RUN_USER" git -C "$INSTALL_DIR" reset --quiet --hard "origin/$BRANCH"
  else
      echo "[install] cloning $REPO_URL into $INSTALL_DIR..."
      mkdir -p "$INSTALL_DIR"
      chown -R "$RUN_USER:$RUN_USER" "$INSTALL_DIR"
      sudo -u "$RUN_USER" git clone --quiet --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
  fi

  # Make scripts executable
  chmod +x "$INSTALL_DIR/update.sh" "$INSTALL_DIR/install.sh"

  # 4. Systemd Unit File Setup
  echo "[install] writing systemd unit at $SERVICE..."
  cat > "$SERVICE" <<EOF
  [Unit]
  Description=openscanner stereo 3D scanner
  After=network-online.target
  Wants=network-online.target

  [Service]
  Type=simple
  User=$RUN_USER
  Group=$RUN_USER
  WorkingDirectory=$INSTALL_DIR
  # Execute the wrapper script which handles updates and then launches the GUI
  ExecStart=/usr/bin/env bash $INSTALL_DIR/update.sh
  Restart=on-failure
  RestartSec=5

  [Install]
  WantedBy=multi-user.target
  EOF

  # Reload systemd and enable service
  systemctl daemon-reload
  systemctl enable openscanner.service

  # 5. USB Privileges
  SUDOERS=/etc/sudoers.d/openscanner
  cat > "$SUDOERS" <<EOF
  # Allows $RUN_USER to manage USB devices without a password for scanner operation.
  $RUN_USER ALL=(root) NOPASSWD: /sbin/shutdown, /usr/sbin/mkfs.vfat, /sbin/mkfs.vfat, /bin/umount, /usr/bin/eject
  EOF
  chmod 440 "$SUDOERS"
  echo "[install] User $RUN_USER granted necessary USB management sudo rights."

  echo
  echo "=============================== ================================ ========="
  echo "[install] SETUP COMPLETE!"
  echo "=============================== ================================ ========="
  echo "The systemd unit file has been updated to run the GUI scanner service."
  echo "To test immediately, run: sudo systemctl start openscanner"
  echo "To check logs: journalctl -u openscanner -f"
  echo "=============================== ================================ ========="
