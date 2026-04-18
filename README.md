# openscanner

A stereo 3D scanner for Raspberry Pi 4 + Waveshare 5" 800x480 touchscreen.

## Features

- Stereo SGBM with optional WLS smoothing
- Touch UI: INTERIOR / SMALL / LARGE scene presets, BG removal, rangefinder, coverage map
- Per-capture export: rectified L/R PNGs, colour + 16-bit raw disparity, PLY point cloud
- On-device 3D point cloud viewer (drag to rotate, zoom)
- USB export without formatting the drive
- Self-describing zip output (calibration, CSV, fused PLY, README)
- Persistent settings, graceful shutdown, auto-update on boot
- Git SHA in status bar for field diagnostics

## Install on a fresh Pi

```bash
curl -sSL https://raw.githubusercontent.com/elmatero6/openscanner/main/install.sh | sudo bash
sudo systemctl start openscanner
```

Or clone manually then:

```bash
sudo ./install.sh
```

Logs: `journalctl -u openscanner -f`

## Layout

```
pi_scanner.py            thin entrypoint
scanner/
  config.py              constants, screen geometry, presets
  settings.py            persistent JSON settings (.settings)
  sysinfo.py             cpu temp, ram, disk, git SHA, shutdown
  stereo.py              SGBM matchers, disparity, crosshair sampling
  calibration.py         load/run/save calibration (reports RMS error)
  export.py              CSV, PNG, PLY, zip, USB copy
  viewer.py              touchscreen point cloud viewer
  ui.py                  UI drawing, hit-testing, modal dialogs
  main.py                capture loop
install.sh               one-shot installer (sudo)
update.sh                auto-pull + launch (run by systemd at boot)
openscanner.service      systemd unit (reference copy)
```

## Hardware

- Raspberry Pi 4 (4GB+ recommended, 8GB comfortable)
- Stereo USB camera producing a single 2x wide frame (default 2560x960)
- Waveshare 5" DSI 800x480 touchscreen
- GPIO 17 momentary button: tap = capture, 3s hold = shutdown

## Capture outputs (per shot N)

| file          | purpose                                                 |
| ------------- | ------------------------------------------------------- |
| `N_L.png`     | rectified left image                                    |
| `N_R.png`     | rectified right image                                   |
| `N_D.png`     | TURBO-coloured disparity preview                        |
| `N_D_raw.png` | 16-bit raw disparity * 16 (divide by 16 for float)      |
| `N.ply`       | binary PLY point cloud in camera frame (metres)         |

All metadata goes into `captures.csv` (single consistent schema).

## Zip / USB export contents

- all `N_*` files and `captures.csv`
- `stereo_calibration.npz` (intrinsics, extrinsics, Q matrix)
- `concat.ply` (concatenation of per-capture PLYs, no pose registration)
- `README.txt` (build version, baseline, focal, file format notes)

## Settings (`.settings` JSON)

```json
{
  "dist_mode": "SMALL",
  "bg_on": false,
  "auto_update": true,
  "screen_rotate": 0,
  "last_save_dest": "local",
  "viewer_subsample": 60000
}
```

Set `auto_update: false` on deployed units to pin the current version.

## Calibration

1. Tap **CAL** on the main screen.
2. Hold a 9x6 chessboard (25mm squares) at various angles/distances.
3. Pull the trigger or tap SAVE PAIR for each. Need 15+ pairs.
4. Tap **CALIBRATE**. You'll see a reprojection-error readout:
   - **< 0.5 px EXCELLENT**, **< 1.5 px OK**, **>= 1.5 px POOR — redo**.

## Troubleshooting

- **"CAL" is red**: no calibration file. Calibrate before trusting depth/PLY.
- **PLY files missing**: calibration required (Q matrix).
- **USB export "read-only"**: most SD adaptors / locked drives. Try another.
- **Update didn't apply**: check `journalctl -u openscanner` for the update log.

## License

See `LICENSE`.
