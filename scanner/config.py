"""Static configuration: paths, screen geometry, presets, colours."""

import cv2

CAMERA_INDEX     = 0
CAPTURE_WIDTH    = 2560
CAPTURE_HEIGHT   = 960
CALIBRATION_FILE = "stereo_calibration.npz"
SAVE_DIR         = "scans"
SETTINGS_FILE    = ".settings"
GPIO_PIN         = 17
AUTO_INTERVAL    = 0.8
SHUTDOWN_HOLD_S  = 3.0

SCREEN_W  = 800
SCREEN_H  = 480
PANEL_W   = SCREEN_W // 2
PREVIEW_H = 340
BTN_Y     = PREVIEW_H
BTN_H     = SCREEN_H - PREVIEW_H

DEFAULT_BASELINE_MM = 60.0
DISP_SCALE = 0.5

DIST_PRESETS = {
    "CLOSE": {"num_disp": 160, "block": 3, "label": "CLOSE 20-60cm"},
    "MED":   {"num_disp": 128, "block": 5, "label": "MED 50-150cm"},
    "FAR":   {"num_disp": 64,  "block": 7, "label": "FAR  1-5m"},
}
DIST_ORDER = ["CLOSE", "MED", "FAR"]

FONT = cv2.FONT_HERSHEY_SIMPLEX

C = {
    "bg":     (15,  15,  15),
    "panel":  (25,  25,  25),
    "white":  (240, 240, 240),
    "dim":    (55,  55,  55),
    "green":  (40,  180, 60),
    "blue":   (200, 100, 30),
    "orange": (30,  140, 240),
    "red":    (50,  50,  210),
    "yellow": (20,  200, 220),
    "teal":   (160, 190, 30),
    "grey":   (100, 100, 100),
    "purple": (180, 60,  160),
}

BTN_W = SCREEN_W // 4
BTNS = [
    ("dist",  "DIST",   "purple", "purple"),
    ("bgrem", "BG OFF", "orange", "dim"),
    ("save",  "SAVE",   "yellow", "yellow"),
    ("clear", "CLEAR",  "red",    "dim"),
]

CAL_W, CAL_H = 80, 34
CAL_X = PANEL_W - CAL_W - 4
CAL_Y = 4

VIEW_W, VIEW_H = 60, 34
VIEW_X = 4
VIEW_Y = 4

PWR_W, PWR_H = 50, 34
PWR_X = PANEL_W - CAL_W - PWR_W - 10
PWR_Y = 4
