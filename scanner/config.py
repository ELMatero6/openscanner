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

# Overall screen
SCREEN_W  = 800
SCREEN_H  = 480
PANEL_W   = SCREEN_W // 2

# Main-UI chrome (Pocket-PC / Win9x toolgun look)
TITLE_H   = 24            # navy caption bar at top
NAV_H     = 24            # white "hyperlink" bar below caption
VF_Y      = TITLE_H + NAV_H        # viewfinder top (48)
STATS_H   = 20            # olive-green debug stats strip
PREVIEW_H = 340           # top-area height (used by calibration + viewer too)
VF_H      = PREVIEW_H - VF_Y - STATS_H   # viewfinder height in main UI (248)
STATS_Y   = VF_Y + VF_H                  # olive strip Y (296)

# Bottom button bar
BTN_Y     = PREVIEW_H                # 340
BTN_H     = SCREEN_H - PREVIEW_H     # 140

# Nav-bar hit regions (touch targets for 3D / Power / Cal links)
NAV_3D_X,  NAV_3D_W  = 0,   170
NAV_PWR_X, NAV_PWR_W = 500, 140
NAV_CAL_X, NAV_CAL_W = 640, 160

DEFAULT_BASELINE_MM = 60.0
DISP_SCALE = 0.5

# Single SGBM preset. Intentionally un-fancy: no scene modes, no depth
# clipping, no BG removal. Tune here if the scanner ever needs to target
# a different subject size.
NUM_DISP   = 128
BLOCK_SIZE = 5

# Per-capture PLY quality gates. Applied in disparity_to_points.
# These keep garbage depth (window glare, textureless walls) out of the
# point cloud at capture time. Nothing downstream can recover phantom
# geometry, so we'd rather drop it here.
MAX_DEPTH_M     = 4.0   # reject Z > this (kills sun/window streaks)
MIN_DEPTH_M     = 0.10  # reject near-lens speckle
MAX_BRIGHTNESS  = 245   # drop blown-out pixels (no valid stereo match there)

FONT      = cv2.FONT_HERSHEY_SIMPLEX
MONO_FONT = cv2.FONT_HERSHEY_PLAIN       # narrower, terminal-ish

# BGR colours
C = {
    # functional (legacy keys - kept so nothing else breaks)
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
    # toolgun / Win9x palette
    "navy":        (128, 0,   0),        # title bar
    "navy_soft":   (152, 32,  32),       # nav link hover
    "win_body":    (200, 200, 200),      # chrome grey
    "win_face":    (212, 212, 212),      # button face
    "win_hi":      (248, 248, 248),      # bevel highlight
    "win_sh":      (128, 128, 128),      # bevel shadow
    "win_dk":      (64,  64,  64),       # outer dark
    "link":        (200, 80,  0),        # hyperlink blue
    "link_visit":  (140, 40,  140),      # visited purple
    "cream":       (220, 232, 245),      # pill background
    "olive_bg":    (80,  110, 80),       # debug strip bg
    "olive_txt":   (170, 220, 180),      # debug strip text
    "black":       (0,   0,   0),
}

BTNS = [
    ("save",  "SAVE",   "yellow", "yellow"),
    ("clear", "CLEAR",  "red",    "dim"),
]
BTN_W = SCREEN_W // len(BTNS)

# Legacy aliases - unused in the new UI, kept because calibration.py imports them.
CAL_W, CAL_H = 80, 34
CAL_X = PANEL_W - CAL_W - 4
CAL_Y = 4
