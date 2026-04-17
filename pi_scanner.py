"""
Pi Stereo 3D Scanner
====================
Raspberry Pi 4 + Waveshare 5" 800x480 touchscreen.

Layout:
  +------------------+------------------+
  |   live camera    |  live depth /    |
  |   + crosshair    |  coverage map    |
  +------------------+------------------+
  |  DIST  |  BG  |  SAVE  |  CLEAR     |
  +--------+------+--------+------------+

Physical button GPIO 17 = capture (semi) or hold (auto).

Dependencies:
    pip3 install opencv-python numpy --break-system-packages
"""

import cv2
import numpy as np
import os
import csv
import time
import zipfile
import threading
from datetime import datetime

try:
    _ = cv2.ximgproc.createDisparityWLSFilter
    XIMGPROC = True
except AttributeError:
    XIMGPROC = False
    print("[WARN] opencv-contrib not installed - WLS filter disabled")

try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO_OK = True
except Exception:
    GPIO_OK = False

CAMERA_INDEX     = 0
CAPTURE_WIDTH    = 2560
CAPTURE_HEIGHT   = 960
CALIBRATION_FILE = "stereo_calibration.npz"
SAVE_DIR         = "scans"
GPIO_PIN         = 17
AUTO_INTERVAL    = 0.8

SCREEN_W  = 800
SCREEN_H  = 480
PANEL_W   = SCREEN_W // 2
PREVIEW_H = 340
BTN_Y     = PREVIEW_H
BTN_H     = SCREEN_H - PREVIEW_H

STEREO_BASELINE_MM = 60.0

DISP_SCALE = 0.5
DIST_PRESETS = {
    "CLOSE": {"num_disp": 160, "block": 3,  "label": "CLOSE 20-60cm"},
    "MED":   {"num_disp": 128, "block": 5,  "label": "MED  50-150cm"},
    "FAR":   {"num_disp": 64,  "block": 7,  "label": "FAR   1-5m"},
}
DIST_ORDER = ["CLOSE", "MED", "FAR"]

os.makedirs(SAVE_DIR, exist_ok=True)
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
BTNS  = [
    ("dist",   "DIST",   "purple", "purple"),
    ("bgrem",  "BG OFF", "orange", "dim"),
    ("save",   "SAVE",   "yellow", "yellow"),
    ("clear",  "CLEAR",  "red",    "dim"),
]

CAL_W, CAL_H = 80, 34
CAL_X = PANEL_W - CAL_W - 4
CAL_Y = 4


def load_cal(path, fallback):
    if not path or not os.path.exists(path):
        print("[WARN] No calibration - running uncalibrated")
        return None
    try:
        d  = np.load(path)
        sz = tuple(d["img_size"].astype(int)) if "img_size" in d else fallback
        m1L, m2L = cv2.initUndistortRectifyMap(
            d["mtxL"], d["distL"], d["R1"], d["P1"], sz, cv2.CV_16SC2)
        m1R, m2R = cv2.initUndistortRectifyMap(
            d["mtxR"], d["distR"], d["R2"], d["P2"], sz, cv2.CV_16SC2)
        print(f"[CAL] loaded {sz[0]}x{sz[1]}")
        return {"m1L": m1L, "m2L": m2L, "m1R": m1R, "m2R": m2R, "Q": d["Q"]}
    except Exception as e:
        print(f"[CAL ERROR] {e}")
        return None


def rectify(left, right, cal):
    return (cv2.remap(left,  cal["m1L"], cal["m2L"], cv2.INTER_LINEAR),
            cv2.remap(right, cal["m1R"], cal["m2R"], cv2.INTER_LINEAR))


def build_matcher(num_disp=128, block=3):
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block,
        P1=8*3*block*block,
        P2=32*3*block*block,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=150,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    if not XIMGPROC:
        return left_matcher, None, None

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    wls = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls.setLambda(16000)
    wls.setSigmaColor(2.0)
    return left_matcher, right_matcher, wls


def get_disparity(left, right, matchers):
    left_matcher, right_matcher, wls = matchers
    s  = DISP_SCALE
    gl = cv2.resize(cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY), None, fx=s, fy=s)
    gr = cv2.resize(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), None, fx=s, fy=s)

    disp_L = left_matcher.compute(gl, gr)

    if wls is not None:
        disp_R     = right_matcher.compute(gr, gl)
        left_small = cv2.resize(left, (gl.shape[1], gl.shape[0]))
        disp_L     = wls.filter(disp_L, left_small, disparity_map_right=disp_R)

    return disp_L.astype(np.float32) / 16.0


def build_live_matcher():
    b = 5
    return cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=64, blockSize=b,
        P1=8*3*b*b, P2=32*3*b*b, disp12MaxDiff=1,
        uniquenessRatio=10, speckleWindowSize=80, speckleRange=2,
        preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_HH4)


def sample_crosshair_disp(disp):
    h, w   = disp.shape[:2]
    cx, cy = w // 2, h // 2
    r      = max(6, min(w, h) // 5)
    patch  = disp[max(0,cy-r):cy+r, max(0,cx-r):cx+r]
    valid  = patch[patch > 0]
    if len(valid) < 4:
        return None
    return float(np.median(valid))


def cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read()) / 1000.0
    except Exception:
        return None


def main():
    print("Pi Stereo Scanner - baseline")


if __name__ == "__main__":
    main()
