"""Touchscreen 3D point cloud viewer.

Pi-friendly software renderer:
  - subsample to ~60k points by default (configurable in settings)
  - orthographic projection (no perspective math, faster)
  - drag to rotate, +/- buttons to zoom, RESET, BACK
  - back-to-front painter via z-sort + numpy fancy index
"""

import os
import time

import cv2
import numpy as np

from . import display
from .config import (
    BTN_H, BTN_Y, C, FONT, SCREEN_H, SCREEN_W, PREVIEW_H,
)
from .export import _read_ply


# View canvas takes the entire preview area (the buttons row stays as before)
VIEW_W = SCREEN_W
VIEW_H = PREVIEW_H


def _load_subsampled(path, max_points):
    xyz, rgb = _read_ply(path)
    if len(xyz) > max_points:
        idx = np.random.choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    return xyz, rgb


def _rot_matrix(yaw, pitch):
    cy, sy = np.cos(yaw),   np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    Ry = np.array([[ cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    return Rx @ Ry


def _render(xyz, rgb, yaw, pitch, scale, w, h):
    """Software rasterise with painter's algorithm (back-to-front)."""
    R = _rot_matrix(yaw, pitch)
    pts = xyz @ R.T

    sx = (pts[:, 0] * scale + w / 2).astype(np.int32)
    sy = (-pts[:, 1] * scale + h / 2).astype(np.int32)

    in_bounds = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
    sx, sy, z, col = sx[in_bounds], sy[in_bounds], pts[:, 2][in_bounds], rgb[in_bounds]

    if len(sx) == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)

    order = np.argsort(-z)
    sx, sy, col = sx[order], sy[order], col[order]

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    # rgb stored as RGB - flip to BGR for OpenCV
    canvas[sy, sx] = col[:, ::-1]
    return canvas


def show(ply_paths, max_points=60000):
    """Display interactive point-cloud viewer. Returns when user taps BACK.

    ply_paths: list of paths to merge and view (e.g. all per-capture PLYs).
    """
    if not ply_paths:
        _splash("No point clouds to view", "Capture some shots first.")
        return

    # Load + merge
    all_xyz, all_rgb = [], []
    for p in ply_paths:
        try:
            xyz, rgb = _read_ply(p)
            all_xyz.append(xyz); all_rgb.append(rgb)
        except Exception as e:
            print(f"[VIEW] skip {p}: {e}")
    if not all_xyz:
        _splash("All point clouds failed to load")
        return

    xyz = np.concatenate(all_xyz).astype(np.float32)
    rgb = np.concatenate(all_rgb)

    # Centre cloud at origin so rotation feels natural
    centroid = np.median(xyz, axis=0)
    xyz -= centroid

    # Subsample
    if len(xyz) > max_points:
        idx = np.random.choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]

    # Initial scale: fit cloud into 80% of view height
    span = float(np.percentile(np.linalg.norm(xyz, axis=1), 95)) or 1.0
    base_scale = (VIEW_H * 0.4) / span

    state = {
        "yaw": 0.0, "pitch": 0.0, "scale": base_scale,
        "action": None, "quit": False,
    }

    def _drain_input():
        for ev in display.events():
            if isinstance(ev, display.Tap):
                if ev.y > BTN_Y:
                    bw = SCREEN_W // 4
                    state["action"] = ["zoom_in", "zoom_out", "reset", "back"][min(ev.x // bw, 3)]
            elif isinstance(ev, display.Drag) and ev.y < BTN_Y:
                state["yaw"]   += ev.dx * 0.01
                state["pitch"] += ev.dy * 0.01
                state["pitch"]  = max(-1.5, min(1.5, state["pitch"]))
            elif isinstance(ev, display.Quit):
                state["quit"] = True
            elif isinstance(ev, display.Key) and ev.char in ("q", "\x1b"):
                state["quit"] = True

    canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

    while True:
        view = _render(xyz, rgb, state["yaw"], state["pitch"],
                       state["scale"], VIEW_W, VIEW_H)
        canvas[:VIEW_H, :] = view
        canvas[VIEW_H:, :] = 0

        # Stats overlay
        cv2.rectangle(canvas, (0, 0), (260, 24), (0, 0, 0), -1)
        cv2.putText(canvas, f"{len(xyz)} pts  yaw={np.degrees(state['yaw']):+.0f}  "
                            f"pitch={np.degrees(state['pitch']):+.0f}",
                    (6, 17), FONT, 0.42, C["white"], 1)
        cv2.putText(canvas, "drag to rotate", (SCREEN_W - 130, 17),
                    FONT, 0.42, C["grey"], 1)

        # Buttons
        bw = SCREEN_W // 4
        for i, (lbl, col) in enumerate([
            ("ZOOM +", C["blue"]),
            ("ZOOM -", C["blue"]),
            ("RESET",  C["dim"]),
            ("BACK",   C["red"]),
        ]):
            bx = i * bw
            cv2.rectangle(canvas, (bx + 2, BTN_Y + 2),
                          (bx + bw - 2, SCREEN_H - 2), col, -1)
            cv2.rectangle(canvas, (bx + 2, BTN_Y + 2),
                          (bx + bw - 2, SCREEN_H - 2), C["white"], 1)
            (tw, th), _ = cv2.getTextSize(lbl, FONT, 0.7, 2)
            cv2.putText(canvas, lbl,
                        (bx + (bw - tw) // 2, BTN_Y + (BTN_H + th) // 2),
                        FONT, 0.7, C["white"], 2)

        display.show(canvas)
        _drain_input()
        time.sleep(0.03)

        action = state["action"]; state["action"] = None
        if action == "zoom_in":
            state["scale"] *= 1.25
        elif action == "zoom_out":
            state["scale"] /= 1.25
        elif action == "reset":
            state["yaw"] = state["pitch"] = 0.0
            state["scale"] = base_scale
        elif action == "back" or state["quit"]:
            return


def _splash(line1, line2=""):
    """Brief modal message when there's nothing to view."""
    end = time.time() + 1.6
    while time.time() < end:
        ov = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        cv2.rectangle(ov, (60, 160), (740, 320), (25, 25, 25), -1)
        cv2.rectangle(ov, (60, 160), (740, 320), C["white"], 1)
        (tw, _), _ = cv2.getTextSize(line1, FONT, 0.7, 2)
        cv2.putText(ov, line1, ((SCREEN_W - tw) // 2, 220), FONT, 0.7, C["white"], 2)
        if line2:
            (tw2, _), _ = cv2.getTextSize(line2, FONT, 0.5, 1)
            cv2.putText(ov, line2, ((SCREEN_W - tw2) // 2, 270), FONT, 0.5, C["grey"], 1)
        display.show(ov)
        display.events()
        time.sleep(0.05)
