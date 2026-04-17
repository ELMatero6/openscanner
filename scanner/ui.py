"""UI rendering: viewfinder, status bar, buttons, overlays. No state owned here."""

import time

import cv2
import numpy as np

from .config import (
    BTN_H, BTN_W, BTN_Y, BTNS, C, CAL_X, CAL_Y, CAL_W, CAL_H,
    DIST_PRESETS, FONT, PANEL_W, PREVIEW_H, PWR_X, PWR_Y, PWR_W, PWR_H,
    SCREEN_H, SCREEN_W, VIEW_X, VIEW_Y, VIEW_W, VIEW_H,
)


def fit_to_panel(img):
    """Letterbox image into PANEL_W x PREVIEW_H without stretching."""
    ih, iw = img.shape[:2]
    s = min(PANEL_W / iw, PREVIEW_H / ih)
    nw, nh = int(iw * s), int(ih * s)
    out = np.zeros((PREVIEW_H, PANEL_W, 3), np.uint8)
    out[(PREVIEW_H - nh) // 2:(PREVIEW_H - nh) // 2 + nh,
        (PANEL_W - nw) // 2:(PANEL_W - nw) // 2 + nw] = cv2.resize(img, (nw, nh))
    return out


def make_empty_right_panel(msg1, msg2=""):
    p = np.full((PREVIEW_H, PANEL_W, 3), C["panel"], np.uint8)
    (w1, _), _ = cv2.getTextSize(msg1, FONT, 0.55, 1)
    cv2.putText(p, msg1, ((PANEL_W - w1) // 2, PREVIEW_H // 2 - 10),
                FONT, 0.55, C["grey"], 1)
    if msg2:
        (w2, _), _ = cv2.getTextSize(msg2, FONT, 0.42, 1)
        cv2.putText(p, msg2, ((PANEL_W - w2) // 2, PREVIEW_H // 2 + 18),
                    FONT, 0.42, C["dim"], 1)
    return p


def _rangefinder(canvas, dist_m):
    if dist_m is not None:
        s = f"{dist_m * 100:.0f} cm" if dist_m < 1.0 else f"{dist_m:.2f} m"
        col = C["green"] if 0.2 <= dist_m <= 4.0 else C["yellow"]
    else:
        s = "-- m"; col = C["grey"]
    (tw, th), _ = cv2.getTextSize(s, FONT, 0.9, 2)
    px, py = PANEL_W // 2, 36
    cv2.rectangle(canvas, (px - tw // 2 - 10, py - th - 6),
                  (px + tw // 2 + 10, py + 8), (0, 0, 0), -1)
    cv2.rectangle(canvas, (px - tw // 2 - 10, py - th - 6),
                  (px + tw // 2 + 10, py + 8), col, 1)
    cv2.putText(canvas, s, (px - tw // 2, py), FONT, 0.9, col, 2)


def _status_bar(canvas, state, sys):
    bar_y = PREVIEW_H - 26
    cv2.rectangle(canvas, (0, bar_y), (SCREEN_W, PREVIEW_H), (0, 0, 0), -1)

    rect = "RECT" if state["cal"] else "NO-RECT"
    base = f" Shots:{state['captures']}  {rect}  {state['dist_mode']}  v{state.get('version', '')}"
    cv2.putText(canvas, base, (4, PREVIEW_H - 7), FONT, 0.42, C["white"], 1)

    parts = []
    if sys.get("temp") is not None:
        t = sys["temp"]
        col = C["red"] if t >= 75 else C["yellow"] if t >= 60 else C["white"]
        parts.append((f"{t:.0f}C", col))
    for k, lbl, hi, mid in (("cpu", "CPU", 90, 70),
                            ("ram", "RAM", 85, 70),
                            ("disk", "DSK", 90, 75)):
        v = sys.get(k)
        if v is None:
            continue
        col = C["red"] if v >= hi else C["yellow"] if v >= mid else C["white"]
        parts.append((f"{lbl}:{v}%", col))

    x = SCREEN_W - 4
    for txt, col in reversed(parts):
        (tw, _), _ = cv2.getTextSize(txt, FONT, 0.42, 1)
        x -= tw + 8
        cv2.putText(canvas, txt, (x, PREVIEW_H - 7), FONT, 0.42, col, 1)


def _crosshair(canvas):
    cx, cy = PANEL_W // 2, PREVIEW_H // 2
    r = 14
    cv2.line(canvas, (cx - r, cy), (cx + r, cy), C["white"], 1)
    cv2.line(canvas, (cx, cy - r), (cx, cy + r), C["white"], 1)
    cv2.circle(canvas, (cx, cy), r // 2, C["white"], 1)


def _top_buttons(canvas, state):
    # CAL button (top-right of left panel)
    col = C["green"] if state["cal"] else C["red"]
    lbl = "CAL OK" if state["cal"] else "CAL"
    cv2.rectangle(canvas, (CAL_X, CAL_Y), (CAL_X + CAL_W, CAL_Y + CAL_H), col, -1)
    cv2.rectangle(canvas, (CAL_X, CAL_Y), (CAL_X + CAL_W, CAL_Y + CAL_H), C["white"], 1)
    (tw, th), _ = cv2.getTextSize(lbl, FONT, 0.44, 1)
    cv2.putText(canvas, lbl, (CAL_X + (CAL_W - tw) // 2, CAL_Y + (CAL_H + th) // 2),
                FONT, 0.44, C["white"], 1)

    # Power button
    cv2.rectangle(canvas, (PWR_X, PWR_Y), (PWR_X + PWR_W, PWR_Y + PWR_H), C["dim"], -1)
    cv2.rectangle(canvas, (PWR_X, PWR_Y), (PWR_X + PWR_W, PWR_Y + PWR_H), C["white"], 1)
    cv2.putText(canvas, "PWR", (PWR_X + 8, PWR_Y + 22), FONT, 0.44, C["white"], 1)

    # 3D View button (top-left of left panel)
    cv2.rectangle(canvas, (VIEW_X, VIEW_Y), (VIEW_X + VIEW_W, VIEW_Y + VIEW_H), C["teal"], -1)
    cv2.rectangle(canvas, (VIEW_X, VIEW_Y), (VIEW_X + VIEW_W, VIEW_Y + VIEW_H), C["white"], 1)
    cv2.putText(canvas, "3D", (VIEW_X + 18, VIEW_Y + 24), FONT, 0.6, C["white"], 2)


def _bottom_buttons(canvas, state):
    for i, (key, default_lbl, col_on, col_off) in enumerate(BTNS):
        bx = i * BTN_W
        active = False
        label = default_lbl

        if key == "dist":
            label = state["dist_mode"]; active = True
        elif key == "bgrem":
            active = state.get("bg_thresh") is not None
            label = "BG ON" if active else "BG OFF"
        elif key == "save":
            active = state.get("saving", False)
            label = "ZIPPING..." if active else "SAVE"

        col = C[col_on] if active else C[col_off]
        cv2.rectangle(canvas, (bx + 2, BTN_Y + 2),
                      (bx + BTN_W - 2, SCREEN_H - 2), col, -1)
        cv2.rectangle(canvas, (bx + 2, BTN_Y + 2),
                      (bx + BTN_W - 2, SCREEN_H - 2), C["white"], 1)
        if bx > 0:
            cv2.line(canvas, (bx, BTN_Y), (bx, SCREEN_H), (60, 60, 60), 1)

        (tw, th), _ = cv2.getTextSize(label, FONT, 0.72, 2)
        cv2.putText(canvas, label,
                    (bx + (BTN_W - tw) // 2,
                     BTN_Y + (BTN_H + th) // 2 - (8 if key == "dist" else 0)),
                    FONT, 0.72, C["white"], 2)

        if key == "dist":
            rng = DIST_PRESETS[state["dist_mode"]]["label"]
            (sw, _), _ = cv2.getTextSize(rng, FONT, 0.36, 1)
            cv2.putText(canvas, rng,
                        (bx + (BTN_W - sw) // 2, BTN_Y + BTN_H - 14),
                        FONT, 0.36, C["white"], 1)


def draw_ui(canvas, state, left_panel, right_panel, sys):
    canvas[:] = C["bg"]
    canvas[:PREVIEW_H, :PANEL_W] = left_panel
    cv2.line(canvas, (PANEL_W, 0), (PANEL_W, PREVIEW_H), C["grey"], 1)
    canvas[:PREVIEW_H, PANEL_W:] = right_panel

    _rangefinder(canvas, state.get("crosshair_dist_m"))
    _crosshair(canvas)
    _status_bar(canvas, state, sys)

    rp_lbl = "COVERAGE" if state.get("has_disp") else "LIVE DEPTH"
    (rw, _), _ = cv2.getTextSize(rp_lbl, FONT, 0.38, 1)
    cv2.putText(canvas, rp_lbl,
                (PANEL_W + (PANEL_W - rw) // 2, 14), FONT, 0.38, C["grey"], 1)

    if time.time() < state.get("flash_until", 0.0):
        cv2.rectangle(canvas, (0, 0), (PANEL_W, PREVIEW_H), C["white"], 8)

    _top_buttons(canvas, state)
    _bottom_buttons(canvas, state)

    if state.get("saving"):
        ov = canvas.copy()
        cv2.rectangle(ov, (150, 150), (650, 260), (20, 20, 20), -1)
        cv2.rectangle(ov, (150, 150), (650, 260), C["white"], 1)
        cv2.putText(ov, "Zipping scan folder...",
                    (170, 205), FONT, 0.7, C["white"], 2)
        pct = state.get("zip_pct", 0)
        cv2.putText(ov, f"{pct}%  ({state['captures']} pairs)",
                    (170, 245), FONT, 0.5, C["yellow"], 1)
        cv2.addWeighted(ov, 0.9, canvas, 0.1, 0, canvas)


def hit_button(x, y):
    """Map a touch (x,y) to a logical button name, or None."""
    if VIEW_X <= x <= VIEW_X + VIEW_W and VIEW_Y <= y <= VIEW_Y + VIEW_H:
        return "view"
    if PWR_X <= x <= PWR_X + PWR_W and PWR_Y <= y <= PWR_Y + PWR_H:
        return "power"
    if CAL_X <= x <= CAL_X + CAL_W and CAL_Y <= y <= CAL_Y + CAL_H:
        return "cal"
    if BTN_Y <= y <= SCREEN_H:
        i = x // BTN_W
        if 0 <= i < len(BTNS):
            return BTNS[i][0]
    return None


def confirm_dialog(win, lines, buttons, callback_setter):
    """Render an overlay with text + buttons. Returns the action key chosen.

    buttons: list of (label, colour, action_key).
    callback_setter: callback used to wire up the touch handler in caller's loop.
    """
    state = {"action": None}

    def on_touch(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or y < BTN_Y:
            return
        bw = SCREEN_W // len(buttons)
        i = min(x // bw, len(buttons) - 1)
        param["action"] = buttons[i][2]

    callback_setter(on_touch, state)

    while state["action"] is None:
        ov = np.zeros((SCREEN_H, SCREEN_W, 3), np.uint8)
        cv2.rectangle(ov, (40, 60), (SCREEN_W - 40, PREVIEW_H - 20), (30, 30, 30), -1)
        cv2.rectangle(ov, (40, 60), (SCREEN_W - 40, PREVIEW_H - 20), C["white"], 1)
        for i, line in enumerate(lines):
            col = line[1] if isinstance(line, tuple) else C["white"]
            txt = line[0] if isinstance(line, tuple) else line
            scale = 0.45 if len(txt) > 40 else 0.55
            (tw, _), _ = cv2.getTextSize(txt, FONT, scale, 1)
            cv2.putText(ov, txt, ((SCREEN_W - tw) // 2, 120 + i * 32),
                        FONT, scale, col, 1)
        bw = SCREEN_W // len(buttons)
        for i, (lbl, col, _) in enumerate(buttons):
            bx = i * bw
            cv2.rectangle(ov, (bx + 2, BTN_Y + 2),
                          (bx + bw - 2, SCREEN_H - 2), col, -1)
            cv2.rectangle(ov, (bx + 2, BTN_Y + 2),
                          (bx + bw - 2, SCREEN_H - 2), C["white"], 1)
            (tw, th), _ = cv2.getTextSize(lbl, FONT, 0.7, 2)
            cv2.putText(ov, lbl, (bx + (bw - tw) // 2, BTN_Y + (BTN_H + th) // 2),
                        FONT, 0.7, C["white"], 2)
        cv2.imshow(win, ov)
        cv2.waitKey(50)

    return state["action"]
