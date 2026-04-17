"""UI rendering: Pocket-PC / Garry's Mod toolgun aesthetic.

Layout (top to bottom):
  [0    .. 24 ]  Navy title bar (logo + "openscanner" + clock)
  [24   .. 48 ]  White "hyperlink" nav bar (< 3D View | Power | Cal >)
  [48   ..VF_H]  Viewfinder (left panel) + Live depth / Coverage (right panel)
  [STATS_Y..BTN_Y] Olive debug-stats strip (monospace)
  [BTN_Y.. 480]  Win9x raised buttons (DIST / BG / SAVE / CLEAR)
"""

import time

import cv2
import numpy as np

from .config import (
    BTN_H, BTN_W, BTN_Y, BTNS, C, DIST_PRESETS, FONT, MONO_FONT,
    NAV_3D_X, NAV_3D_W, NAV_CAL_X, NAV_CAL_W, NAV_PWR_X, NAV_PWR_W,
    NAV_H, PANEL_W, PREVIEW_H, SCREEN_H, SCREEN_W, STATS_H, STATS_Y,
    TITLE_H, VF_H, VF_Y,
)


# ---------- helpers ----------------------------------------------------------

def fit_to_panel(img, target_h=None, target_w=None):
    """Letterbox image into target_w x target_h without stretching.

    Defaults to main-UI viewfinder size (VF_H x PANEL_W).
    """
    th = VF_H if target_h is None else target_h
    tw = PANEL_W if target_w is None else target_w
    ih, iw = img.shape[:2]
    s = min(tw / iw, th / ih)
    nw, nh = int(iw * s), int(ih * s)
    out = np.zeros((th, tw, 3), np.uint8)
    out[(th - nh) // 2:(th - nh) // 2 + nh,
        (tw - nw) // 2:(tw - nw) // 2 + nw] = cv2.resize(img, (nw, nh))
    return out


def make_empty_right_panel(msg1, msg2="", target_h=None, target_w=None):
    th = VF_H if target_h is None else target_h
    tw = PANEL_W if target_w is None else target_w
    p = np.full((th, tw, 3), C["cream"], np.uint8)
    (w1, _), _ = cv2.getTextSize(msg1, FONT, 0.55, 1)
    cv2.putText(p, msg1, ((tw - w1) // 2, th // 2 - 10),
                FONT, 0.55, C["black"], 1)
    if msg2:
        (w2, _), _ = cv2.getTextSize(msg2, FONT, 0.42, 1)
        cv2.putText(p, msg2, ((tw - w2) // 2, th // 2 + 18),
                    FONT, 0.42, C["win_sh"], 1)
    return p


def _bevel(canvas, x1, y1, x2, y2, pressed=False):
    """Draw a 2-px Win9x chisel bevel. pressed inverts highlight/shadow."""
    hi = C["win_hi"] if not pressed else C["win_sh"]
    sh = C["win_sh"] if not pressed else C["win_hi"]
    # outer dark frame
    cv2.rectangle(canvas, (x1, y1), (x2, y2), C["win_dk"], 1)
    # inner bevel
    cv2.line(canvas, (x1 + 1, y1 + 1), (x2 - 1, y1 + 1), hi, 1)   # top
    cv2.line(canvas, (x1 + 1, y1 + 1), (x1 + 1, y2 - 1), hi, 1)   # left
    cv2.line(canvas, (x1 + 1, y2 - 1), (x2 - 1, y2 - 1), sh, 1)   # bottom
    cv2.line(canvas, (x2 - 1, y1 + 1), (x2 - 1, y2 - 1), sh, 1)   # right


# ---------- chrome -----------------------------------------------------------

def _title_bar(canvas, version):
    cv2.rectangle(canvas, (0, 0), (SCREEN_W, TITLE_H), C["navy"], -1)

    # Four-colour Windows-like logo
    lx, ly, sq, gap = 4, 4, 7, 2
    cv2.rectangle(canvas, (lx,               ly),
                  (lx + sq,            ly + sq),         (0, 0, 200),   -1)  # red
    cv2.rectangle(canvas, (lx + sq + gap,    ly),
                  (lx + 2*sq + gap,    ly + sq),         (40, 180, 40), -1)  # green
    cv2.rectangle(canvas, (lx,               ly + sq + gap),
                  (lx + sq,            ly + 2*sq + gap), (220, 80, 0),  -1)  # blue
    cv2.rectangle(canvas, (lx + sq + gap,    ly + sq + gap),
                  (lx + 2*sq + gap,    ly + 2*sq + gap), (0, 220, 220), -1)  # yellow

    title = f"openscanner  v{version}"
    cv2.putText(canvas, title, (30, 17), FONT, 0.52, C["white"], 1)

    # Clock (local HH:MM)
    clock = time.strftime("%H:%M")
    (tw, _), _ = cv2.getTextSize(clock, FONT, 0.52, 1)
    cv2.putText(canvas, clock, (SCREEN_W - tw - 6, 17),
                FONT, 0.52, C["white"], 1)

    # Tiny speaker icon just left of the clock
    sx = SCREEN_W - tw - 24
    cv2.rectangle(canvas, (sx, 9), (sx + 4, 16), C["white"], -1)
    pts = np.array([[sx + 4, 8], [sx + 10, 5], [sx + 10, 19], [sx + 4, 16]],
                   np.int32)
    cv2.fillPoly(canvas, [pts], C["white"])
    cv2.line(canvas, (sx + 12, 9), (sx + 14, 12), C["white"], 1)
    cv2.line(canvas, (sx + 12, 15), (sx + 14, 12), C["white"], 1)


def _nav_link(canvas, x, y, text, col, max_w=None):
    """Draw blue-underlined hyperlink text. Returns its pixel width."""
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
    cv2.putText(canvas, text, (x, y), FONT, 0.5, col, 1)
    cv2.line(canvas, (x, y + 2), (x + tw, y + 2), col, 1)
    return tw


def _nav_bar(canvas, state):
    y1, y2 = TITLE_H, TITLE_H + NAV_H
    cv2.rectangle(canvas, (0, y1), (SCREEN_W, y2), C["win_hi"], -1)
    cv2.line(canvas, (0, y2 - 1), (SCREEN_W, y2 - 1), C["win_sh"], 1)

    text_y = y1 + 17

    # Left: < 3D View
    _nav_link(canvas, 8, text_y, "< 3D View", C["link"])

    # Middle-right: Power
    _nav_link(canvas, NAV_PWR_X + 8, text_y, "Power", C["link"])

    # Right: Cal OK / Calibrate >
    cal_txt = "Cal OK >" if state.get("cal") else "Calibrate >"
    cal_col = (40, 140, 40) if state.get("cal") else C["link"]
    (cw, _), _ = cv2.getTextSize(cal_txt, FONT, 0.5, 1)
    _nav_link(canvas, SCREEN_W - cw - 8, text_y, cal_txt, cal_col)


# ---------- viewfinder overlays ---------------------------------------------

def _rangefinder(canvas, dist_m):
    if dist_m is not None:
        s = f"{dist_m * 100:.0f} cm" if dist_m < 1.0 else f"{dist_m:.2f} m"
        col = (30, 120, 30) if 0.2 <= dist_m <= 4.0 else (10, 120, 160)
    else:
        s = "-- m"; col = C["win_sh"]
    (tw, th), _ = cv2.getTextSize(s, FONT, 0.85, 2)
    px, py = PANEL_W // 2, VF_Y + 30
    x1, y1 = px - tw // 2 - 12, py - th - 8
    x2, y2 = px + tw // 2 + 12, py + 10
    # Cream pill with black outline
    cv2.rectangle(canvas, (x1, y1), (x2, y2), C["cream"], -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), C["black"], 1)
    cv2.putText(canvas, s, (px - tw // 2, py), FONT, 0.85, col, 2)


def _crosshair(canvas):
    cx, cy = PANEL_W // 2, VF_Y + VF_H // 2
    r = 14
    cv2.line(canvas, (cx - r, cy), (cx + r, cy), C["white"], 1)
    cv2.line(canvas, (cx, cy - r), (cx, cy + r), C["white"], 1)
    cv2.circle(canvas, (cx, cy), r // 2, C["white"], 1)


def _greeting(canvas, state):
    """Cheeky toolgun-style overlay near the bottom of the viewfinder."""
    if state.get("captures", 0) == 0:
        line = "You have not captured! Pull trigger to capture!"
    else:
        line = f"You have {state['captures']} captures! Nice."
    (tw, th), _ = cv2.getTextSize(line, FONT, 0.45, 1)
    x = (PANEL_W - tw) // 2
    y = VF_Y + VF_H - 10
    # semi-transparent black band behind text
    band = canvas.copy()
    cv2.rectangle(band, (0, y - th - 6), (PANEL_W, y + 4), (0, 0, 0), -1)
    cv2.addWeighted(band, 0.45, canvas, 0.55, 0, canvas)
    cv2.putText(canvas, line, (x, y), FONT, 0.45, C["white"], 1)


# ---------- stats + buttons --------------------------------------------------

def _stats_strip(canvas, state, sys):
    y1, y2 = STATS_Y, STATS_Y + STATS_H
    cv2.rectangle(canvas, (0, y1), (SCREEN_W, y2), C["olive_bg"], -1)
    cv2.line(canvas, (0, y1), (SCREEN_W, y1), C["win_dk"], 1)

    rect = "RECT" if state.get("cal") else "RAW "
    base = (f" Shots:{state['captures']:<3d}  {rect}  "
            f"{state['dist_mode']:<5}  v{state.get('version', '')}")
    cv2.putText(canvas, base, (4, y2 - 6), MONO_FONT, 1.05,
                C["olive_txt"], 1)

    parts = []
    if sys.get("temp") is not None:
        parts.append(f"{sys['temp']:.0f}C")
    for k, lbl in (("cpu", "CPU"), ("ram", "RAM"), ("disk", "DSK")):
        v = sys.get(k)
        if v is not None:
            parts.append(f"{lbl}:{v}%")
    if parts:
        right = "  ".join(parts)
        (tw, _), _ = cv2.getTextSize(right, MONO_FONT, 1.05, 1)
        cv2.putText(canvas, right, (SCREEN_W - tw - 6, y2 - 6),
                    MONO_FONT, 1.05, C["olive_txt"], 1)


def _bottom_buttons(canvas, state):
    # Chrome backplate
    cv2.rectangle(canvas, (0, BTN_Y), (SCREEN_W, SCREEN_H), C["win_body"], -1)
    cv2.line(canvas, (0, BTN_Y), (SCREEN_W, BTN_Y), C["win_hi"], 1)

    for i, (key, default_lbl, _on, _off) in enumerate(BTNS):
        bx1 = i * BTN_W + 4
        by1 = BTN_Y + 6
        bx2 = (i + 1) * BTN_W - 4
        by2 = SCREEN_H - 6

        active = False
        label = default_lbl
        if key == "dist":
            label = state["dist_mode"]; active = True
        elif key == "bgrem":
            active = state.get("bg_thresh") is not None
            label = "BG ON" if active else "BG OFF"
        elif key == "save":
            active = state.get("saving", False)
            label = "SAVING..." if active else "SAVE"

        face = C["win_face"] if not active else (180, 210, 180)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), face, -1)
        _bevel(canvas, bx1, by1, bx2, by2, pressed=active)

        (tw, th), _ = cv2.getTextSize(label, FONT, 0.72, 2)
        tx = bx1 + ((bx2 - bx1) - tw) // 2
        ty = by1 + ((by2 - by1) + th) // 2
        if key == "dist":
            ty -= 10
        cv2.putText(canvas, label, (tx, ty), FONT, 0.72, C["black"], 2)

        if key == "dist":
            rng = DIST_PRESETS[state["dist_mode"]]["label"]
            (sw, _), _ = cv2.getTextSize(rng, FONT, 0.36, 1)
            cv2.putText(canvas, rng,
                        (bx1 + ((bx2 - bx1) - sw) // 2, by2 - 14),
                        FONT, 0.36, C["win_sh"], 1)


# ---------- top-level draw ---------------------------------------------------

def draw_ui(canvas, state, left_panel, right_panel, sys):
    canvas[:] = C["win_body"]

    # Viewfinder panels
    canvas[VF_Y:VF_Y + VF_H, :PANEL_W] = left_panel
    cv2.line(canvas, (PANEL_W, VF_Y), (PANEL_W, VF_Y + VF_H), C["win_dk"], 1)
    canvas[VF_Y:VF_Y + VF_H, PANEL_W:] = right_panel

    # Right panel header pill
    rp_lbl = "COVERAGE" if state.get("has_disp") else "LIVE DEPTH"
    (rw, _), _ = cv2.getTextSize(rp_lbl, FONT, 0.4, 1)
    cv2.rectangle(canvas,
                  (PANEL_W + (PANEL_W - rw) // 2 - 6, VF_Y + 2),
                  (PANEL_W + (PANEL_W + rw) // 2 + 6, VF_Y + 18),
                  C["cream"], -1)
    cv2.rectangle(canvas,
                  (PANEL_W + (PANEL_W - rw) // 2 - 6, VF_Y + 2),
                  (PANEL_W + (PANEL_W + rw) // 2 + 6, VF_Y + 18),
                  C["black"], 1)
    cv2.putText(canvas, rp_lbl,
                (PANEL_W + (PANEL_W - rw) // 2, VF_Y + 15),
                FONT, 0.4, C["black"], 1)

    _rangefinder(canvas, state.get("crosshair_dist_m"))
    _crosshair(canvas)
    _greeting(canvas, state)

    if time.time() < state.get("flash_until", 0.0):
        cv2.rectangle(canvas, (0, VF_Y), (PANEL_W, VF_Y + VF_H),
                      C["white"], 8)

    _stats_strip(canvas, state, sys)
    _title_bar(canvas, state.get("version", ""))
    _nav_bar(canvas, state)
    _bottom_buttons(canvas, state)

    if state.get("saving"):
        _saving_overlay(canvas, state)


def _saving_overlay(canvas, state):
    """Win9x-style modal progress dialog."""
    x1, y1, x2, y2 = 150, 160, 650, 300
    # shadow
    cv2.rectangle(canvas, (x1 + 4, y1 + 4), (x2 + 4, y2 + 4), C["win_dk"], -1)
    # body
    cv2.rectangle(canvas, (x1, y1), (x2, y2), C["win_body"], -1)
    _bevel(canvas, x1, y1, x2, y2, pressed=False)
    # title bar
    cv2.rectangle(canvas, (x1 + 3, y1 + 3), (x2 - 3, y1 + 22), C["navy"], -1)
    cv2.putText(canvas, "Exporting...", (x1 + 10, y1 + 18),
                FONT, 0.5, C["white"], 1)

    kind = "USB drive" if state.get("save_kind") == "usb" else "ZIP archive"
    cv2.putText(canvas, f"Saving {state['captures']} captures to {kind}",
                (x1 + 16, y1 + 55), FONT, 0.5, C["black"], 1)

    # progress bar
    pct = max(0, min(100, state.get("zip_pct", 0)))
    pb_x1, pb_y1, pb_x2, pb_y2 = x1 + 16, y1 + 80, x2 - 16, y1 + 108
    cv2.rectangle(canvas, (pb_x1, pb_y1), (pb_x2, pb_y2), C["win_hi"], -1)
    _bevel(canvas, pb_x1, pb_y1, pb_x2, pb_y2, pressed=True)
    fill = pb_x1 + int((pb_x2 - pb_x1) * pct / 100)
    cv2.rectangle(canvas, (pb_x1 + 2, pb_y1 + 2), (fill, pb_y2 - 2),
                  C["navy"], -1)
    cv2.putText(canvas, f"{pct}%", (pb_x1 + 8, pb_y2 - 9),
                FONT, 0.45, C["white"] if pct > 10 else C["black"], 1)


# ---------- hit testing ------------------------------------------------------

def hit_button(x, y):
    """Map a touch (x,y) to a logical button name, or None."""
    # Nav bar (between title and viewfinder)
    if TITLE_H <= y < VF_Y:
        if NAV_3D_X <= x < NAV_3D_X + NAV_3D_W:
            return "view"
        if NAV_PWR_X <= x < NAV_PWR_X + NAV_PWR_W:
            return "power"
        if NAV_CAL_X <= x < NAV_CAL_X + NAV_CAL_W:
            return "cal"
    # Bottom button row
    if BTN_Y <= y <= SCREEN_H:
        i = x // BTN_W
        if 0 <= i < len(BTNS):
            return BTNS[i][0]
    return None


# ---------- modal dialog ----------------------------------------------------

def confirm_dialog(win, lines, buttons, callback_setter):
    """Win9x-ish modal with title bar + buttons.

    buttons: list of (label, colour, action_key).
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
        ov = np.full((SCREEN_H, SCREEN_W, 3), C["win_body"], np.uint8)

        # Dialog body
        dx1, dy1, dx2, dy2 = 40, 60, SCREEN_W - 40, PREVIEW_H - 20
        cv2.rectangle(ov, (dx1 + 4, dy1 + 4), (dx2 + 4, dy2 + 4),
                      C["win_dk"], -1)
        cv2.rectangle(ov, (dx1, dy1), (dx2, dy2), C["win_body"], -1)
        _bevel(ov, dx1, dy1, dx2, dy2, pressed=False)

        # Title bar
        cv2.rectangle(ov, (dx1 + 3, dy1 + 3), (dx2 - 3, dy1 + 24),
                      C["navy"], -1)
        title = lines[0][0] if isinstance(lines[0], tuple) else lines[0]
        cv2.putText(ov, title, (dx1 + 10, dy1 + 19),
                    FONT, 0.5, C["white"], 1)

        for i, line in enumerate(lines[1:], start=1):
            col = line[1] if isinstance(line, tuple) else C["black"]
            txt = line[0] if isinstance(line, tuple) else line
            if not txt:
                continue
            (tw, _), _ = cv2.getTextSize(txt, FONT, 0.5, 1)
            cv2.putText(ov, txt, ((SCREEN_W - tw) // 2, dy1 + 55 + i * 28),
                        FONT, 0.5, col, 1)

        # Buttons along the bottom (Win9x style raised)
        bw = SCREEN_W // len(buttons)
        for i, (lbl, col, _) in enumerate(buttons):
            bx1 = i * bw + 4
            bx2 = (i + 1) * bw - 4
            by1 = BTN_Y + 6
            by2 = SCREEN_H - 6
            cv2.rectangle(ov, (bx1, by1), (bx2, by2), col, -1)
            _bevel(ov, bx1, by1, bx2, by2, pressed=False)
            (tw, th), _ = cv2.getTextSize(lbl, FONT, 0.7, 2)
            cv2.putText(ov, lbl,
                        (bx1 + ((bx2 - bx1) - tw) // 2,
                         by1 + ((by2 - by1) + th) // 2),
                        FONT, 0.7, C["white"], 2)

        cv2.imshow(win, ov)
        cv2.waitKey(50)

    return state["action"]
