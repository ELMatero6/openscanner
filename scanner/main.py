"""Main capture loop. Wires together camera, stereo, UI, export, settings."""

import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np

try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO_OK = True
except Exception:
    GPIO_OK = False

from . import __version__
from . import logger as log_mod
from . import settings as settings_mod
from .calibration import load_calibration, rectify, run_wizard
from .config import (
    AUTO_INTERVAL, BTN_Y, C, CALIBRATION_FILE, CAMERA_INDEX,
    CAPTURE_HEIGHT, CAPTURE_WIDTH, DEFAULT_BASELINE_MM, DIST_ORDER,
    DIST_PRESETS, FONT, GPIO_PIN, PANEL_W, PREVIEW_H, SAVE_DIR,
    SCREEN_H, SCREEN_W, SETTINGS_FILE, SHUTDOWN_HOLD_S, VF_H,
)
from .export import (
    fuse_plys, init_csv, save_capture, usb_export, usb_find_block_device,
    usb_format_vfat, usb_mount, usb_unmount_and_eject, usb_writable,
    zip_export,
)
from .stereo import (
    build_live_matcher, build_matcher, compute_disparity,
    sample_disp_at,
)
from .sysinfo import cpu_temp, git_sha, shutdown, sys_stats
from .ui import confirm_dialog, draw_ui, fit_to_panel, hit_button, make_empty_right_panel
from . import viewer


def _gpio_low():
    return GPIO_OK and GPIO.input(GPIO_PIN) == GPIO.LOW


def _open_camera():
    """Open the stereo camera with retry. Returns cap or None."""
    for attempt in range(3):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if cap.isOpened():
            return cap
        cap.release()
        time.sleep(1.0)
    return None


def _ply_paths():
    """Sorted list of per-capture PLYs in SAVE_DIR (excludes fused.ply)."""
    return sorted(
        os.path.join(SAVE_DIR, f) for f in os.listdir(SAVE_DIR)
        if f.endswith(".ply") and not f.startswith("fused")
    )


def _confirm_shutdown(win):
    """Two-tap power-off confirmation."""
    holder = {"cb": None, "state": None}
    def setter(cb, state):
        holder["cb"] = cb; holder["state"] = state
        cv2.setMouseCallback(win, cb, state)
    action = confirm_dialog(
        win,
        [("Shut down the scanner?", C["white"]),
         "Wait 5s after power-off LED stops blinking",
         "before unplugging power."],
        [("SHUT DOWN", C["red"], "yes"), ("CANCEL", C["dim"], "no")],
        setter,
    )
    return action == "yes"


def _save_destination(win):
    """Ask LOCAL / USB / CANCEL."""
    holder = {"cb": None, "state": None}
    def setter(cb, state):
        holder["cb"] = cb; holder["state"] = state
        cv2.setMouseCallback(win, cb, state)
    return confirm_dialog(
        win,
        [("Where do you want to save?", C["white"])],
        [("LOCAL ZIP", C["blue"], "local"),
         ("USB DRIVE", C["purple"], "usb"),
         ("CANCEL",    C["dim"],    "cancel")],
        setter,
    )


def _do_local_zip(state, version, cal, log):
    out = f"{SAVE_DIR}__{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    state["saving"] = True
    state["zip_pct"] = 0
    state["save_kind"] = "zip"
    state.pop("zip_msg", None)
    log.info("local zip: starting -> %s", out)
    def _bg():
        try:
            zip_export(SAVE_DIR, out, cal, version,
                       progress_cb=lambda p: state.update(zip_pct=p))
            state["zip_done"] = out
            state["zip_msg"] = (True, out)
            log.info("local zip: done -> %s", out)
        except Exception as e:
            state["zip_msg"] = (False, str(e))
            log.exception("local zip: failed")
        finally:
            state["saving"] = False
    threading.Thread(target=_bg, daemon=True).start()


def _usb_format_confirm(win):
    """Ask FORMAT / APPEND / CANCEL. Returns one of those keys."""
    def setter(cb, st):
        cv2.setMouseCallback(win, cb, st)
    return confirm_dialog(
        win,
        [("USB drive options", C["white"]),
         "FORMAT wipes the drive first (FAT32).",
         "APPEND writes a new folder beside any",
         "existing data on the drive."],
        [("FORMAT",  C["red"],    "format"),
         ("APPEND",  C["blue"],   "append"),
         ("CANCEL",  C["dim"],    "cancel")],
        setter,
    )


def _do_usb_export(win, state, version, cal, log):
    """Detect + mount USB, optionally format, export, then eject. Async.

    The main loop watches state["usb_msg"] and shows the result dialog so
    the UI stays responsive during multi-minute copies.
    """
    dev, parent, mnt = usb_find_block_device()
    def setter(cb, st):
        cv2.setMouseCallback(win, cb, st)

    if not dev:
        log.warning("usb export: no drive detected")
        confirm_dialog(win,
            [("No USB drive detected", C["red"]),
             "Plug in a USB drive (FAT32/exFAT/ext4)",
             "and try again."],
            [("OK", C["blue"], "ok")], setter)
        return

    choice = _usb_format_confirm(win)
    cv2.setMouseCallback(win, lambda *a, **k: None, None)
    if choice == "cancel":
        log.info("usb export: cancelled by user")
        return

    do_format = (choice == "format")
    log.info("usb export: dev=%s parent=%s mnt=%s format=%s",
             dev, parent, mnt, do_format)

    state["saving"] = True
    state["zip_pct"] = 0
    state["save_kind"] = "usb"
    state.pop("usb_msg", None)

    def _bg():
        try:
            target = mnt
            if do_format:
                state["zip_pct"] = 5
                if not usb_format_vfat(dev, parent, label="SCANNER"):
                    state["usb_msg"] = (False, "Format failed - see log")
                    return
                target = None  # force remount below

            if not target:
                state["zip_pct"] = 10
                target = usb_mount(dev)
                if not target:
                    state["usb_msg"] = (False, "Mount failed - see log")
                    return

            if not usb_writable(target):
                state["usb_msg"] = (False, f"{target} not writable")
                return

            ok, msg = usb_export(SAVE_DIR, target, cal, version,
                                 progress_cb=lambda p: state.update(zip_pct=p))
            log.info("usb export: %s (%s)", "ok" if ok else "fail", msg)

            if ok:
                state["zip_pct"] = 100
                ejected = usb_unmount_and_eject(dev, parent)
                msg = (msg + "  -  Safe to unplug."
                       if ejected else msg + "  -  NOT ejected; sync manually.")
            state["usb_msg"] = (ok, msg)
        except Exception as e:
            state["usb_msg"] = (False, f"Error: {e}")
            log.exception("usb export: unexpected error")
        finally:
            state["saving"] = False
    threading.Thread(target=_bg, daemon=True).start()


def run():
    settings = settings_mod.load(SETTINGS_FILE)
    log_path = log_mod.init(SAVE_DIR, settings.get("log_level", "INFO"))
    log = log_mod.get("scanner.main")

    log.info("=" * 60)
    log.info("openscanner v%s  (%s)", __version__, git_sha())
    log.info("save dir: %s  log: %s", SAVE_DIR, log_path)
    log.info("GPIO: %s", f"pin {GPIO_PIN}" if GPIO_OK else "not available")
    log.info("settings: %s", settings)
    log.info("=" * 60)

    cap = _open_camera()
    if cap is None:
        log.error("camera unavailable after 3 tries - exiting")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info("camera open: %dx%d", actual_w, actual_h)

    cal = load_calibration(CALIBRATION_FILE, (actual_w // 2, actual_h))

    dist_mode = settings["dist_mode"] if settings["dist_mode"] in DIST_PRESETS else "MED"
    matcher   = build_matcher(**{k: DIST_PRESETS[dist_mode][k] for k in ("num_disp", "block")})
    live_m    = build_live_matcher()
    csv_path  = init_csv(SAVE_DIR)

    WIN = "Scanner"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    canvas = np.zeros((SCREEN_H, SCREEN_W, 3), np.uint8)

    state = {
        "cal":              cal,
        "mode":             "semi",
        "captures":         sum(1 for f in os.listdir(SAVE_DIR) if f.endswith("_L.png")),
        "saving":           False,
        "zip_pct":          0,
        "flash_until":      0.0,
        "has_disp":         False,
        "dist_mode":        dist_mode,
        "bg_thresh":        None,
        "crosshair_dist_m": None,
        "version":          git_sha(),
        "baseline_mm":      cal["baseline_mm"] if cal else DEFAULT_BASELINE_MM,
    }

    sys_holder = {}

    def _poll_stats():
        while True:
            s = sys_stats(SAVE_DIR)
            s["temp"] = cpu_temp()
            sys_holder.update(s)
            time.sleep(2.0)
    threading.Thread(target=_poll_stats, daemon=True).start()

    right_panel = make_empty_right_panel("No scans yet", "Pull trigger to capture")
    coverage_acc = None
    coverage_cnt = 0

    touch = {"action": None}
    def on_touch(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            btn = hit_button(x, y)
            if btn:
                param["action"] = btn
    cv2.setMouseCallback(WIN, on_touch, touch)

    last_auto       = 0.0
    gpio_was        = False
    gpio_press_time = 0.0
    rf_buffer       = []

    log.info("ready - entering capture loop")

    left = right = None
    left_panel = make_empty_right_panel("Camera starting...", "")
    rf_disp = np.zeros((120, 160), dtype=np.float32)
    cam_fail_count = 0

    while True:
        ret, frame = cap.read()
        now = time.time()

        if ret:
            cam_fail_count = 0
            mid   = frame.shape[1] // 2
            left  = frame[:, :mid]
            right = frame[:, mid:]

            if state["cal"]:
                left, right = rectify(left, right, state["cal"])

            # Viewfinder = rectified left eye, letterboxed
            left_panel = fit_to_panel(left)

            # Live rangefinder + BG mask source - cheap matcher at 160x120
            RF_W, RF_H = 160, 120
            gl_rf = cv2.resize(cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY), (RF_W, RF_H))
            gr_rf = cv2.resize(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), (RF_W, RF_H))
            rf_disp = live_m.compute(gl_rf, gr_rf).astype(np.float32) / 16.0

            # Distance at crosshair - resolution-independent sampling
            d_at_cross = sample_disp_at(rf_disp, 0.5, 0.5, 0.08)
            if d_at_cross and d_at_cross > 0 and state["cal"]:
                # focal_px scales with image; at RF_W the focal is focal_full * RF_W/full
                f_rf = state["cal"]["focal_px"] * (RF_W / (actual_w // 2))
                raw_m = (state["cal"]["baseline_mm"] / 1000.0 * f_rf) / d_at_cross
                if 0.10 <= raw_m <= 8.0:
                    rf_buffer.append(raw_m)
                    if len(rf_buffer) > 8:
                        rf_buffer.pop(0)
            if rf_buffer:
                state["crosshair_dist_m"] = round(float(np.median(rf_buffer)), 2)
            else:
                state["crosshair_dist_m"] = None

            # Live BG removal: mask the viewfinder using rf_disp upscaled
            if state["bg_thresh"] is not None:
                disp_up = cv2.resize(rf_disp, (PANEL_W, VF_H))
                mask = (disp_up >= state["bg_thresh"] * 0.85).astype(np.uint8)
                kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
                left_panel = (left_panel * mask[:, :, None]).astype(np.uint8)
        else:
            # Camera read failed - common during heavy USB writes sharing the bus.
            # Keep UI/state/touches alive with the last good viewfinder so we don't
            # freeze on an overlay and miss async completion dialogs.
            cam_fail_count += 1
            if cam_fail_count in (10, 100, 500):
                log.warning("camera read returning no frame (count=%d)", cam_fail_count)

        # GPIO
        gpio_now = _gpio_low()
        gpio_trigger = gpio_now and not gpio_was
        if gpio_now and not gpio_was:
            gpio_press_time = now
        if gpio_now and (now - gpio_press_time) > SHUTDOWN_HOLD_S:
            if _confirm_shutdown(WIN):
                shutdown(); return
            gpio_press_time = now + 1e9
            cv2.setMouseCallback(WIN, on_touch, touch)
        gpio_was = gpio_now

        action = touch["action"]; touch["action"] = None

        if action == "dist":
            i = DIST_ORDER.index(state["dist_mode"])
            state["dist_mode"] = DIST_ORDER[(i + 1) % len(DIST_ORDER)]
            p = DIST_PRESETS[state["dist_mode"]]
            matcher = build_matcher(p["num_disp"], p["block"])
            settings["dist_mode"] = state["dist_mode"]
            settings_mod.save(SETTINGS_FILE, settings)

        elif action == "bgrem":
            if state["bg_thresh"] is not None:
                state["bg_thresh"] = None
                settings["bg_on"] = False
                log.info("bg removal: off")
            else:
                t = sample_disp_at(rf_disp, 0.5, 0.5, 0.08)
                if t and t > 0:
                    state["bg_thresh"] = t
                    settings["bg_on"] = True
                    log.info("bg removal: on (threshold=%.2f)", t)
                else:
                    log.warning("bg removal: no valid depth at crosshair")
            settings_mod.save(SETTINGS_FILE, settings)

        elif action == "save":
            if state["captures"] == 0:
                log.info("save: nothing to save")
            elif not state["saving"]:
                dest = _save_destination(WIN)
                cv2.setMouseCallback(WIN, on_touch, touch)
                if dest == "local":
                    settings["last_save_dest"] = "local"
                    settings_mod.save(SETTINGS_FILE, settings)
                    _do_local_zip(state, __version__, state["cal"], log)
                elif dest == "usb":
                    settings["last_save_dest"] = "usb"
                    settings_mod.save(SETTINGS_FILE, settings)
                    _do_usb_export(WIN, state, __version__, state["cal"], log)

        elif action == "clear":
            state["captures"] = 0
            state["has_disp"] = False
            coverage_acc = None
            coverage_cnt = 0
            right_panel = make_empty_right_panel("No scans yet", "Pull trigger to capture")
            for f in os.listdir(SAVE_DIR):
                if f.endswith((".png", ".csv", ".ply", ".txt")):
                    os.remove(os.path.join(SAVE_DIR, f))
            init_csv(SAVE_DIR)

        elif action == "cal":
            new_cal = run_wizard(cap, actual_w, actual_h, WIN,
                                 CALIBRATION_FILE, GPIO_OK, _gpio_low)
            cv2.setMouseCallback(WIN, on_touch, touch)
            if new_cal:
                state["cal"] = new_cal
                state["baseline_mm"] = new_cal["baseline_mm"]

        elif action == "view":
            paths = _ply_paths()
            viewer.show(WIN, paths,
                        max_points=settings.get("viewer_subsample", 60000))
            cv2.setMouseCallback(WIN, on_touch, touch)

        elif action == "power":
            if _confirm_shutdown(WIN):
                shutdown(); return
            cv2.setMouseCallback(WIN, on_touch, touch)

        # Show the result dialog for an async save when the worker finishes.
        # We only pop this when saving is False so progress overlay stays smooth.
        if not state["saving"]:
            def _setter(cb, st):
                cv2.setMouseCallback(WIN, cb, st)
            if "usb_msg" in state:
                ok, msg = state.pop("usb_msg")
                confirm_dialog(WIN,
                    [(("Export complete!" if ok else "Export failed"),
                      C["green"] if ok else C["red"]),
                     msg,
                     ("Safe to unplug USB drive." if ok else "")],
                    [("OK", C["blue"], "ok")], _setter)
                cv2.setMouseCallback(WIN, on_touch, touch)
            elif "zip_msg" in state:
                ok, msg = state.pop("zip_msg")
                confirm_dialog(WIN,
                    [(("Zip complete!" if ok else "Zip failed"),
                      C["green"] if ok else C["red"]),
                     (os.path.basename(msg) if ok else msg)],
                    [("OK", C["blue"], "ok")], _setter)
                cv2.setMouseCallback(WIN, on_touch, touch)

        # Capture trigger - only when we have fresh L/R frames
        do_capture = False
        if ret and state["mode"] == "semi" and gpio_trigger:
            do_capture = True
        elif ret and state["mode"] == "auto" and gpio_now and now - last_auto >= AUTO_INTERVAL:
            do_capture = True

        if do_capture:
            last_auto = now
            disp = compute_disparity(left, right, matcher)

            # Coverage map
            disp_pos = np.clip(disp, 0, None)
            if coverage_acc is None:
                coverage_acc = disp_pos.copy()
            else:
                if disp_pos.shape != coverage_acc.shape:
                    disp_pos = cv2.resize(disp_pos,
                                          (coverage_acc.shape[1], coverage_acc.shape[0]))
                coverage_acc = np.maximum(coverage_acc, disp_pos)
            coverage_cnt += 1

            cov_norm = cv2.normalize(coverage_acc, None, 0, 255,
                                     cv2.NORM_MINMAX).astype(np.uint8)
            cov_col = cv2.applyColorMap(cov_norm, cv2.COLORMAP_TURBO)
            cov_col[coverage_acc <= 0] = 0
            right_panel = cv2.resize(cov_col, (PANEL_W, VF_H))
            cv2.putText(right_panel, f"Coverage  ({coverage_cnt} shots)",
                        (8, 20), FONT, 0.45, (255, 255, 255), 1)
            state["has_disp"] = True

            # Apply BG mask consistently (sample + apply at same disp scale)
            save_left, save_right, save_disp = left, right, disp
            if state["bg_thresh"] is not None:
                # Resample the BG threshold from the high-res disp to match scale
                hi_thresh = sample_disp_at(disp, 0.5, 0.5, 0.08)
                if hi_thresh and hi_thresh > 0:
                    mask = (disp >= hi_thresh * 0.85).astype(np.uint8)
                    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
                    mask_full = cv2.resize(mask, (left.shape[1], left.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)
                    save_left  = (left  * mask_full[:, :, None]).astype(np.uint8)
                    save_right = (right * mask_full[:, :, None]).astype(np.uint8)
                    save_disp  = disp * mask

            idx = state["captures"] + 1
            meta = save_capture(SAVE_DIR, idx, save_left, save_right, save_disp,
                                state["cal"], state, csv_path)
            state["captures"] = idx
            state["flash_until"] = now + 0.15
            valid_px = int((save_disp > 0).sum())
            total_px = int(save_disp.size)
            cov = 100.0 * valid_px / max(total_px, 1)
            log.info("capture #%d  coverage=%.1f%%  dist=%s  ply=%s",
                     idx, cov, state["dist_mode"],
                     "yes" if meta and meta.get("PLY") and state["cal"] else "no")

        sys_holder.setdefault("temp", None)
        draw_ui(canvas, state, left_panel, right_panel, sys_holder)
        cv2.imshow(WIN, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    if GPIO_OK:
        GPIO.cleanup()
    log.info("shutdown clean")
