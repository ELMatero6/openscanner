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
from . import display
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
from .sysinfo import (
    apply_update, check_for_updates, cpu_temp, git_sha, shutdown, sys_stats,
)
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


class CameraReader:
    """Reads frames off a cv2.VideoCapture on a dedicated thread.

    cv2.VideoCapture.read() can block for hundreds of milliseconds when
    the USB bus is contended (e.g. during a USB write). Keeping reads off
    the main thread means the UI keeps rendering at full rate even when
    the camera stalls. We always expose the latest frame; stale frames
    are fine because the UI handles "no fresh frame" gracefully.
    """

    def __init__(self, cap):
        self.cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._ts = 0.0
        self._ok = 0
        self._fail = 0
        self._stop = False
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while not self._stop:
            try:
                ret, frame = self.cap.read()
            except Exception:
                ret = False
                frame = None
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame
                    self._ts = time.time()
                    self._ok += 1
            else:
                self._fail += 1
                time.sleep(0.02)

    def read(self, max_age=0.5):
        """Return (frame, age_s). frame=None if no fresh frame available."""
        with self._lock:
            f = self._frame
            ts = self._ts
        if f is None:
            return None, float("inf")
        age = time.time() - ts
        return (f if age <= max_age else None), age

    def stats(self):
        return self._ok, self._fail

    def stop(self):
        self._stop = True
        try:
            self._t.join(timeout=1.0)
        except Exception:
            pass


class StereoWorker:
    """Runs the stereo pipeline off the main thread.

    pygame/SDL2 rendering has to stay on the main thread, but rectify +
    SGBM + BG-mask do not. This worker pulls the latest frame from the
    CameraReader, rectifies, computes live rf_disp, applies BG mask,
    builds the viewfinder panel, and publishes a snapshot for the main
    loop to blit.

    Capture trigger is asynchronous too: main calls request_capture(),
    the worker runs the expensive full-res SGBM on its next frame, and
    main picks up the result via poll_capture() to hand off to save.

    Shared state flows through `state` (reads) and the internal lock
    (snapshot + capture handoff). We intentionally NEVER touch pygame
    or any SDL call from this thread.
    """

    def __init__(self, cam, state, live_m, actual_w):
        self.cam         = cam
        self.state       = state
        self.live_m      = live_m
        self.actual_half = actual_w // 2

        self._snap_lock  = threading.Lock()
        self._snap       = None

        self._cap_lock   = threading.Lock()
        self._cap_req    = False
        self._cap_result = None

        self._pause      = threading.Event()
        self._stop       = False
        self._t          = threading.Thread(target=self._loop, daemon=True,
                                            name="stereo")
        self._t.start()

    def _loop(self):
        rf_buffer = []
        while not self._stop:
            if self._pause.is_set():
                time.sleep(0.05)
                continue
            frame, _age = self.cam.read(max_age=0.5)
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                self._process(frame, rf_buffer)
            except Exception:
                # Never let a bad frame kill the worker silently.
                import traceback
                traceback.print_exc()
                time.sleep(0.05)

    def _process(self, frame, rf_buffer):
        mid   = frame.shape[1] // 2
        left  = frame[:, :mid]
        right = frame[:, mid:]

        cal = self.state.get("cal")
        if cal:
            left, right = rectify(left, right, cal)

        RF_W, RF_H = 160, 120
        gl_rf = cv2.resize(cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY), (RF_W, RF_H))
        gr_rf = cv2.resize(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), (RF_W, RF_H))
        rf_disp = self.live_m.compute(gl_rf, gr_rf).astype(np.float32) / 16.0

        d_at_cross = sample_disp_at(rf_disp, 0.5, 0.5, 0.08)
        if d_at_cross and d_at_cross > 0 and cal:
            f_rf  = cal["focal_px"] * (RF_W / self.actual_half)
            raw_m = (cal["baseline_mm"] / 1000.0 * f_rf) / d_at_cross
            if 0.10 <= raw_m <= 8.0:
                rf_buffer.append(raw_m)
                if len(rf_buffer) > 8:
                    rf_buffer.pop(0)
        crosshair_m = (round(float(np.median(rf_buffer)), 2)
                       if rf_buffer else None)
        # Single-key dict writes are atomic under CPython's GIL, safe to
        # publish directly to `state` so draw_ui sees the fresh value.
        self.state["crosshair_dist_m"] = crosshair_m

        bg_thresh = self.state.get("bg_thresh")
        if bg_thresh is not None:
            mask  = _bg_mask_from_rf(rf_disp, bg_thresh,
                                     (left.shape[0], left.shape[1]))
            left  = left  * mask[:, :, None]
            right = right * mask[:, :, None]

        left_panel = fit_to_panel(left)

        with self._snap_lock:
            self._snap = {
                "left":       left,
                "right":      right,
                "rf_disp":    rf_disp,
                "left_panel": left_panel,
            }

        do_capture = False
        with self._cap_lock:
            if self._cap_req:
                self._cap_req = False
                do_capture = True
        if do_capture:
            matcher = self.state.get("matcher")
            if matcher is not None:
                disp = compute_disparity(left, right, matcher)
                with self._cap_lock:
                    self._cap_result = (left.copy(), right.copy(), disp)

    def latest(self):
        with self._snap_lock:
            return self._snap

    def request_capture(self):
        with self._cap_lock:
            self._cap_req = True

    def poll_capture(self):
        with self._cap_lock:
            r = self._cap_result
            self._cap_result = None
            return r

    def pause(self):
        """Freeze the worker so another consumer (wizard/viewer) owns cap."""
        self._pause.set()

    def resume(self):
        self._pause.clear()

    def stop(self):
        self._stop = True
        try:
            self._t.join(timeout=1.0)
        except Exception:
            pass


def _bg_mask_from_rf(rf_disp, bg_thresh, out_shape):
    """Build a foreground mask from the cheap live disparity.

    Covers the full viewfinder width, including the SGBM "invalid"
    stripe on the left edge: SGBM leaves the leftmost ~num_disp columns
    with disp=-1, so a naive threshold would chop off anything in that
    strip. We propagate the first valid column's mask leftward to fill
    it, so the mask follows the subject rather than the matcher's
    blind spot.

    The mask is then horizontally dilated to cover the right eye's
    parallax-shifted view of the same object, and nearest-upscaled
    to the full frame so we can apply it to both eyes before SGBM.
    """
    mask_lo = (rf_disp >= bg_thresh * 0.85).astype(np.uint8)

    # Fill the SGBM-invalid left stripe (disp<0) by copying the first
    # valid column across it. This is what stops the "one side cut off"
    # artifact - the subject is preserved to the left edge of the frame.
    valid_col = (rf_disp >= 0).any(axis=0)
    if valid_col.any():
        first = int(np.argmax(valid_col))
        if first > 0:
            mask_lo[:, :first] = mask_lo[:, first:first + 1]

    # Horizontal dilation ~ max live-disparity so the right eye's
    # shifted view of the object still falls inside the mask. 41 at
    # 160 wide is about 25% - covers tight close-up subjects too.
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 7))
    mask_lo = cv2.morphologyEx(mask_lo, cv2.MORPH_CLOSE, kern)
    mask_lo = cv2.morphologyEx(mask_lo, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    h, w = out_shape
    return cv2.resize(mask_lo, (w, h), interpolation=cv2.INTER_NEAREST)


def _ply_paths():
    """Sorted list of per-capture PLYs in SAVE_DIR (excludes concat.ply)."""
    return sorted(
        os.path.join(SAVE_DIR, f) for f in os.listdir(SAVE_DIR)
        if f.endswith(".ply") and not f.startswith(("fused", "concat"))
    )


def _power_menu():
    """Top-level power screen. Returns 'shutdown' | 'update' | 'cancel'."""
    return confirm_dialog(
        [("Power options", C["white"])],
        [("SHUT DOWN",    C["red"],    "shutdown"),
         ("CHECK UPDATES", C["blue"],  "update"),
         ("CANCEL",       C["dim"],    "cancel")],
    )


def _confirm_shutdown():
    """Final shutdown confirmation."""
    action = confirm_dialog(
        [("Shut down the scanner?", C["white"]),
         "Wait 5s after power-off LED stops blinking",
         "before unplugging power."],
        [("SHUT DOWN", C["red"], "yes"), ("CANCEL", C["dim"], "no")],
    )
    return action == "yes"


def _do_update_check(log):
    """Run the check-for-updates flow. Blocks briefly on git fetch.

    Returns True if the user asked us to install and we should exit for
    a systemd-driven restart. Returns False otherwise (up to date, user
    cancelled, or fetch failed).
    """
    log.info("update check: fetching origin")
    ok, behind, branch, detail = check_for_updates()
    if not ok:
        log.warning("update check failed: %s", detail)
        confirm_dialog(
            [("Update check failed", C["red"]), detail or "unknown error"],
            [("OK", C["blue"], "ok")],
        )
        return False

    if behind == 0:
        log.info("update check: already at origin/%s", branch)
        confirm_dialog(
            [("Already up to date", C["green"]),
             f"On origin/{branch} @ {git_sha()}"],
            [("OK", C["blue"], "ok")],
        )
        return False

    log.info("update check: %d commit(s) behind origin/%s", behind, branch)
    choice = confirm_dialog(
        [(f"{behind} new commit{'s' if behind != 1 else ''} available",
          C["green"]),
         f"Branch: {branch}",
         "Install now? The scanner will restart."],
        [("INSTALL", C["blue"], "yes"), ("CANCEL", C["dim"], "no")],
    )
    if choice != "yes":
        log.info("update: cancelled by user")
        return False

    pulled, detail = apply_update(branch)
    if not pulled:
        log.warning("update pull failed: %s", detail)
        confirm_dialog(
            [("Update failed", C["red"]), detail or "git pull failed"],
            [("OK", C["blue"], "ok")],
        )
        return False

    log.info("update: pulled, requesting restart")
    confirm_dialog(
        [("Update installed", C["green"]),
         "Restarting now..."],
        [("OK", C["blue"], "ok")],
    )
    return True


def _save_destination():
    """Ask LOCAL / USB / CANCEL."""
    return confirm_dialog(
        [("Where do you want to save?", C["white"])],
        [("LOCAL ZIP", C["blue"], "local"),
         ("USB DRIVE", C["purple"], "usb"),
         ("CANCEL",    C["dim"],    "cancel")],
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


def _usb_format_confirm():
    """Ask FORMAT / APPEND / CANCEL. Returns one of those keys."""
    return confirm_dialog(
        [("USB drive options", C["white"]),
         "FORMAT wipes the drive first (FAT32).",
         "APPEND writes a new folder beside any",
         "existing data on the drive."],
        [("FORMAT",  C["red"],    "format"),
         ("APPEND",  C["blue"],   "append"),
         ("CANCEL",  C["dim"],    "cancel")],
    )


def _do_usb_export(state, version, cal, log):
    """Detect + mount USB, optionally format, export, then eject. Async.

    The main loop watches state["usb_msg"] and shows the result dialog so
    the UI stays responsive during multi-minute copies.
    """
    dev, parent, mnt = usb_find_block_device()

    if not dev:
        log.warning("usb export: no drive detected")
        confirm_dialog(
            [("No USB drive detected", C["red"]),
             "Plug in a USB drive (FAT32/exFAT/ext4)",
             "and try again."],
            [("OK", C["blue"], "ok")])
        return

    choice = _usb_format_confirm()
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

    cam = CameraReader(cap)

    cal = load_calibration(CALIBRATION_FILE, (actual_w // 2, actual_h))

    dist_mode = settings["dist_mode"] if settings["dist_mode"] in DIST_PRESETS else DIST_ORDER[0]
    matcher   = build_matcher(**{k: DIST_PRESETS[dist_mode][k] for k in ("num_disp", "block")})
    live_m    = build_live_matcher()
    csv_path  = init_csv(SAVE_DIR)

    display.init(SCREEN_W, SCREEN_H, title="openscanner")

    canvas = np.zeros((SCREEN_H, SCREEN_W, 3), np.uint8)

    # `state` is the shared bus between the main thread and the stereo
    # worker. Main writes most fields (cal, dist_mode, bg_thresh, matcher);
    # worker reads them and publishes crosshair_dist_m back. Single-key
    # dict writes are atomic under CPython so no lock is needed here.
    state = {
        "cal":              cal,
        "matcher":          matcher,
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

    worker = StereoWorker(cam, state, live_m, actual_w)

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

    touch = {"action": None, "quit": False}
    def _drain_input():
        for ev in display.events():
            if isinstance(ev, display.Tap):
                btn = hit_button(ev.x, ev.y)
                if btn:
                    touch["action"] = btn
            elif isinstance(ev, display.Quit):
                touch["quit"] = True
            elif isinstance(ev, display.Key) and ev.char in ("q", "\x1b"):
                touch["quit"] = True

    last_auto       = 0.0
    gpio_was        = False
    gpio_press_time = 0.0

    log.info("ready - entering capture loop")

    left_panel = make_empty_right_panel("Camera starting...", "")
    rf_disp = np.zeros((120, 160), dtype=np.float32)
    heartbeat_t = time.time()
    heartbeat_frames = 0

    while True:
        now = time.time()
        # The worker does all the stereo work; main thread just reads its
        # last snapshot for rendering. This is what keeps the UI fluid.
        snap = worker.latest()
        ret = snap is not None
        if ret:
            left_panel = snap["left_panel"]
            rf_disp    = snap["rf_disp"]

        # Heartbeat every ~5s so we can spot stalls / drop in framerate
        heartbeat_frames += 1
        if now - heartbeat_t >= 5.0:
            fps = heartbeat_frames / (now - heartbeat_t)
            ok, fail = cam.stats()
            log.info("heartbeat: fps=%.1f cam_ok=%d cam_fail=%d",
                     fps, ok, fail)
            heartbeat_t = now
            heartbeat_frames = 0

        # GPIO
        gpio_now = _gpio_low()
        gpio_trigger = gpio_now and not gpio_was
        if gpio_now and not gpio_was:
            gpio_press_time = now
        if gpio_now and (now - gpio_press_time) > SHUTDOWN_HOLD_S:
            if _confirm_shutdown():
                shutdown(); return
            gpio_press_time = now + 1e9
        gpio_was = gpio_now

        _drain_input()
        if touch["quit"]:
            break
        action = touch["action"]; touch["action"] = None

        if action == "dist":
            i = DIST_ORDER.index(state["dist_mode"])
            state["dist_mode"] = DIST_ORDER[(i + 1) % len(DIST_ORDER)]
            p = DIST_PRESETS[state["dist_mode"]]
            # Publish the new matcher through `state` so the worker picks
            # it up on its next capture request.
            state["matcher"] = build_matcher(p["num_disp"], p["block"])
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
                dest = _save_destination()
                if dest == "local":
                    settings["last_save_dest"] = "local"
                    settings_mod.save(SETTINGS_FILE, settings)
                    _do_local_zip(state, __version__, state["cal"], log)
                elif dest == "usb":
                    settings["last_save_dest"] = "usb"
                    settings_mod.save(SETTINGS_FILE, settings)
                    _do_usb_export(state, __version__, state["cal"], log)

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
            # Wizard owns `cap` directly for its own capture loop; pause
            # the worker so they don't fight for frames on the same
            # VideoCapture object.
            worker.pause()
            try:
                new_cal = run_wizard(cap, actual_w, actual_h,
                                     CALIBRATION_FILE, GPIO_OK, _gpio_low)
            finally:
                worker.resume()
            if new_cal:
                state["cal"] = new_cal
                state["baseline_mm"] = new_cal["baseline_mm"]

        elif action == "view":
            # Viewer is a modal takeover; no point running stereo while
            # nobody can see the viewfinder.
            worker.pause()
            try:
                paths = _ply_paths()
                viewer.show(paths,
                            max_points=settings.get("viewer_subsample", 60000))
            finally:
                worker.resume()

        elif action == "power":
            # Pause stereo while a modal sequence runs on top of the UI.
            worker.pause()
            try:
                choice = _power_menu()
                if choice == "shutdown":
                    if _confirm_shutdown():
                        shutdown(); return
                elif choice == "update":
                    if _do_update_check(log):
                        # New code is on disk; exit non-zero so systemd
                        # (Restart=on-failure) relaunches us into the new
                        # build. update.sh on the next boot will see we're
                        # already up-to-date and skip the pull.
                        log.info("exiting for restart into updated build")
                        worker.stop(); cam.stop(); cap.release(); display.quit()
                        if GPIO_OK:
                            GPIO.cleanup()
                        import sys
                        sys.exit(42)
            finally:
                worker.resume()

        # Show the result dialog for an async save when the worker finishes.
        # We only pop this when saving is False so progress overlay stays smooth.
        if not state["saving"]:
            if "usb_msg" in state:
                ok, msg = state.pop("usb_msg")
                confirm_dialog(
                    [(("Export complete!" if ok else "Export failed"),
                      C["green"] if ok else C["red"]),
                     msg,
                     ("Safe to unplug USB drive." if ok else "")],
                    [("OK", C["blue"], "ok")])
            elif "zip_msg" in state:
                ok, msg = state.pop("zip_msg")
                confirm_dialog(
                    [(("Zip complete!" if ok else "Zip failed"),
                      C["green"] if ok else C["red"]),
                     (os.path.basename(msg) if ok else msg)],
                    [("OK", C["blue"], "ok")])

        # Capture trigger - enqueue on the worker so full-res SGBM runs
        # off the main thread. The worker snapshots L/R at the moment it
        # picks up the request, so the capture reflects the live scene,
        # not whatever frame main happened to be holding.
        if ret and state["mode"] == "semi" and gpio_trigger:
            worker.request_capture()
        elif ret and state["mode"] == "auto" and gpio_now and now - last_auto >= AUTO_INTERVAL:
            last_auto = now
            worker.request_capture()

        cap_result = worker.poll_capture()
        if cap_result is not None:
            save_left, save_right, save_disp = cap_result

            disp_pos = np.clip(save_disp, 0, None)
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
        display.show(canvas)

        # Tiny sleep keeps CPU from pinning when frames are stale.
        time.sleep(0.005)

    worker.stop()
    cam.stop()
    cap.release()
    display.quit()
    if GPIO_OK:
        GPIO.cleanup()
    log.info("shutdown clean")
