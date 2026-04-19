"""Stereo calibration: load existing, run wizard, compute reprojection error."""

import os
import time
import threading

import cv2
import numpy as np

from . import display
from . import logger as log_mod
from .config import (
    BTN_H, BTN_Y, C, FONT, GPIO_PIN, PANEL_W, PREVIEW_H, SCREEN_H, SCREEN_W,
)

log = log_mod.get("scanner.calibration")


def load_calibration(path, fallback_size):
    """Load .npz calibration. Returns dict with rect maps + Q + baseline_mm,
    focal_px, or None if missing/broken.
    """
    if not path or not os.path.exists(path):
        print("[CAL] none - running uncalibrated")
        return None
    try:
        d  = np.load(path)
        sz = tuple(d["img_size"].astype(int)) if "img_size" in d else fallback_size
        m1L, m2L = cv2.initUndistortRectifyMap(
            d["mtxL"], d["distL"], d["R1"], d["P1"], sz, cv2.CV_16SC2)
        m1R, m2R = cv2.initUndistortRectifyMap(
            d["mtxR"], d["distR"], d["R2"], d["P2"], sz, cv2.CV_16SC2)
        T = d["T"].flatten()
        baseline_mm = float(np.linalg.norm(T)) * 1000.0
        focal_px = float(d["P1"][0, 0])
        print(f"[CAL] loaded {sz[0]}x{sz[1]}  baseline={baseline_mm:.1f}mm  fx={focal_px:.0f}px")
        return {
            "m1L": m1L, "m2L": m2L, "m1R": m1R, "m2R": m2R,
            "Q": d["Q"], "baseline_mm": baseline_mm, "focal_px": focal_px,
            "img_size": sz, "path": path,
        }
    except Exception as e:
        print(f"[CAL] load failed: {e}")
        return None


def rectify(left, right, cal):
    return (
        cv2.remap(left,  cal["m1L"], cal["m2L"], cv2.INTER_LINEAR),
        cv2.remap(right, cal["m1R"], cal["m2R"], cv2.INTER_LINEAR),
    )


def _find_corners_hq(gray, chess):
    """High-quality corner detection on full-res grayscale.

    Uses findChessboardCornersSB (sector-based, subpixel-accurate) when
    available in the installed OpenCV build, otherwise falls back to the
    classic findChessboardCorners + cornerSubPix with a wide window.
    Returns (ok, corners_float32) where corners has shape (N, 1, 2).
    """
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = (cv2.CALIB_CB_NORMALIZE_IMAGE |
                 cv2.CALIB_CB_EXHAUSTIVE |
                 cv2.CALIB_CB_ACCURACY)
        ok, corners = cv2.findChessboardCornersSB(gray, chess, flags)
        if ok:
            return True, corners.astype(np.float32)
        return False, None

    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
             cv2.CALIB_CB_NORMALIZE_IMAGE |
             cv2.CALIB_CB_FILTER_QUADS)
    ok, corners = cv2.findChessboardCorners(gray, chess, flags)
    if not ok:
        return False, None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-4)
    refined = cv2.cornerSubPix(gray, corners, (21, 21), (-1, -1), crit)
    return True, refined.astype(np.float32)


def _per_pair_rms(objpts, ipts_L, ipts_R, mtxL, distL, mtxR, distR):
    """Per-pair reprojection RMS in pixels, averaged over L + R corners."""
    out = []
    for i, op in enumerate(objpts):
        _, rvL, tvL = cv2.solvePnP(op, ipts_L[i], mtxL, distL)
        _, rvR, tvR = cv2.solvePnP(op, ipts_R[i], mtxR, distR)
        pL, _ = cv2.projectPoints(op, rvL, tvL, mtxL, distL)
        pR, _ = cv2.projectPoints(op, rvR, tvR, mtxR, distR)
        eL = np.linalg.norm(pL.reshape(-1, 2) - ipts_L[i].reshape(-1, 2), axis=1)
        eR = np.linalg.norm(pR.reshape(-1, 2) - ipts_R[i].reshape(-1, 2), axis=1)
        out.append(float(np.sqrt(((eL ** 2).sum() + (eR ** 2).sum()) /
                                  (len(eL) + len(eR)))))
    return out


def reprojection_error(objpts, ipts_L, ipts_R, mtxL, distL, mtxR, distR, R, T, sz):
    """Average reprojection error in pixels. < 0.5 is excellent, > 1.5 is poor."""
    rvecs_L, tvecs_L = [], []
    for i, op in enumerate(objpts):
        ok, rv, tv = cv2.solvePnP(op, ipts_L[i], mtxL, distL)
        rvecs_L.append(rv); tvecs_L.append(tv)
    err_total, n_total = 0.0, 0
    for i, op in enumerate(objpts):
        proj_L, _ = cv2.projectPoints(op, rvecs_L[i], tvecs_L[i], mtxL, distL)
        err_total += float(np.sum(np.linalg.norm(
            proj_L.reshape(-1, 2) - ipts_L[i].reshape(-1, 2), axis=1)))
        n_total += len(op)
    return err_total / max(n_total, 1)


def save_calibration(path, mtxL, distL, mtxR, distR, R, T, R1, R2, P1, P2, Q,
                     sz, roi1, roi2, rms):
    np.savez(
        path,
        mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR,
        R=R, T=T, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        img_size=np.array(sz),
        roi1=np.array(roi1), roi2=np.array(roi2),
        rms=np.array([rms]),
    )


def run_wizard(cap, actual_w, actual_h, cal_path, gpio_ok, gpio_in):
    """Touchscreen calibration wizard. Returns loaded calibration dict or None."""
    CHESS    = (9, 6)
    SQ_M     = 0.02421
    CAL_DIR  = "cal_pairs"
    MIN_PAIR = 15
    os.makedirs(f"{CAL_DIR}/left",  exist_ok=True)
    os.makedirs(f"{CAL_DIR}/right", exist_ok=True)

    objp = np.zeros((CHESS[0] * CHESS[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESS[0], 0:CHESS[1]].T.reshape(-1, 2)
    objp *= SQ_M

    objpts, ipts_L, ipts_R = [], [], []
    count    = 0
    msg      = "Hold board still - trigger or tap to save"
    msg_col  = C["white"]
    both     = False
    cL = cR  = None
    tick     = 0
    touch    = {"action": None}
    gpio_was = False

    def _drain_input():
        for ev in display.events():
            if isinstance(ev, display.Tap):
                if ev.y > BTN_Y:
                    bw = SCREEN_W // 3
                    touch["action"] = ["save", "calibrate", "cancel"][min(ev.x // bw, 2)]
                else:
                    touch["action"] = "save"
            elif isinstance(ev, display.Key) and ev.char in ("q", "\x1b"):
                touch["action"] = "cancel"

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02); continue

        mid   = frame.shape[1] // 2
        left  = frame[:, :mid]
        right = frame[:, mid:]

        gpio_now = gpio_in()
        if gpio_now and not gpio_was:
            touch["action"] = "save"
        gpio_was = gpio_now

        tick += 1
        if tick % 6 == 0:
            gL = cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY)
            gR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            s = 0.25
            gLs = cv2.resize(gL, None, fx=s, fy=s)
            gRs = cv2.resize(gR, None, fx=s, fy=s)
            try:
                retL, cL = cv2.findChessboardCorners(gLs, CHESS, cv2.CALIB_CB_FAST_CHECK)
                retR, cR = cv2.findChessboardCorners(gRs, CHESS, cv2.CALIB_CB_FAST_CHECK)
                both = retL and retR
                if retL: cL = cL / s
                if retR: cR = cR / s
            except Exception:
                both = False

        canvas = np.zeros((SCREEN_H, SCREEN_W, 3), np.uint8)
        from .ui import fit_to_panel
        canvas[:PREVIEW_H, :PANEL_W] = fit_to_panel(left, target_h=PREVIEW_H)
        canvas[:PREVIEW_H, PANEL_W:] = fit_to_panel(right, target_h=PREVIEW_H)
        cv2.line(canvas, (PANEL_W, 0), (PANEL_W, PREVIEW_H), C["grey"], 1)

        det_col = C["green"] if both else C["red"]
        det_lbl = "BOTH FOUND - pull trigger or tap" if both else "searching for board..."
        cv2.rectangle(canvas, (0, PREVIEW_H - 28), (SCREEN_W, PREVIEW_H), (0, 0, 0), -1)
        cv2.putText(canvas, det_lbl, (8, PREVIEW_H - 10), FONT, 0.5, det_col, 1)
        cv2.putText(canvas, msg,     (8, PREVIEW_H - 26), FONT, 0.4, msg_col, 1)
        cv2.putText(canvas, f"Pairs: {count}/{MIN_PAIR}",
                    (SCREEN_W - 170, PREVIEW_H - 10), FONT, 0.5, C["white"], 1)

        bw3 = SCREEN_W // 3
        for i, (lbl, col) in enumerate([
            ("SAVE PAIR", C["blue"]),
            ("CALIBRATE", C["green"] if count >= MIN_PAIR else C["dim"]),
            ("CANCEL",    C["red"]),
        ]):
            bx = i * bw3
            cv2.rectangle(canvas, (bx + 2, BTN_Y + 2), (bx + bw3 - 2, SCREEN_H - 2), col, -1)
            cv2.rectangle(canvas, (bx + 2, BTN_Y + 2), (bx + bw3 - 2, SCREEN_H - 2), C["white"], 1)
            (tw, th), _ = cv2.getTextSize(lbl, FONT, 0.7, 2)
            cv2.putText(canvas, lbl,
                        (bx + (bw3 - tw) // 2, BTN_Y + (BTN_H + th) // 2),
                        FONT, 0.7, C["white"], 2)

        display.show(canvas)
        _drain_input()
        time.sleep(0.005)

        action = touch["action"]; touch["action"] = None

        if action == "save":
            # Re-detect on FULL-res grayscale with a proper detector. The
            # live preview uses a fast low-res detector to keep the UI
            # responsive, but its corners are only pixel-accurate at full
            # res (~+/-2 px initial error) - cornerSubPix cannot always
            # pull that in, and the leftover noise is exactly what
            # produces 3-pixel RMS calibrations.
            gL_f = cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY)
            gR_f = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            okL, cL_hq = _find_corners_hq(gL_f, CHESS)
            okR, cR_hq = _find_corners_hq(gR_f, CHESS)
            if okL and okR:
                objpts.append(objp); ipts_L.append(cL_hq); ipts_R.append(cR_hq)
                cv2.imwrite(f"{CAL_DIR}/left/{count}.png",  left)
                cv2.imwrite(f"{CAL_DIR}/right/{count}.png", right)
                count += 1
                msg = (f"Saved pair {count}!  ({MIN_PAIR - count} more needed)"
                       if count < MIN_PAIR else f"Saved pair {count}!  Ready to calibrate")
                msg_col = C["green"]
            else:
                msg, msg_col = "Corners failed full-res detection - hold steadier", C["red"]

        elif action == "calibrate":
            if count < MIN_PAIR:
                msg, msg_col = f"Need {MIN_PAIR} pairs, have {count}", C["red"]
                continue

            result = {"done": False, "cal": None, "err": None, "rms": None}

            def _compute():
                try:
                    sz = (mid, actual_h)
                    op, pL, pR = list(objpts), list(ipts_L), list(ipts_R)

                    # Two-pass calibration: fit, find pairs whose per-pair
                    # reprojection is > 2*mean (or worst 20%), drop them,
                    # refit. One bad board frame can otherwise dominate
                    # the mean RMS and ruin downstream depth.
                    dropped = 0
                    for iteration in range(2):
                        _, mtxL, distL, _, _ = cv2.calibrateCamera(op, pL, sz, None, None)
                        _, mtxR, distR, _, _ = cv2.calibrateCamera(op, pR, sz, None, None)

                        if iteration == 0 and len(op) > 10:
                            per = _per_pair_rms(op, pL, pR, mtxL, distL, mtxR, distR)
                            log.info("per-pair RMS: %s",
                                     " ".join(f"{e:.2f}" for e in per))
                            thresh = max(2.0 * float(np.mean(per)),
                                         float(np.percentile(per, 80)))
                            keep = [i for i, e in enumerate(per) if e <= thresh]
                            # Never drop below 10 pairs - stereo needs variety.
                            if len(keep) >= 10 and len(keep) < len(op):
                                dropped = len(op) - len(keep)
                                op = [op[i] for i in keep]
                                pL = [pL[i] for i in keep]
                                pR = [pR[i] for i in keep]
                                log.info("dropped %d high-RMS pairs (thresh=%.2f px)",
                                         dropped, thresh)
                                continue
                        break

                    rms, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
                        op, pL, pR,
                        mtxL, distL, mtxR, distR, sz,
                        criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5),
                        flags=cv2.CALIB_FIX_INTRINSIC,
                    )
                    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                        mtxL, distL, mtxR, distR, sz, R, T,
                        flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1,
                    )
                    save_calibration(cal_path,
                                     mtxL, distL, mtxR, distR, R, T,
                                     R1, R2, P1, P2, Q, sz, roi1, roi2, rms)
                    log.info("stereoCalibrate RMS=%.3f px on %d pairs (dropped %d)",
                             rms, len(op), dropped)
                    result["rms"]     = rms
                    result["dropped"] = dropped
                    result["kept"]    = len(op)
                    result["cal"]     = load_calibration(cal_path, sz)
                except Exception as e:
                    result["err"] = str(e)
                finally:
                    result["done"] = True

            threading.Thread(target=_compute, daemon=True).start()

            dots = 0
            while not result["done"]:
                dots = (dots + 1) % 4
                ov = np.zeros((SCREEN_H, SCREEN_W, 3), np.uint8)
                cv2.rectangle(ov, (100, 170), (700, 290), (25, 25, 25), -1)
                cv2.rectangle(ov, (100, 170), (700, 290), C["white"], 1)
                cv2.putText(ov, "Computing" + "." * (dots + 1),
                            (130, 225), FONT, 0.9, C["white"], 2)
                cv2.putText(ov, f"{count} pairs - ~30-60s on Pi",
                            (130, 265), FONT, 0.5, C["yellow"], 1)
                display.show(ov)
                # drain events so the dialog doesn't queue up taps
                display.events()
                time.sleep(0.4)

            if result["err"]:
                msg, msg_col = f"Failed: {result['err'][:45]}", C["red"]
            elif result["cal"]:
                rms = result["rms"]
                rms_col = C["green"] if rms < 0.5 else C["yellow"] if rms < 1.5 else C["red"]
                rms_lbl = "EXCELLENT" if rms < 0.5 else "OK" if rms < 1.5 else "POOR"
                _show_rms(rms, rms_lbl, rms_col,
                          kept=result.get("kept", 0),
                          dropped=result.get("dropped", 0))
                return result["cal"]

        elif action == "cancel":
            return None


def _show_rms(rms, label, colour, kept=0, dropped=0):
    """Brief reprojection error report screen."""
    pairs_line = (f"Used {kept} pairs"
                  + (f" (dropped {dropped} outliers)" if dropped else ""))
    end = time.time() + 3.5
    while time.time() < end:
        ov = np.zeros((SCREEN_H, SCREEN_W, 3), np.uint8)
        cv2.rectangle(ov, (60, 120), (740, 360), (25, 25, 25), -1)
        cv2.rectangle(ov, (60, 120), (740, 360), C["white"], 1)
        cv2.putText(ov, "Calibration complete", (160, 165), FONT, 0.8, C["white"], 2)
        cv2.putText(ov, f"Reprojection error: {rms:.3f} px",
                    (160, 215), FONT, 0.7, C["white"], 1)
        cv2.putText(ov, label, (160, 275), FONT, 1.2, colour, 3)
        cv2.putText(ov, pairs_line, (160, 325), FONT, 0.5, C["grey"], 1)
        display.show(ov)
        display.events()
        time.sleep(0.05)
