"""Stereo calibration: load existing, run wizard, compute reprojection error."""

import os
import time
import threading

import cv2
import numpy as np

from .config import (
    BTN_H, BTN_Y, C, CAL_X, CAL_Y, CAL_W, CAL_H,
    FONT, GPIO_PIN, PANEL_W, PREVIEW_H, SCREEN_H, SCREEN_W,
)


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


def run_wizard(cap, actual_w, actual_h, win, cal_path, gpio_ok, gpio_in):
    """Touchscreen calibration wizard. Returns loaded calibration dict or None."""
    CHESS    = (9, 6)
    SQ_M     = 0.025
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

    def on_touch(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if y > BTN_Y:
            bw = SCREEN_W // 3
            param["action"] = ["save", "calibrate", "cancel"][min(x // bw, 2)]
        else:
            param["action"] = "save"

    cv2.setMouseCallback(win, on_touch, touch)

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
        canvas[:PREVIEW_H, :PANEL_W] = fit_to_panel(left)
        canvas[:PREVIEW_H, PANEL_W:] = fit_to_panel(right)
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

        cv2.imshow(win, canvas)
        cv2.waitKey(1)

        action = touch["action"]; touch["action"] = None

        if action == "save":
            if both and cL is not None and cR is not None:
                gL_f = cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY)
                gR_f = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
                crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                cL2 = cv2.cornerSubPix(gL_f, cL, (11, 11), (-1, -1), crit)
                cR2 = cv2.cornerSubPix(gR_f, cR, (11, 11), (-1, -1), crit)
                objpts.append(objp); ipts_L.append(cL2); ipts_R.append(cR2)
                cv2.imwrite(f"{CAL_DIR}/left/{count}.png",  left)
                cv2.imwrite(f"{CAL_DIR}/right/{count}.png", right)
                count += 1
                msg = (f"Saved pair {count}!  ({MIN_PAIR - count} more needed)"
                       if count < MIN_PAIR else f"Saved pair {count}!  Ready to calibrate")
                msg_col = C["green"]
            else:
                msg, msg_col = "Board not detected in both eyes - hold still", C["red"]

        elif action == "calibrate":
            if count < MIN_PAIR:
                msg, msg_col = f"Need {MIN_PAIR} pairs, have {count}", C["red"]
                continue

            result = {"done": False, "cal": None, "err": None, "rms": None}

            def _compute():
                try:
                    sz = (mid, actual_h)
                    _, mtxL, distL, _, _ = cv2.calibrateCamera(objpts, ipts_L, sz, None, None)
                    _, mtxR, distR, _, _ = cv2.calibrateCamera(objpts, ipts_R, sz, None, None)
                    rms, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
                        objpts, ipts_L, ipts_R,
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
                    result["rms"] = rms
                    result["cal"] = load_calibration(cal_path, sz)
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
                cv2.imshow(win, ov)
                cv2.waitKey(400)

            if result["err"]:
                msg, msg_col = f"Failed: {result['err'][:45]}", C["red"]
            elif result["cal"]:
                rms = result["rms"]
                rms_col = C["green"] if rms < 0.5 else C["yellow"] if rms < 1.5 else C["red"]
                rms_lbl = "EXCELLENT" if rms < 0.5 else "OK" if rms < 1.5 else "POOR"
                _show_rms(win, rms, rms_lbl, rms_col)
                return result["cal"]

        elif action == "cancel":
            return None


def _show_rms(win, rms, label, colour):
    """Brief reprojection error report screen."""
    end = time.time() + 3.0
    while time.time() < end:
        ov = np.zeros((SCREEN_H, SCREEN_W, 3), np.uint8)
        cv2.rectangle(ov, (60, 130), (740, 330), (25, 25, 25), -1)
        cv2.rectangle(ov, (60, 130), (740, 330), C["white"], 1)
        cv2.putText(ov, "Calibration complete", (160, 175), FONT, 0.8, C["white"], 2)
        cv2.putText(ov, f"Reprojection error: {rms:.3f} px",
                    (160, 225), FONT, 0.7, C["white"], 1)
        cv2.putText(ov, label, (160, 280), FONT, 1.2, colour, 3)
        cv2.imshow(win, ov)
        cv2.waitKey(50)
