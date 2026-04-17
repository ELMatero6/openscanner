"""Stereo matching: build SGBM matchers, compute disparity, sample crosshair."""

import cv2
import numpy as np

from .config import DISP_SCALE

try:
    _ = cv2.ximgproc.createDisparityWLSFilter
    XIMGPROC = True
except AttributeError:
    XIMGPROC = False


def build_matcher(num_disp=128, block=3):
    """High-quality SGBM matcher (used at capture time)."""
    left = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block,
        P1=8 * 3 * block * block,
        P2=32 * 3 * block * block,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=150,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    if not XIMGPROC:
        return left, None, None
    right = cv2.ximgproc.createRightMatcher(left)
    wls = cv2.ximgproc.createDisparityWLSFilter(left)
    wls.setLambda(16000)
    wls.setSigmaColor(2.0)
    return left, right, wls


def build_live_matcher():
    """Lightweight matcher for live preview / rangefinder."""
    b = 5
    return cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=64, blockSize=b,
        P1=8 * 3 * b * b, P2=32 * 3 * b * b, disp12MaxDiff=1,
        uniquenessRatio=10, speckleWindowSize=80, speckleRange=2,
        preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_HH4,
    )


def compute_disparity(left, right, matchers):
    """Compute float32 disparity map at DISP_SCALE * input resolution."""
    left_m, right_m, wls = matchers
    s = DISP_SCALE
    gl = cv2.resize(cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY), None, fx=s, fy=s)
    gr = cv2.resize(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), None, fx=s, fy=s)
    disp = left_m.compute(gl, gr)
    if wls is not None:
        disp_r = right_m.compute(gr, gl)
        small  = cv2.resize(left, (gl.shape[1], gl.shape[0]))
        disp   = wls.filter(disp, small, disparity_map_right=disp_r)
    return disp.astype(np.float32) / 16.0


def sample_disp_at(disp, cx_frac=0.5, cy_frac=0.5, radius_frac=0.05):
    """Median valid disparity in a patch around (cx_frac, cy_frac).

    Resolution-independent: caller can sample at any disparity scale and
    the patch grows with the image, so the sampling region is the same
    fraction of the scene either way.
    """
    h, w = disp.shape[:2]
    cx, cy = int(w * cx_frac), int(h * cy_frac)
    r = max(4, int(min(w, h) * radius_frac))
    patch = disp[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
    valid = patch[patch > 0]
    if len(valid) < 4:
        return None
    return float(np.median(valid))


def disp_to_colour(disp, num_disp):
    """TURBO-coloured disparity for display, transparent where invalid."""
    clipped = np.clip(disp, 0, num_disp)
    norm    = (clipped / num_disp * 255).astype(np.uint8)
    hm      = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    hm[disp <= 0] = 0
    return hm
