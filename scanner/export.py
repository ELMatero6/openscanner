"""Capture I/O: CSV manifest, PNG images, PLY point clouds, zip + USB export."""

import csv
import logging
import os
import re
import shutil
import subprocess
import time
import zipfile
from datetime import datetime

import cv2
import numpy as np

from .config import (
    BLOCK_SIZE, CALIBRATION_FILE, DISP_SCALE, MAX_BRIGHTNESS,
    MAX_DEPTH_M, MIN_DEPTH_M, NUM_DISP, SAVE_DIR,
)
from .stereo import XIMGPROC

log = logging.getLogger("scanner.export")


CSV_HEADERS = [
    "capture", "timestamp",
    "file_L", "file_R", "file_D", "file_D_raw", "file_PLY",
    "img_w", "img_h",
    "rectified", "num_disp", "block_size", "wls",
    "blur_score", "mean_disp", "coverage_pct",
    "valid_pixels", "total_pixels",
    "baseline_mm", "focal_px", "disp_scale",
]


def init_csv(save_dir):
    """Create save_dir and captures.csv with headers if absent. Idempotent."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "captures.csv")
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADERS)
    return path


def append_csv(csv_path, row):
    """Append a row dict matching CSV_HEADERS. Missing keys become empty."""
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow([row.get(k, "") for k in CSV_HEADERS])


def disparity_to_points(disp, Q, colour_img,
                        min_depth_m=MIN_DEPTH_M,
                        max_depth_m=MAX_DEPTH_M,
                        max_brightness=MAX_BRIGHTNESS):
    """Reproject disparity to 3D points. Returns (Nx3 xyz_m, Nx3 rgb_uint8).

    Applies quality gates to keep garbage depth out of the cloud:
      - depth clamp (min/max Z): kills window/sun streaks to infinity
        and near-lens speckle noise
      - overexposure mask: drops pixels where any channel is blown out,
        because stereo matching has no signal there and produces
        essentially random disparity
    """
    pts3d = cv2.reprojectImageTo3D(disp, Q)
    if colour_img.shape[:2] != disp.shape[:2]:
        colour_img = cv2.resize(colour_img, (disp.shape[1], disp.shape[0]))
    z = pts3d[:, :, 2]
    mask = (disp > 0) & np.isfinite(z) & (z > min_depth_m) & (z < max_depth_m)
    if max_brightness < 255:
        # BGR max per pixel - one blown channel is enough to reject
        bright = colour_img.max(axis=2)
        mask &= bright < max_brightness
    xyz = pts3d[mask]
    rgb = cv2.cvtColor(colour_img, cv2.COLOR_BGR2RGB)[mask]
    return xyz.astype(np.float32), rgb.astype(np.uint8)


def write_ply(path, xyz, rgb):
    """Binary little-endian PLY with vertex colour. Pi-friendly: no temp arrays."""
    n = len(xyz)
    if n == 0:
        return False
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    vert_dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"),  ("g", "u1"),  ("b", "u1"),
    ])
    verts = np.empty(n, dtype=vert_dtype)
    verts["x"] = xyz[:, 0]; verts["y"] = xyz[:, 1]; verts["z"] = xyz[:, 2]
    verts["r"] = rgb[:, 0]; verts["g"] = rgb[:, 1]; verts["b"] = rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        verts.tofile(f)
    return True


def save_capture(save_dir, idx, left, right, disp, cal, state, csv_path):
    """Persist a capture. Returns dict of filenames written."""
    n = idx
    fnames = {
        "L":     f"{n}_L.png",
        "R":     f"{n}_R.png",
        "D":     f"{n}_D.png",
        "D_raw": f"{n}_D_raw.png",
        "PLY":   f"{n}.ply",
    }

    cv2.imwrite(os.path.join(save_dir, fnames["L"]), left)
    cv2.imwrite(os.path.join(save_dir, fnames["R"]), right)

    # Coloured heatmap (display)
    clipped  = np.clip(disp, 0, NUM_DISP)
    norm     = (clipped / NUM_DISP * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    coloured[disp <= 0] = 0
    cv2.imwrite(os.path.join(save_dir, fnames["D"]), coloured)

    # Raw 16-bit disparity * 16 (sub-pixel SGBM precision)
    raw = (np.clip(disp, 0, None) * 16).astype(np.uint16)
    cv2.imwrite(os.path.join(save_dir, fnames["D_raw"]), raw)

    # PLY point cloud (only if calibrated - need Q matrix)
    ply_written = False
    if cal is not None:
        # SGBM runs at DISP_SCALE to keep the Pi fast; disp is in half-res
        # pixels on a half-res grid. The calibration's Q matrix is in the
        # original (full-res) geometry, so the cleanest fix is to convert
        # the disparity map back to full-res space: upscale the image, and
        # scale values by 1/s so they're in full-res pixels too. Then Q
        # needs no rewriting - reprojectImageTo3D just works.
        fh, fw = left.shape[:2]
        disp_full = cv2.resize(disp, (fw, fh),
                               interpolation=cv2.INTER_NEAREST) / DISP_SCALE
        xyz, rgb = disparity_to_points(disp_full, cal["Q"], left)
        ply_path = os.path.join(save_dir, fnames["PLY"])
        ply_written = write_ply(ply_path, xyz, rgb)

    # Metadata
    gray       = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    blur       = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)
    valid_mask = disp > 0
    total_px   = int(disp.size)
    valid_px   = int(valid_mask.sum())
    coverage   = round(valid_px / total_px * 100, 2)
    mean_disp  = round(float(disp[valid_mask].mean()) if valid_px > 0 else 0.0, 3)

    baseline = cal["baseline_mm"] if cal else state.get("baseline_mm", 60.0)
    focal    = cal["focal_px"]    if cal else 0.0

    append_csv(csv_path, {
        "capture":       n,
        "timestamp":     datetime.now().isoformat(),
        "file_L":        fnames["L"],
        "file_R":        fnames["R"],
        "file_D":        fnames["D"],
        "file_D_raw":    fnames["D_raw"],
        "file_PLY":      fnames["PLY"] if ply_written else "",
        "img_w":         left.shape[1],
        "img_h":         left.shape[0],
        "rectified":     1 if cal else 0,
        "num_disp":      NUM_DISP,
        "block_size":    BLOCK_SIZE,
        "wls":           1 if XIMGPROC else 0,
        "blur_score":    blur,
        "mean_disp":     mean_disp,
        "coverage_pct":  coverage,
        "valid_pixels":  valid_px,
        "total_pixels":  total_px,
        "baseline_mm":   round(baseline, 2),
        "focal_px":      round(focal, 2),
        "disp_scale":    DISP_SCALE,
    })

    return fnames


def fuse_plys(save_dir, output_path, max_total_points=2_000_000):
    """Concatenate all per-capture PLYs into one. Subsamples if total exceeds cap.

    No ICP / pose registration - all clouds in their own camera frame.
    Useful as a quick visual check and as input to offline registration tools.
    """
    plys = sorted(
        f for f in os.listdir(save_dir)
        if f.endswith(".ply") and not f.startswith("fused")
    )
    if not plys:
        return False
    all_xyz, all_rgb = [], []
    for fname in plys:
        try:
            xyz, rgb = _read_ply(os.path.join(save_dir, fname))
            all_xyz.append(xyz); all_rgb.append(rgb)
        except Exception as e:
            log.warning("fuse: skip %s (%s)", fname, e)
    if not all_xyz:
        return False
    xyz = np.concatenate(all_xyz)
    rgb = np.concatenate(all_rgb)
    if len(xyz) > max_total_points:
        idx = np.random.choice(len(xyz), max_total_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    return write_ply(output_path, xyz, rgb)


def _read_ply(path):
    """Read a binary little-endian PLY written by write_ply."""
    with open(path, "rb") as f:
        header_end = b""
        while not header_end.endswith(b"end_header\n"):
            header_end += f.readline()
        n = 0
        for line in header_end.decode("ascii").splitlines():
            if line.startswith("element vertex"):
                n = int(line.split()[2]); break
        dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("r", "u1"),  ("g", "u1"),  ("b", "u1"),
        ])
        data = np.fromfile(f, dtype=dtype, count=n)
    xyz = np.column_stack([data["x"], data["y"], data["z"]])
    rgb = np.column_stack([data["r"], data["g"], data["b"]])
    return xyz, rgb


def write_readme(path, n_captures, cal, version):
    lines = [
        "openscanner export",
        "==================",
        f"version:    {version}",
        f"exported:   {datetime.now().isoformat()}",
        f"captures:   {n_captures}",
        f"calibrated: {'yes' if cal else 'no'}",
    ]
    if cal:
        lines += [
            f"baseline:   {cal['baseline_mm']:.1f} mm",
            f"focal:      {cal['focal_px']:.1f} px",
            f"img size:   {cal['img_size'][0]} x {cal['img_size'][1]}",
        ]
    lines += [
        "",
        "Files per capture N:",
        "  N_L.png      left rectified image",
        "  N_R.png      right rectified image",
        "  N_D.png      colour disparity preview (TURBO)",
        "  N_D_raw.png  16-bit raw disparity * 16 (recover float by /16)",
        "  N.ply        binary PLY point cloud (camera frame, metres)",
        "",
        "captures.csv  per-capture metadata + processing parameters",
        "stereo_calibration.npz  intrinsics + extrinsics + Q matrix",
        "concat.ply    concatenation of all PLYs (no pose registration)",
        "",
        "To recover float disparity:  disp = imread(N_D_raw.png, IMREAD_UNCHANGED).astype(float32) / 16",
        "To recover 3D from disparity: xyz = reprojectImageTo3D(disp, Q)  (Q in npz)",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def zip_export(save_dir, out_path, cal, version, progress_cb=None):
    """Zip scans dir + calibration.npz + concat.ply + README into one archive."""
    fused = os.path.join(save_dir, "concat.ply")
    fuse_plys(save_dir, fused)

    readme = os.path.join(save_dir, "README.txt")
    captures = sum(1 for f in os.listdir(save_dir) if f.endswith("_L.png"))
    write_readme(readme, captures, cal, version)

    files = sorted(
        f for f in os.listdir(save_dir)
        if f.endswith((".png", ".ply", ".csv", ".txt"))
    )
    if cal and os.path.exists(CALIBRATION_FILE):
        # include calibration alongside scans (write under same name in zip)
        pass

    total = len(files) + (1 if cal and os.path.exists(CALIBRATION_FILE) else 0)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, fname in enumerate(files):
            zf.write(os.path.join(save_dir, fname), fname)
            if progress_cb:
                progress_cb(int((i + 1) / total * 100))
        if cal and os.path.exists(CALIBRATION_FILE):
            zf.write(CALIBRATION_FILE, os.path.basename(CALIBRATION_FILE))
            if progress_cb:
                progress_cb(100)
    return out_path


_USB_MOUNT_DIR = "/tmp/openscanner_usb"


def _current_mount(dev):
    """Return mountpoint for a block device, or None."""
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if parts[0] == dev:
                    return parts[1]
    except Exception:
        pass
    return None


def usb_find_block_device():
    """Locate a removable USB block device.

    Returns (dev, parent, mountpoint_or_None). dev is the partition
    (e.g. /dev/sda1) if present, else the raw disk (/dev/sda). parent
    is always the raw disk - used for eject/power-off.
    """
    try:
        blocks = sorted(os.listdir("/sys/block"))
    except Exception:
        return None, None, None

    for blk in blocks:
        if not blk.startswith("sd"):
            continue
        try:
            with open(f"/sys/block/{blk}/removable") as f:
                if f.read().strip() != "1":
                    continue
        except Exception:
            continue
        try:
            parts = [p for p in os.listdir(f"/sys/block/{blk}")
                     if p.startswith(blk) and p != blk]
        except Exception:
            parts = []
        dev = f"/dev/{sorted(parts)[0]}" if parts else f"/dev/{blk}"
        parent = f"/dev/{blk}"
        return dev, parent, _current_mount(dev)
    return None, None, None


# Kept for backward compat with older call sites.
def find_usb_drive():
    dev, _, mnt = usb_find_block_device()
    return dev, mnt


def usb_mount(dev):
    """Mount dev via udisksctl (polkit - no sudo needed on the Pi).

    Returns the mountpoint, or None on failure.
    """
    existing = _current_mount(dev)
    if existing:
        return existing
    try:
        r = subprocess.run(
            ["udisksctl", "mount", "-b", dev, "--no-user-interaction"],
            capture_output=True, text=True, timeout=20,
        )
        if r.returncode == 0:
            for token in r.stdout.split():
                t = token.rstrip(".")
                if t.startswith(("/media", "/run/media")):
                    log.info("usb mount: %s -> %s", dev, t)
                    return t
        log.warning("udisksctl mount failed (rc=%d): %s",
                    r.returncode, r.stderr.strip() or r.stdout.strip())
    except Exception as e:
        log.warning("udisksctl mount error: %s", e)

    # Fallback: sudo mount to /tmp
    try:
        os.makedirs(_USB_MOUNT_DIR, exist_ok=True)
        uid = os.getuid()
        r = subprocess.run(
            ["sudo", "mount", "-o", f"uid={uid},gid={uid}", dev, _USB_MOUNT_DIR],
            capture_output=True, text=True, timeout=20,
        )
        if r.returncode == 0:
            log.info("usb mount (fallback): %s -> %s", dev, _USB_MOUNT_DIR)
            return _USB_MOUNT_DIR
        log.warning("mount fallback failed: %s", r.stderr.strip())
    except Exception as e:
        log.warning("mount fallback error: %s", e)
    return None


def usb_unmount_and_eject(dev, parent):
    """Flush, unmount, and power-off the parent disk. Safe to unplug after."""
    try:
        subprocess.run(["sync"], timeout=10)
    except Exception:
        pass
    # Try udisksctl first
    for cmd in (
        ["udisksctl", "unmount", "-b", dev, "--no-user-interaction"],
        ["sudo", "umount", dev],
    ):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if r.returncode == 0:
                log.info("usb unmount: %s via %s", dev, cmd[0])
                break
        except Exception as e:
            log.warning("unmount (%s): %s", cmd[0], e)
    # Power-off the parent so the kernel drops the device
    for cmd in (
        ["udisksctl", "power-off", "-b", parent, "--no-user-interaction"],
        ["sudo", "eject", parent],
    ):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if r.returncode == 0:
                log.info("usb eject: %s via %s", parent, cmd[0])
                return True
        except Exception as e:
            log.warning("eject (%s): %s", cmd[0], e)
    return False


def usb_format_vfat(dev, parent, label="SCANNER"):
    """Unmount and re-format dev as FAT32. Destructive."""
    if _current_mount(dev):
        try:
            subprocess.run(
                ["udisksctl", "unmount", "-b", dev, "--no-user-interaction"],
                capture_output=True, text=True, timeout=15,
            )
        except Exception:
            pass
        try:
            subprocess.run(["sudo", "umount", dev],
                           capture_output=True, text=True, timeout=15)
        except Exception:
            pass
    # FAT32 labels max 11 ASCII uppercase chars
    safe = re.sub(r"[^A-Z0-9_]", "", label.upper())[:11] or "SCANNER"
    log.info("usb format: mkfs.vfat -F 32 -n %s %s", safe, dev)
    try:
        r = subprocess.run(
            ["sudo", "mkfs.vfat", "-F", "32", "-n", safe, dev],
            capture_output=True, text=True, timeout=180,
        )
        if r.returncode != 0:
            log.error("mkfs.vfat failed (rc=%d): %s",
                      r.returncode, r.stderr.strip() or r.stdout.strip())
            return False
    except Exception as e:
        log.error("mkfs.vfat error: %s", e)
        return False
    # Give the kernel a moment to re-read the partition
    time.sleep(1.5)
    try:
        subprocess.run(["sudo", "partprobe", parent],
                       capture_output=True, text=True, timeout=10)
    except Exception:
        pass
    return True


def usb_writable(mnt):
    """Test write access without writing junk: try creating + deleting a tmp file."""
    test = os.path.join(mnt, ".openscanner_writable_test")
    try:
        with open(test, "w") as f:
            f.write("ok")
        os.remove(test)
        return True
    except Exception:
        return False


def usb_disk_free_mb(mnt):
    try:
        st = os.statvfs(mnt)
        return int(st.f_bavail * st.f_frsize / (1024 * 1024))
    except Exception:
        return None


def usb_export(save_dir, mnt, cal, version, progress_cb=None):
    """Copy scans + calibration + zip + README to a mounted drive.

    Returns (ok, message). Caller handles mount/unmount/eject.
    """
    if not usb_writable(mnt):
        return False, "USB drive is read-only or full"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(mnt, f"scanner_export_{ts}")
    try:
        os.makedirs(dst, exist_ok=False)
    except FileExistsError:
        return False, "Export folder already exists - try again"
    except PermissionError:
        return False, "No write permission on USB drive"

    try:
        scans_dst = os.path.join(dst, "scans")
        os.makedirs(scans_dst, exist_ok=True)
        scan_files = [f for f in os.listdir(save_dir)
                      if f.endswith((".png", ".ply", ".csv", ".txt"))]
        for i, fname in enumerate(scan_files):
            shutil.copy2(os.path.join(save_dir, fname), scans_dst)
            if progress_cb:
                progress_cb(int((i + 1) / max(len(scan_files), 1) * 80))

        zip_path = os.path.join(dst, f"scans_{ts}.zip")
        zip_export(save_dir, zip_path, cal, version,
                   progress_cb=lambda p: progress_cb(80 + p // 5) if progress_cb else None)

        if os.path.exists(CALIBRATION_FILE):
            shutil.copy2(CALIBRATION_FILE, dst)

        cal_dir = "cal_pairs"
        if os.path.isdir(cal_dir):
            shutil.copytree(cal_dir, os.path.join(dst, "cal_pairs"))

        # Best effort sync - don't fail if no permission
        try:
            subprocess.run(["sync"], timeout=10)
        except Exception:
            pass

        if progress_cb:
            progress_cb(100)
        return True, f"Exported to {os.path.basename(dst)}"
    except OSError as e:
        return False, f"Copy failed: {e}"
