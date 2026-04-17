"""Capture I/O: CSV manifest, PNG images, PLY point clouds, zip + USB export."""

import csv
import os
import shutil
import subprocess
import zipfile
from datetime import datetime

import cv2
import numpy as np

from .config import CALIBRATION_FILE, DISP_SCALE, SAVE_DIR
from .stereo import XIMGPROC


CSV_HEADERS = [
    "capture", "timestamp",
    "file_L", "file_R", "file_D", "file_D_raw", "file_PLY",
    "img_w", "img_h",
    "rectified", "dist_mode", "num_disp", "block_size",
    "wls", "bg_removal",
    "blur_score", "mean_disp", "coverage_pct",
    "valid_pixels", "total_pixels",
    "baseline_mm", "focal_px", "disp_scale",
]


def init_csv(save_dir):
    """Create captures.csv with headers if absent. Idempotent."""
    path = os.path.join(save_dir, "captures.csv")
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADERS)
    return path


def append_csv(csv_path, row):
    """Append a row dict matching CSV_HEADERS. Missing keys become empty."""
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow([row.get(k, "") for k in CSV_HEADERS])


def disparity_to_points(disp, Q, colour_img, max_distance_m=5.0):
    """Reproject disparity to 3D points. Returns (Nx3 xyz_m, Nx3 rgb_uint8)."""
    pts3d = cv2.reprojectImageTo3D(disp, Q)
    mask = (disp > 0) & np.isfinite(pts3d[:, :, 2]) & (pts3d[:, :, 2] > 0)
    mask &= pts3d[:, :, 2] < max_distance_m
    xyz = pts3d[mask]
    if colour_img.shape[:2] != disp.shape[:2]:
        colour_img = cv2.resize(colour_img, (disp.shape[1], disp.shape[0]))
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
    from .config import DIST_PRESETS
    num_disp = DIST_PRESETS[state.get("dist_mode", "MED")]["num_disp"]
    clipped  = np.clip(disp, 0, num_disp)
    norm     = (clipped / num_disp * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    coloured[disp <= 0] = 0
    cv2.imwrite(os.path.join(save_dir, fnames["D"]), coloured)

    # Raw 16-bit disparity * 16 (sub-pixel SGBM precision)
    raw = (np.clip(disp, 0, None) * 16).astype(np.uint16)
    cv2.imwrite(os.path.join(save_dir, fnames["D_raw"]), raw)

    # PLY point cloud (only if calibrated - need Q matrix)
    ply_written = False
    if cal is not None:
        # disp is at DISP_SCALE - need to scale Q's translation column
        # Q = [[1,0,0,-cx],[0,1,0,-cy],[0,0,0,f],[0,0,-1/Tx,(cx-cx')/Tx]]
        # Scaling intrinsics: cx,cy,f all multiply by s. New Q:
        s = DISP_SCALE
        Q_scaled = cal["Q"].copy()
        Q_scaled[0, 3] *= s   # -cx
        Q_scaled[1, 3] *= s   # -cy
        Q_scaled[2, 3] *= s   # f
        # Disparity also at scale s; Q[3,2] = -1/Tx is in original-pixel units.
        # When disp is in scaled-pixel units, multiply by s to compensate:
        Q_scaled[3, 2] *= s
        # Use rectified left at matching scale for colour
        left_small = cv2.resize(left, (disp.shape[1], disp.shape[0]))
        xyz, rgb = disparity_to_points(disp, Q_scaled, left_small)
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

    p        = DIST_PRESETS[state.get("dist_mode", "MED")]
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
        "dist_mode":     state.get("dist_mode", "MED"),
        "num_disp":      p["num_disp"],
        "block_size":    p["block"],
        "wls":           1 if XIMGPROC else 0,
        "bg_removal":    1 if state.get("bg_thresh") is not None else 0,
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
            print(f"[FUSE] skip {fname}: {e}")
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
        "fused.ply     concatenation of all PLYs (no pose registration)",
        "",
        "To recover float disparity:  disp = imread(N_D_raw.png, IMREAD_UNCHANGED).astype(float32) / 16",
        "To recover 3D from disparity: xyz = reprojectImageTo3D(disp, Q)  (Q in npz)",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def zip_export(save_dir, out_path, cal, version, progress_cb=None):
    """Zip scans dir + calibration.npz + fused.ply + README into one archive."""
    fused = os.path.join(save_dir, "fused.ply")
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


def find_usb_drive():
    """Return (device, mountpoint) of first removable mount, or (None, None)."""
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                dev, mnt = parts[0], parts[1]
                if not dev.startswith("/dev/sd"):
                    continue
                if any(mnt.startswith(p) for p in ("/media", "/mnt", "/run/media")):
                    return dev, mnt
    except Exception:
        pass
    return None, None


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
    """Copy scans + calibration + zip + README to USB. Returns (ok, message).

    Does NOT format. Caller must verify usb_writable() first.
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
            subprocess.run(["sync"], timeout=5)
        except Exception:
            pass

        if progress_cb:
            progress_cb(100)
        return True, f"Exported to {os.path.basename(dst)}"
    except OSError as e:
        return False, f"Copy failed: {e}"
