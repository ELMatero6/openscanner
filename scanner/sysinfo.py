"""CPU temp, system stats, version stamp from git."""

import os
import subprocess
import time


def cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read()) / 1000.0
    except Exception:
        return None


def sys_stats(save_dir):
    stats = {"cpu": None, "ram": None, "disk": None}

    try:
        def read_cpu():
            with open("/proc/stat") as f:
                parts = f.readline().split()
            vals = [int(x) for x in parts[1:]]
            return vals[3], sum(vals)
        i1, t1 = read_cpu()
        time.sleep(0.05)
        i2, t2 = read_cpu()
        dt = t2 - t1
        stats["cpu"] = round((1 - (i2 - i1) / dt) * 100) if dt > 0 else 0
    except Exception:
        pass

    try:
        mem = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":")
                mem[k.strip()] = int(v.split()[0])
        total = mem["MemTotal"]
        avail = mem["MemAvailable"]
        stats["ram"] = round((total - avail) / total * 100)
    except Exception:
        pass

    try:
        st = os.statvfs(save_dir)
        total = st.f_blocks * st.f_frsize
        free  = st.f_bfree  * st.f_frsize
        stats["disk"] = round((total - free) / total * 100)
    except Exception:
        pass

    return stats


def git_sha(short=True):
    """Return current commit SHA, or 'dev' if not in a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            timeout=2,
        )
        return out.decode().strip()
    except Exception:
        return "dev"


def shutdown():
    """Trigger graceful system shutdown. Falls back to print on dev machines."""
    try:
        subprocess.Popen(["sudo", "shutdown", "-h", "now"])
    except Exception as e:
        print(f"[SHUTDOWN] failed: {e}")


# Install dir = repo root (one level above this file's package dir).
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_for_updates(timeout_s=15):
    """Fetch origin and report how many commits we're behind.

    Returns (ok, behind, branch, detail). On network failure or git error,
    ok=False and detail holds a short, user-visible reason.
    """
    try:
        fetch = subprocess.run(
            ["git", "-C", _REPO_DIR, "fetch", "origin"],
            capture_output=True, text=True, timeout=timeout_s,
        )
        if fetch.returncode != 0:
            return False, 0, "", (fetch.stderr.strip().splitlines() or ["fetch failed"])[-1][:80]

        branch = subprocess.run(
            ["git", "-C", _REPO_DIR, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or "main"

        behind = subprocess.run(
            ["git", "-C", _REPO_DIR, "rev-list", "--count",
             f"HEAD..origin/{branch}"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or "0"

        return True, int(behind), branch, ""
    except subprocess.TimeoutExpired:
        return False, 0, "", "timed out (offline?)"
    except Exception as e:
        return False, 0, "", str(e)[:80]


def apply_update(branch, timeout_s=30):
    """Fast-forward pull. Returns (ok, detail)."""
    try:
        r = subprocess.run(
            ["git", "-C", _REPO_DIR, "pull", "--ff-only", "origin", branch],
            capture_output=True, text=True, timeout=timeout_s,
        )
        if r.returncode != 0:
            return False, (r.stderr.strip().splitlines() or ["pull failed"])[-1][:80]
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "pull timed out"
    except Exception as e:
        return False, str(e)[:80]
