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
