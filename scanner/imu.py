"""BNO055 IMU on a background thread.

Publishes the latest orientation quaternion for save_capture() to stamp
onto each capture. Graceful fallback: if the library or chip is absent,
latest() returns None and the scanner keeps working without IMU data.
"""

import threading
import time

from . import logger as log_mod

log = log_mod.get("scanner.imu")

try:
    import board
    import adafruit_bno055
    _IMPORT_OK = True
    _IMPORT_ERR = None
except Exception as e:
    _IMPORT_OK = False
    _IMPORT_ERR = str(e)


# The Teyleten/GY-BNO055 breakouts float ADDR high -> 0x29. Proper
# Adafruit boards tie it low -> 0x28. Try both at construction.
BNO055_ADDRS = (0x29, 0x28)


class ImuReader:
    """Thread-safe BNO055 poller.

    Runs at ~20 Hz on a daemon thread. latest() returns the most recent
    valid quaternion as (w, x, y, z) or None if no reading yet. A missing
    chip is non-fatal - the rest of the scanner runs without IMU.
    """

    def __init__(self, addresses=BNO055_ADDRS, poll_hz=20.0):
        self._lock  = threading.Lock()
        self._quat  = None
        self._ok    = 0
        self._fail  = 0
        self._stop  = False
        self._imu   = None
        self._addr  = None

        if not _IMPORT_OK:
            log.warning("imu: library not available (%s) - continuing without IMU",
                        _IMPORT_ERR)
            return

        try:
            i2c = board.I2C()
        except Exception as e:
            log.warning("imu: I2C bus init failed (%s) - continuing without IMU", e)
            return

        for addr in addresses:
            try:
                self._imu  = adafruit_bno055.BNO055_I2C(i2c, address=addr)
                self._addr = addr
                log.info("imu: BNO055 @ 0x%02x ready", addr)
                break
            except Exception as e:
                log.debug("imu: 0x%02x: %s", addr, e)

        if self._imu is None:
            log.warning("imu: BNO055 not found at %s - continuing without IMU",
                        ", ".join(f"0x{a:02x}" for a in addresses))
            return

        self._period = 1.0 / poll_hz
        self._t = threading.Thread(target=self._loop, daemon=True, name="imu")
        self._t.start()

    def _loop(self):
        while not self._stop:
            t0 = time.time()
            try:
                q = self._imu.quaternion
                if q is not None and all(v is not None for v in q):
                    with self._lock:
                        self._quat = tuple(float(v) for v in q)
                        self._ok += 1
                else:
                    self._fail += 1
            except Exception:
                # BNO055 can occasionally NACK during heavy I2C traffic.
                # Not a crash - just try again next tick.
                self._fail += 1
            dt = time.time() - t0
            if dt < self._period:
                time.sleep(self._period - dt)

    def latest(self):
        """Return (w, x, y, z) tuple or None if no valid reading yet."""
        with self._lock:
            return self._quat

    def available(self):
        return self._imu is not None

    def stats(self):
        return self._ok, self._fail

    def stop(self):
        self._stop = True
