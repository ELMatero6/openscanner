"""Display + input abstraction. Pygame on SDL2.

Why not cv2.imshow?
  - cv2.imshow() needs GTK/Qt, which needs X or Wayland. That rules out
    Pi OS Lite without a compositor.
  - SDL2 has a KMSDRM backend - it draws directly to the kernel's KMS
    interface, no compositor required. Booting straight from Pi OS Lite
    into the scanner with no desktop is then a one-line systemd unit.
  - Pygame's event loop is non-blocking and supports both mouse + touch.
    cv2.waitKey() pumps GTK's loop; if the main thread blocks for any
    reason the window goes unresponsive.

Public surface is intentionally small:
  init(w, h)    set up the screen
  show(frame)   blit a HxWx3 BGR numpy array
  events()      drain SDL events into a list of high-level event objects
  quit()        clean shutdown

Events are typed objects (Tap, Drag, Release, Key, Quit) - see classes
below. Touch and mouse both produce Tap/Drag/Release; calling code
shouldn't care which.
"""

import logging
import os

# These MUST be set before pygame is imported. Pi OS Lite users get
# KMSDRM (no compositor); desktop users get whatever SDL picks (x11/wayland).
os.environ.setdefault("SDL_VIDEODRIVER",
                      os.environ.get("OPENSCANNER_SDL_DRIVER", "kmsdrm"))
os.environ.setdefault("SDL_FBDEV", "/dev/fb0")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402

log = logging.getLogger("scanner.display")


class Tap:
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)


class Drag:
    __slots__ = ("x", "y", "dx", "dy")

    def __init__(self, x, y, dx, dy):
        self.x = int(x)
        self.y = int(y)
        self.dx = int(dx)
        self.dy = int(dy)


class Release:
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)


class Key:
    __slots__ = ("char",)

    def __init__(self, char):
        self.char = char


class Quit:
    pass


_screen = None
_size = (0, 0)


def init(w, h, title="openscanner", fullscreen=True):
    """Open a fullscreen window of the requested size."""
    global _screen, _size
    pygame.display.init()
    pygame.font.quit()  # we don't use pygame fonts; cv2 draws all text
    drivers_tried = [os.environ.get("SDL_VIDEODRIVER", "kmsdrm")]
    flags = pygame.FULLSCREEN if fullscreen else 0
    try:
        _screen = pygame.display.set_mode((w, h), flags)
    except pygame.error as e:
        log.warning("display init failed on %s (%s); falling back to dummy",
                    drivers_tried[0], e)
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.display.quit()
        pygame.display.init()
        _screen = pygame.display.set_mode((w, h), 0)
    pygame.display.set_caption(title)
    pygame.mouse.set_visible(False)
    _size = (w, h)
    log.info("display %dx%d driver=%s", w, h, pygame.display.get_driver())


def show(frame_bgr):
    """Blit a HxWx3 BGR uint8 numpy array. Resizes to fit if needed."""
    h, w = frame_bgr.shape[:2]
    rgb = frame_bgr[:, :, ::-1]
    surf = pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")
    if (w, h) != _size:
        surf = pygame.transform.scale(surf, _size)
    _screen.blit(surf, (0, 0))
    pygame.display.flip()


def events():
    """Drain pending SDL events into our high-level event list.

    Mouse + touch are unified: both surface as Tap / Drag / Release.
    Touch coords arrive normalised (0..1) so we scale to screen pixels.
    """
    out = []
    for ev in pygame.event.get():
        t = ev.type
        if t == pygame.QUIT:
            out.append(Quit())
        elif t == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            out.append(Tap(ev.pos[0], ev.pos[1]))
        elif t == pygame.MOUSEBUTTONUP and ev.button == 1:
            out.append(Release(ev.pos[0], ev.pos[1]))
        elif t == pygame.MOUSEMOTION and ev.buttons[0]:
            out.append(Drag(ev.pos[0], ev.pos[1], ev.rel[0], ev.rel[1]))
        elif t == pygame.FINGERDOWN:
            out.append(Tap(ev.x * _size[0], ev.y * _size[1]))
        elif t == pygame.FINGERUP:
            out.append(Release(ev.x * _size[0], ev.y * _size[1]))
        elif t == pygame.FINGERMOTION:
            out.append(Drag(ev.x * _size[0], ev.y * _size[1],
                            ev.dx * _size[0], ev.dy * _size[1]))
        elif t == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                out.append(Key("\x1b"))
            elif ev.key == pygame.K_q:
                out.append(Key("q"))
    return out


def quit():
    pygame.display.quit()
    pygame.quit()
