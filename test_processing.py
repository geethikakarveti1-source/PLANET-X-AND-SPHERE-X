"""
Tests that need NO internet: they use fake images.
Run:  python test_processing.py
"""
import base64
import io

import numpy as np
from PIL import Image

from processing import marker_fraction, stack_frames, subtract_median, to_png_uri


def fake_frames(n=7, size=30):
    """A fixed star at (10, 10) plus a dot that moves 1 pixel per frame."""
    frames = []
    rng = np.random.default_rng(0)
    for i in range(n):
        img = rng.normal(0, 0.1, (size, size))
        img[10, 10] += 50          # star that never moves
        img[20, 5 + i] += 30       # mover: x goes 5, 6, 7, ...
        frames.append({"time_mjd": 61000.0 + i, "wavelength_um": 1.2 + 0.01 * i,
                       "image": img, "marker": (15.0, 15.0)})
    return frames


def test_stack():
    frames = [{"time_mjd": float(i), "wavelength_um": 1.0 + i,
               "image": np.full((4, 4), float(i)), "marker": (1.0, 1.0)} for i in range(4)]
    out = stack_frames(frames, 2)
    assert len(out) == 2
    assert np.allclose(out[0]["image"], 0.5) and np.allclose(out[1]["image"], 2.5)
    assert out[0]["wl_min"] == 1.0 and out[0]["wl_max"] == 2.0
    assert stack_frames(frames, 1) is frames           # n=1 changes nothing


def test_subtract_removes_star_keeps_mover():
    frames, ok = subtract_median(fake_frames())
    assert ok
    f3 = frames[3]["image"]
    assert abs(f3[10, 10]) < 1.0                       # star is gone
    assert f3[20, 8] > 20                              # mover at x = 5 + 3 is still there
    assert abs(f3[20, 5]) < 1.0                        # and not where it was at frame 0


def test_subtract_needs_three_frames():
    _, ok = subtract_median(fake_frames(2))
    assert not ok


def test_nan_is_ok():
    frames = fake_frames()
    frames[2]["image"][0, 0] = np.nan
    out, ok = subtract_median(frames)
    assert ok and np.isfinite(out[0]["image"][5, 5])
    stacked = stack_frames(frames, 3)
    assert len(stacked) == 3


def test_png():
    for diverging in (False, True):
        uri = to_png_uri(fake_frames()[0]["image"], diverging=diverging)
        assert uri.startswith("data:image/png;base64,")
        im = Image.open(io.BytesIO(base64.b64decode(uri.split(",")[1])))
        assert im.size == (30, 30)
    # an all-NaN image must not crash
    to_png_uri(np.full((5, 5), np.nan))


def test_marker_fraction():
    # centre pixel of a 30x30 image sits near the middle of the picture
    fx, fy = marker_fraction(14.5, 14.5, (30, 30))
    assert abs(fx - 0.5) < 0.02 and abs(fy - 0.5) < 0.02
    # row 0 is the BOTTOM row of the picture (north up), so its fraction is near 1
    _, fy0 = marker_fraction(0, 0, (30, 30))
    assert fy0 > 0.95


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("all good")
