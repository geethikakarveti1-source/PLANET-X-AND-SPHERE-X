"""
processing.py - the image math for the SPHEREx viewer.

Each "frame" is a dict:
    time_mjd, wavelength_um, image (2D numpy array), header (astropy header),
    marker (x, y) = where your target sits in the image (pixels)

Only numpy and Pillow are needed at the top. astropy / reproject are only
imported inside the two functions that need the sky coordinates (WCS).
"""
import base64
import io
import warnings

import numpy as np
from PIL import Image


# ---------- sky alignment (needs astropy + reproject) ----------

def align_frames(frames):
    """Put every frame on the first frame's sky grid, so stars stay still and
    only moving things move. Returns (frames, all_ok).
    If 'reproject' is not installed, frames are returned unchanged."""
    try:
        from astropy.wcs import WCS
        from reproject import reproject_interp
    except ImportError:
        return frames, False
    ref_header = frames[0]["header"]
    ref_wcs = WCS(ref_header)
    shape = frames[0]["image"].shape
    out, all_ok = [], True
    for f in frames:
        try:
            arr, _ = reproject_interp((f["image"], WCS(f["header"])), ref_wcs, shape_out=shape)
            out.append(dict(f, image=arr, header=ref_header))
        except Exception:
            out.append(f)
            all_ok = False
    return out, all_ok


def marker_pixels(frames, ra, dec):
    """Work out where (ra, dec) falls in each frame's pixels. Adds f['marker']."""
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS
    import astropy.units as u
    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    for f in frames:
        try:
            x, y = WCS(f["header"]).world_to_pixel(c)
            f["marker"] = (float(x), float(y))
        except Exception:
            f["marker"] = None
    return frames


# ---------- pure numpy / Pillow part (easy to test) ----------

def _crop_common(frames):
    """If frames differ by a pixel or two in size, cut them all to the smallest."""
    h = min(f["image"].shape[0] for f in frames)
    w = min(f["image"].shape[1] for f in frames)
    return [dict(f, image=f["image"][:h, :w]) for f in frames]


def stack_frames(frames, n):
    """Average every n frames in time order. Less noise, fewer frames.
    Note: frames in a stack can have different wavelengths; we keep the range."""
    if n <= 1 or len(frames) < 2:
        return frames
    frames = _crop_common(frames)
    out = []
    for i in range(0, len(frames), n):
        g = frames[i:i + n]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            img = np.nanmean(np.stack([f["image"] for f in g]), axis=0)
        wls = [f["wavelength_um"] for f in g]
        marks = [f["marker"] for f in g if f.get("marker")]
        out.append({
            "time_mjd": float(np.mean([f["time_mjd"] for f in g])),
            "wavelength_um": float(np.mean(wls)),
            "wl_min": float(min(wls)), "wl_max": float(max(wls)),
            "image": img,
            "marker": tuple(np.mean(marks, axis=0)) if marks else None,
            "n": len(g),
        })
    return out


def subtract_median(frames):
    """Subtract the middle (median) image of all frames from each frame.
    Things that never move (stars, galaxies) fade out; movers stay.
    Returns (frames, ok). Needs at least 3 frames."""
    if len(frames) < 3:
        return frames, False
    frames = _crop_common(frames)
    cube = np.stack([f["image"] for f in frames])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(cube, axis=0)
    return [dict(f, image=f["image"] - med) for f in frames], True


def to_png_uri(img, diverging=False):
    """2D array -> PNG as a data: link. North is up (image is flipped).
    diverging=True is for subtracted images: mid-gray means 'no change'."""
    good = img[np.isfinite(img)]
    if good.size == 0:
        scaled = np.zeros(img.shape)
    elif diverging:
        lim = float(np.percentile(np.abs(good), 99.5))
        if lim <= 0:
            lim = 1e-9
        scaled = np.clip(np.nan_to_num(img, nan=0.0) / (2 * lim) + 0.5, 0, 1)
    else:
        lo, hi = np.percentile(good, [1, 99.5])
        if hi <= lo:
            hi = lo + 1e-9
        scaled = np.clip((np.nan_to_num(img, nan=lo) - lo) / (hi - lo), 0, 1)
        scaled = np.arcsinh(10 * scaled) / np.arcsinh(10)   # makes faint things show
    im = Image.fromarray((scaled * 255).astype(np.uint8)[::-1])
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def marker_fraction(x, y, shape):
    """Pixel (x, y) -> position as fractions (0..1) of the picture, matching
    the up-down flip done in to_png_uri. Used to draw the target circle."""
    h, w = shape
    return (x + 0.5) / w, (h - 1 - y + 0.5) / h
