"""
fetch.py - get small SPHEREx image cutouts around a sky position.

Based on the IRSA tutorial "Download a collection of SPHEREx Spectral Image
cutouts" (caltech-ipac.github.io/irsa-tutorials/spherex-cutouts).

Install:  pip install numpy astropy astroquery

Use:
    from fetch import parse_coord, fetch_cutouts
    ra, dec = parse_coord("08h55m10.831s", "-7d14m42.53s")   # WISE 0855
    frames = fetch_cutouts(ra, dec, size_deg=0.05, bandpass="SPHEREx-D2")
    # frames = list of dicts, sorted by time, each with:
    #   "time_mjd", "wavelength_um", "image" (2D numpy array), "url"
"""
import concurrent.futures
import time

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.utils.data import conf
from astropy.wcs import WCS
from astroquery.ipac.irsa import Irsa

# Reading SPHEREx files can be slow, so allow a longer wait.
conf.remote_timeout = 120


def parse_coord(ra, dec):
    """Accept decimal degrees (210.8, 54.3) or text like '08h55m10.831s', '-7d14m42.53s'."""
    try:
        return float(ra), float(dec)
    except ValueError:
        c = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame="icrs")  # hms/dms text
        return c.ra.deg, c.dec.deg


def find_images(ra, dec, size_deg, bandpass):
    """Ask IRSA which SPHEREx images cover this point. Returns a table (oldest first)."""
    query = f"""
    SELECT
        'https://irsa.ipac.caltech.edu/' || a.uri || '?center={ra},{dec}d&size={size_deg}' AS uri,
        p.time_bounds_lower
    FROM spherex.artifact a
    JOIN spherex.plane p ON a.planeid = p.planeid
    WHERE 1 = CONTAINS(POINT('ICRS', {ra}, {dec}), p.poly)
          AND p.energy_bandpassname = '{bandpass}'
    ORDER BY p.time_bounds_lower
    """
    return Irsa.query_tap(query).to_table()


def read_one(url, time_mjd, coord, tries=3):
    """Download one cutout. Retry a few times because IRSA sometimes drops a request."""
    for attempt in range(tries):
        try:
            with fits.open(url, cache=False) as hdul:
                header = hdul[1].header            # HDU 1 = IMAGE
                image = np.array(hdul[1].data, dtype=float)
                # wavelength at the target's pixel (same trick as the tutorial)
                x, y = WCS(header).world_to_pixel(coord)
                wcs_w = WCS(header, fobj=hdul, key="W")
                wcs_w.sip = None
                wl, _ = wcs_w.pixel_to_world(x, y)
                return {
                    "time_mjd": float(time_mjd),
                    "wavelength_um": float(wl.to(u.micrometer).value),
                    "image": image,
                    "header": header.copy(),   # needed later to line frames up
                    "url": url,
                }
        except Exception:
            time.sleep(1 + attempt)
    return None  # gave up on this one


def fetch_cutouts(ra, dec, size_deg=0.05, bandpass="SPHEREx-D2",
                  max_images=60, workers=8):
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    table = find_images(ra, dec, size_deg, bandpass)
    print(f"IRSA found {len(table)} images")
    rows = list(table)[:max_images]

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(read_one, r["uri"], r["time_bounds_lower"], coord)
                   for r in rows]
        frames = [f.result() for f in futures]

    frames = [f for f in frames if f is not None]
    frames.sort(key=lambda f: f["time_mjd"])
    print(f"Got {len(frames)} cutouts ({len(rows) - len(frames)} failed)")
    return frames


if __name__ == "__main__":
    # Quick test on WISE 0855 (position is given in the challenge text)
    ra, dec = parse_coord("08h55m10.831s", "-7d14m42.53s")
    frames = fetch_cutouts(ra, dec, size_deg=0.05, bandpass="SPHEREx-D2", max_images=10)
    for f in frames:
        print(f"MJD {f['time_mjd']:.3f}  {f['wavelength_um']:.3f} um  shape {f['image'].shape}")
