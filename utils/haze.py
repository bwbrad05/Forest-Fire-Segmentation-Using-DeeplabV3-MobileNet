# Smoke / haze detection and a prototype haze correction for Landsat-8 tiles.
#
# Why these signals
# -----------------
# Smoke aerosol scatters short wavelengths strongly and long wavelengths weakly, so
# under smoke the blue band rises while SWIR barely changes. Clouds are different:
# they are bright in SWIR1 as well. Two quantities therefore separate the cases:
#   * HOT (Haze Optimized Transform, Zhang et al. 2002) -- blue minus a fraction of
#     red. Over clear land blue and red are highly correlated, so departures from
#     that relation measure haze thickness. Here the clear-line slope is fixed at
#     HOT_SLOPE instead of being fitted per scene (a simplification).
#   * SWIR1 reflectance -- above CLOUD_SWIR1 a pixel is treated as cloud, not smoke.
#     The value is read off this dataset (cloud pixels reach 0.6-0.9, a dense-smoke
#     tile peaks near 0.22), not taken from the literature.
#   * Water -- both SWIR bands very dark. Water is blue-heavy and would otherwise
#     score as haze (rivers, sea and turbid coast all did in a first version). An
#     NIR-below-green rule was tried first and rejected: it called 4.7% of all burned
#     pixels water, because char is dark in NIR; the SWIR rule calls 0.2%.
#
# The correction is the dark-object-subtraction half of the HOT+DOS method
# (Zhang et al. 2002; Makarau et al. 2014): per HOT level, the darkest pixels of each
# visible band are assumed to be pure haze offset and are subtracted. Only B1-B4 are
# corrected, because smoke is close to transparent in NIR/SWIR, which is exactly the
# part of the spectrum the burn signal lives in. It is a prototype and has not been
# validated against ground truth.
#
# Band order of the 8-channel stack: 0=B1 Coastal 1=B2 Blue 2=B3 Green 3=B4 Red
# 4=B5 NIR 5=B6 SWIR1 6=B7 SWIR2 7=B9 Cirrus. Values are Landsat Collection-2
# surface-reflectance digital numbers; the conversion below is only used to give the
# thresholds a stable scale, and every threshold that matters is calibrated from the
# data itself (see ``scripts/smoke_score.py --calibrate``).

import numpy as np

SR_SCALE = 2.75e-5
SR_OFFSET = -0.2

HOT_SLOPE = 0.5
CLOUD_SWIR1 = 0.20
WATER_SWIR1 = 0.0
WATER_SWIR2 = -0.10
# A tile whose clear (non-cloud, non-water) share is below this has no reliable haze score.
MIN_CLEAR_FRAC = 0.30

# Calibrated on the 227 tiles by ``python scripts/smoke_score.py --calibrate``:
# HOT_SMOKE_THR  = 95th percentile of HOT over cloud-free pixels of the cleanest
#                  quarter of tiles, i.e. the haze level clear skies rarely exceed.
# REF_DARK       = 2nd percentile of B1..B4 over the same pixels, the "no haze"
#                  dark-object level the correction subtracts down to.
HOT_SMOKE_THR = 0.0447
REF_DARK = (-0.0210, -0.0593, -0.0844, -0.1297)


def to_reflectance(dn):
    return np.asarray(dn, dtype=np.float64) * SR_SCALE + SR_OFFSET


def to_dn(refl):
    return (np.asarray(refl, dtype=np.float64) - SR_OFFSET) / SR_SCALE


def hot_map(refl):
    """Haze Optimized Transform per pixel from a (>=7, H, W) reflectance stack."""
    return refl[1] - HOT_SLOPE * refl[3]


def cloud_mask(refl):
    return refl[5] > CLOUD_SWIR1


def water_mask(refl):
    return (refl[5] < WATER_SWIR1) & (refl[6] < WATER_SWIR2)


def nodata_mask(dn):
    return np.asarray(dn)[:7].sum(axis=0) == 0


def smoke_features(dn, hot_thr=None):
    """Per-tile descriptors from a raw (8, H, W) digital-number array.

    hot_median  median HOT over valid pixels that are neither cloud nor water -- the
                main score, an overall haze level (NaN when under MIN_CLEAR_FRAC of the
                tile is clear);
    smoke_frac  share of valid pixels that are hazy (HOT above ``hot_thr``), not
                cloud and not water;
    cloud_frac  share of valid pixels that are bright in SWIR1;
    water_frac  share of valid pixels that look like water.
    """
    thr = HOT_SMOKE_THR if hot_thr is None else hot_thr
    refl = to_reflectance(dn[:7])
    valid = ~nodata_mask(dn)
    cloud = cloud_mask(refl)
    water = water_mask(refl)
    clear = valid & ~cloud & ~water
    hot = hot_map(refl)
    n_valid = max(int(valid.sum()), 1)
    return dict(
        hot_median=float(np.median(hot[clear])) if clear.sum() >= MIN_CLEAR_FRAC * n_valid else float("nan"),
        smoke_frac=float(((hot > thr) & clear).sum()) / n_valid,
        cloud_frac=float((cloud & valid).sum()) / n_valid,
        water_frac=float((water & valid & ~cloud).sum()) / n_valid,
    )


def dehaze_hot_dos(dn, ref_dark=None, n_bins=24, dark_percentile=2.0, min_pixels=200):
    """HOT-binned dark-object subtraction on the visible bands.

    ``dn`` is (8, H, W) raw digital numbers; the result has the same shape and dtype.
    Cloud, water and no-data pixels, and all bands other than B1-B4, are returned unchanged.
    """
    ref = REF_DARK if ref_dark is None else ref_dark
    arr = np.asarray(dn)
    refl = to_reflectance(arr[:7])
    valid = ~nodata_mask(arr)
    cloud = cloud_mask(refl)
    water = water_mask(refl)
    skip = cloud | water | ~valid
    use = ~skip
    if use.sum() < 10 * min_pixels:
        return arr.copy()

    hot = hot_map(refl)
    edges = np.unique(np.quantile(hot[use], np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return arr.copy()
    idx = np.clip(np.digitize(hot, edges[1:-1]), 0, len(edges) - 2)

    centers, members = [], []
    for k in range(len(edges) - 1):
        sel = use & (idx == k)
        if sel.sum() >= min_pixels:
            centers.append(float(np.median(hot[sel])))
            members.append(sel)
    if len(centers) < 2:
        return arr.copy()
    centers = np.asarray(centers)

    out = refl.copy()
    for b in range(4):
        dark = np.array([np.percentile(refl[b][sel], dark_percentile) for sel in members])
        offset = np.maximum(dark - ref[b], 0.0)
        amount = np.interp(hot, centers, offset)
        out[b] = np.where(skip, refl[b], refl[b] - amount)

    result = arr.astype(np.float64).copy()
    result[:7] = to_dn(out)
    result[:7][:, ~valid] = arr[:7][:, ~valid]
    info = np.iinfo(arr.dtype) if np.issubdtype(arr.dtype, np.integer) else None
    if info is not None:
        result = np.clip(np.rint(result), info.min, info.max)
    return result.astype(arr.dtype)
