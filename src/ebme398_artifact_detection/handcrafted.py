from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.feature import local_binary_pattern
from skimage.filters import sobel
from skimage.measure import shannon_entropy

try:
    from skimage.feature import graycomatrix as _glcm
except ImportError:  # pragma: no cover
    from skimage.feature import greycomatrix as _glcm


_GLCM_DIST = [1, 2, 4, 8]
_GLCM_ANG = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
_GLCM_LEVELS = 32
_LBP_P, _LBP_R, _LBP_METHOD = 24, 3, "uniform"
_LBP_BINS = _LBP_P + 2


def glcm_11_features(P: np.ndarray) -> np.ndarray:
    P = P.astype(np.float64)
    P = P / (P.sum() + 1e-12)
    levels = P.shape[0]
    i, j = np.indices((levels, levels))
    px = P.sum(axis=1)
    py = P.sum(axis=0)
    ux = (i[:, 0] * px).sum()
    uy = (j[0, :] * py).sum()
    sx = np.sqrt(((i[:, 0] - ux) ** 2 * px).sum()) + 1e-12
    sy = np.sqrt(((j[0, :] - uy) ** 2 * py).sum()) + 1e-12
    asm = (P**2).sum()
    contrast = ((i - j) ** 2 * P).sum()
    corr = (((i - ux) * (j - uy) * P).sum()) / (sx * sy)
    variance = (((i - ux) ** 2) * P).sum()
    idm = (P / (1.0 + (i - j) ** 2)).sum()
    pxpy = np.zeros(2 * levels - 1)
    pxmy = np.zeros(levels)
    for a in range(levels):
        for b in range(levels):
            pxpy[a + b] += P[a, b]
            pxmy[abs(a - b)] += P[a, b]
    k = np.arange(2 * levels - 1)
    sum_avg = (k * pxpy).sum()
    sum_entropy = -(pxpy * np.log(pxpy + 1e-12)).sum()
    sum_var = ((k - sum_avg) ** 2 * pxpy).sum()
    diff_k = np.arange(levels)
    diff_entropy = -(pxmy * np.log(pxmy + 1e-12)).sum()
    diff_var = ((diff_k - (diff_k * pxmy).sum()) ** 2 * pxmy).sum()
    entropy = -(P * np.log(P + 1e-12)).sum()
    return np.array(
        [asm, contrast, corr, variance, idm, sum_avg, sum_var, sum_entropy, entropy, diff_var, diff_entropy],
        dtype=np.float32,
    )


def glcm_44(gray_u8: np.ndarray) -> tuple[np.ndarray, list[str]]:
    quantized = (gray_u8.astype(np.float32) / 255.0 * (_GLCM_LEVELS - 1)).astype(np.uint8)
    glcm = _glcm(quantized, distances=_GLCM_DIST, angles=_GLCM_ANG, levels=_GLCM_LEVELS, symmetric=True, normed=True)
    features = []
    names: list[str] = []
    for distance_index, distance in enumerate(_GLCM_DIST):
        per_angle = [glcm_11_features(glcm[:, :, distance_index, angle_index]) for angle_index in range(len(_GLCM_ANG))]
        features.append(np.mean(np.stack(per_angle, axis=0), axis=0))
        names += [f"glcm11_{k}_d{distance}" for k in range(11)]
    return np.concatenate(features), names


def _boxcount(mask: np.ndarray, k: int) -> int:
    reduced = np.add.reduceat(np.add.reduceat(mask, np.arange(0, mask.shape[0], k), axis=0), np.arange(0, mask.shape[1], k), axis=1)
    return np.count_nonzero((reduced > 0) & (reduced < k * k))


def fractal_dimension(mask: np.ndarray) -> float:
    mask = mask.astype(bool)
    p = min(mask.shape)
    n = 2 ** int(np.floor(np.log2(p)))
    mask = mask[:n, :n]
    sizes = 2 ** np.arange(int(np.log2(n)), 1, -1)
    counts = np.array([_boxcount(mask, k) for k in sizes], dtype=np.float64)
    coeffs = np.polyfit(np.log(1.0 / sizes), np.log(counts + 1e-12), 1)
    return float(coeffs[0])


def fractal_4(gray_u8: np.ndarray) -> tuple[np.ndarray, list[str]]:
    gray = gray_u8.astype(np.float32)
    masks = [gray < 220, gray < 200, sobel(gray / 255.0) > np.percentile(sobel(gray / 255.0), 75), gray < np.median(gray)]
    feats = np.array([fractal_dimension(mask) for mask in masks], dtype=np.float32)
    names = ["fd_notwhite220", "fd_notwhite200", "fd_edges75pct", "fd_below_median"]
    return feats, names


def lbp_30(gray_u8: np.ndarray) -> tuple[np.ndarray, list[str]]:
    lbp = local_binary_pattern(gray_u8, P=_LBP_P, R=_LBP_R, method=_LBP_METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, _LBP_BINS + 1), density=True)
    hist = hist.astype(np.float32)
    extra = np.array([lbp.mean(), lbp.std(), shannon_entropy(lbp), hist[-1]], dtype=np.float32)
    names = [f"lbp_hist_{i}" for i in range(len(hist))] + ["lbp_mean", "lbp_std", "lbp_entropy", "lbp_nonuniform_frac"]
    return np.concatenate([hist, extra]), names


def hsv_6(hsv: np.ndarray) -> tuple[np.ndarray, list[str]]:
    h = hsv[:, :, 0].astype(np.float32) / 179.0
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    feats = np.array([h.mean(), s.mean(), v.mean(), h.std(), s.std(), v.std()], dtype=np.float32)
    return feats, ["h_mean", "s_mean", "v_mean", "h_std", "s_std", "v_std"]


def hs_hist_26(hsv: np.ndarray) -> tuple[np.ndarray, list[str]]:
    h = hsv[:, :, 0].astype(np.float32) / 179.0
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    hist, _, _ = np.histogram2d(h.ravel(), s.ravel(), bins=[np.linspace(0, 1, 14), np.array([0.0, 0.5, 1.0])], density=True)
    return hist.flatten().astype(np.float32), [f"hs_hist_{i}" for i in range(26)]


def entropy_9(rgb_u8: np.ndarray, gray_u8: np.ndarray, hsv: np.ndarray) -> tuple[np.ndarray, list[str]]:
    values = []
    names = []
    for channel, channel_name, divisor in ((hsv[:, :, 0], "H", 179.0), (hsv[:, :, 1], "S", 255.0), (hsv[:, :, 2], "V", 255.0)):
        values.append(float(shannon_entropy(channel.astype(np.float32) / divisor)))
        names.append(f"entropy_{channel_name}")
    for idx, channel_name in enumerate(("R", "G", "B")):
        values.append(float(shannon_entropy(rgb_u8[:, :, idx] / 255.0)))
        names.append(f"entropy_{channel_name}")
    values.append(float(shannon_entropy(gray_u8 / 255.0)))
    names.append("entropy_gray")
    high_sat = hsv[:, :, 1] > 30
    values.append(float(shannon_entropy(gray_u8[high_sat] / 255.0)) if high_sat.mean() > 0.01 else 0.0)
    names.append("entropy_gray_highSat")
    edge_map = sobel(gray_u8.astype(np.float32) / 255.0)
    values.append(float(shannon_entropy(edge_map)))
    names.append("entropy_edges")
    return np.array(values, dtype=np.float32), names


def fft_power_15(gray_u8: np.ndarray, n_bands: int = 15) -> tuple[np.ndarray, list[str]]:
    gray = gray_u8.astype(np.float32) / 255.0
    fourier = np.fft.fftshift(np.fft.fft2(gray))
    power = np.abs(fourier) ** 2
    h, w = power.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    bands = np.linspace(0, rr.max(), n_bands + 1)
    features = []
    for idx in range(n_bands):
        mask = (rr >= bands[idx]) & (rr < bands[idx + 1])
        features.append(float(power[mask].mean()) if mask.any() else 0.0)
    feats = np.asarray(features, dtype=np.float32)
    feats = feats / (feats.sum() + 1e-12)
    return feats, [f"fft_band_{idx}" for idx in range(n_bands)]


def extract_kba_features(image_u8_chw: torch.Tensor) -> tuple[np.ndarray, list[str]]:
    rgb = image_u8_chw.permute(1, 2, 0).cpu().numpy()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    parts = []
    names: list[str] = []
    for extractor in (glcm_44, fractal_4, lbp_30):
        feat, feat_names = extractor(gray)
        parts.append(feat)
        names += feat_names
    for extractor in (hsv_6, hs_hist_26):
        feat, feat_names = extractor(hsv)
        parts.append(feat)
        names += feat_names
    feat, feat_names = entropy_9(rgb, gray, hsv)
    parts.append(feat)
    names += feat_names
    feat, feat_names = fft_power_15(gray)
    parts.append(feat)
    names += feat_names
    return np.concatenate(parts).astype(np.float32), names


def featurize_meta_csv(meta_csv: str | Path, out_csv: str | Path) -> tuple[Path, list[str]]:
    meta = pd.read_csv(meta_csv)
    rows = []
    feature_names: list[str] | None = None
    passthrough_cols = [column for column in ("slide_id", "tile_idx", "x", "y", "y0") if column in meta.columns]
    for row in meta.itertuples(index=False):
        image_u8 = torch.load(row.path)
        features, names = extract_kba_features(image_u8)
        if feature_names is None:
            feature_names = names
        item = {"path": row.path, "y_label": int(row.y_label)}
        for column in passthrough_cols:
            item[column] = getattr(row, column)
        item.update({feature_names[idx]: float(features[idx]) for idx in range(len(feature_names))})
        rows.append(item)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv, feature_names or []
