"""
image_processor.py
------------------
All image processing / filter operations for Vision Editor.
Each function receives a BGR numpy array and returns a processed BGR numpy array.
"""

import cv2
import numpy as np


# ──────────────────────────────────────────────
#  FILTERS  (kernel-based convolutions)
# ──────────────────────────────────────────────

def apply_filter(image: np.ndarray, filter_name: str, ksize: int = 5) -> np.ndarray:
    """
    Dispatcher for all 6 required filters.
    ksize must be odd and >= 3.
    """
    if ksize % 2 == 0:
        ksize += 1          # force odd
    ksize = max(3, ksize)

    dispatch = {
        "Laplacian":  _laplacian,
        "Sobel":      _sobel,
        "Averaging":  _averaging,
        "Median":     _median,
        "Gaussian":   _gaussian,
        "Bilateral":  _bilateral,
    }
    fn = dispatch.get(filter_name)
    if fn is None:
        raise ValueError(f"Unknown filter: {filter_name}")
    return fn(image, ksize)


def _laplacian(img, ksize):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap  = cv2.Laplacian(gray, cv2.CV_64F, ksize=min(ksize, 31))
    lap  = cv2.convertScaleAbs(lap)
    return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)


def _sobel(img, ksize):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ksize = ksize if ksize <= 7 else 7      # Sobel max kernel = 7
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(sx, sy)
    mag = cv2.convertScaleAbs(mag)
    return cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)


def _averaging(img, ksize):
    return cv2.blur(img, (ksize, ksize))


def _median(img, ksize):
    return cv2.medianBlur(img, ksize)


def _gaussian(img, ksize):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def _bilateral(img, ksize):
    # diameter ~= ksize, sigmaColor and sigmaSpace fixed sensible values
    d = ksize if ksize <= 15 else 15
    return cv2.bilateralFilter(img, d, sigmaColor=75, sigmaSpace=75)


# ──────────────────────────────────────────────
#  POINT OPERATIONS
# ──────────────────────────────────────────────

def adjust_brightness(image: np.ndarray, value: float) -> np.ndarray:
    """value ∈ [-100, 100]"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + int(value * 2.55), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def adjust_contrast(image: np.ndarray, value: float) -> np.ndarray:
    """value ∈ [-100, 100]  →  alpha ∈ [0.0, 3.0]"""
    alpha = 1.0 + value / 100.0
    alpha = max(0.0, alpha)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)


def adjust_exposure(image: np.ndarray, value: float) -> np.ndarray:
    """Simulated exposure via gamma: value ∈ [-100, 100]"""
    gamma = np.power(2.0, value / 50.0)          # maps -100→0.25, 0→1, 100→4
    lut = np.array([min(255, int((i / 255.0) ** (1.0 / gamma) * 255))
                    for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, lut)


def adjust_highlights(image: np.ndarray, value: float) -> np.ndarray:
    """Brighten/darken only bright areas (luminance > 128)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    v   = hsv[:, :, 2]
    mask = v / 255.0                              # weight by brightness
    hsv[:, :, 2] = np.clip(v + value * mask * 1.27, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def adjust_shadows(image: np.ndarray, value: float) -> np.ndarray:
    """Brighten/darken only dark areas (luminance < 128)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    v   = hsv[:, :, 2]
    mask = 1.0 - (v / 255.0)                     # weight by darkness
    hsv[:, :, 2] = np.clip(v + value * mask * 1.27, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def adjust_saturation(image: np.ndarray, value: float) -> np.ndarray:
    """value ∈ [-100, 100]"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + int(value * 1.27), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def adjust_warmth(image: np.ndarray, value: float) -> np.ndarray:
    """value ∈ [-100, 100] – positive = warmer (more red), negative = cooler (more blue)"""
    img = image.astype(np.int32)
    shift = int(value * 0.5)
    img[:, :, 2] = np.clip(img[:, :, 2] + shift, 0, 255)   # R channel
    img[:, :, 0] = np.clip(img[:, :, 0] - shift, 0, 255)   # B channel
    return img.astype(np.uint8)


def apply_vignette(image: np.ndarray, value: float) -> np.ndarray:
    """value ∈ [0, 100]  – 0 = no vignette, 100 = heavy vignette"""
    if value <= 0:
        return image.copy()
    h, w = image.shape[:2]
    sigma = 1.0 - (value / 100.0) * 0.8        # sigma ∈ [0.2, 1.0]
    Y, X  = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    dist   = np.sqrt(((X - cx) / (w / 2)) ** 2 + ((Y - cy) / (h / 2)) ** 2)
    vig    = np.exp(-dist ** 2 / (2 * sigma ** 2))
    vig    = (vig * 255).astype(np.uint8)
    vig3   = cv2.merge([vig, vig, vig])
    return cv2.multiply(image, vig3, scale=1.0 / 255.0).astype(np.uint8)


# ──────────────────────────────────────────────
#  GEOMETRIC
# ──────────────────────────────────────────────

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    M    = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))


def zoom_image(image: np.ndarray, scale: float,
               method: str = "Bilinear") -> np.ndarray:
    h, w  = image.shape[:2]
    nh, nw = int(h * scale), int(w * scale)
    interp = cv2.INTER_LINEAR if method == "Bilinear" else cv2.INTER_NEAREST
    zoomed = cv2.resize(image, (nw, nh), interpolation=interp)
    # crop/pad back to original size
    if scale >= 1.0:
        y0 = (nh - h) // 2
        x0 = (nw - w) // 2
        return zoomed[y0:y0 + h, x0:x0 + w]
    else:
        canvas = np.zeros_like(image)
        y0 = (h - nh) // 2
        x0 = (w - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = zoomed
        return canvas


# ──────────────────────────────────────────────
#  ENHANCEMENT
# ──────────────────────────────────────────────

def histogram_equalization(image: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def gamma_correction(image: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    gamma = max(0.1, gamma)
    lut   = np.array([(i / 255.0) ** (1.0 / gamma) * 255
                      for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, lut)


# ──────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────

def compute_histogram(image: np.ndarray) -> dict:
    """Returns per-channel histogram arrays (B, G, R) for a BGR image."""
    hists = {}
    colors = ("B", "G", "R")
    for i, c in enumerate(colors):
        hists[c] = cv2.calcHist([image], [i], None, [256], [0, 256]).flatten()
    return hists
