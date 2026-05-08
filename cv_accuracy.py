"""
cv_accuracy.py
--------------
Task Two: The CV Accuracy Challenge.
Face Detection using Haar Cascades on original vs. pre-processed images.
"""

import time
import cv2
import numpy as np
from image_processor import histogram_equalization, adjust_brightness


def _load_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    return cascade


def _degrade_image(image: np.ndarray) -> np.ndarray:
    """Simulate a low-quality image: darken + add Gaussian noise."""
    dark = cv2.convertScaleAbs(image, alpha=0.35, beta=-30)
    noise = np.random.normal(0, 25, dark.shape).astype(np.int16)
    noisy = np.clip(dark.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def detect_faces(image: np.ndarray, cascade: cv2.CascadeClassifier):
    """
    Run face detection.
    Returns (faces_rect_list, elapsed_ms, annotated_image).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    t0   = time.time()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    elapsed_ms = (time.time() - t0) * 1000

    annotated = image.copy()
    if len(faces):
        for (x, y, w, h) in faces:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return faces, elapsed_ms, annotated


def run_accuracy_experiment(image: np.ndarray):
    """
    Full pipeline:
    1. Degrade the image.
    2. Detect faces on the degraded image (baseline).
    3. Enhance with histogram equalization.
    4. Detect faces on the enhanced image.
    5. Return a results dict.
    """
    cascade = _load_cascade()

    # ── Step 1: Degrade ──────────────────────────────
    degraded = _degrade_image(image)

    # ── Step 2: Baseline detection ───────────────────
    orig_faces, orig_time, orig_annotated = detect_faces(degraded, cascade)

    # ── Step 3: Pre-process (enhance) ────────────────
    enhanced = histogram_equalization(degraded)

    # ── Step 4: Post-processing detection ────────────
    enh_faces, enh_time, enh_annotated = detect_faces(enhanced, cascade)

    # ── Step 5: Build results dict ───────────────────
    # Accuracy: use the count ratio vs. reference run on the original
    ref_faces, _, _ = detect_faces(image, cascade)
    ref_count = max(len(ref_faces), 1)

    orig_accuracy = min(100.0, (len(orig_faces) / ref_count) * 100)
    enh_accuracy  = min(100.0, (len(enh_faces)  / ref_count) * 100)

    results = {
        "original_image":     image,
        "degraded_image":     degraded,
        "enhanced_image":     enhanced,
        "orig_annotated":     orig_annotated,
        "enh_annotated":      enh_annotated,
        "ref_faces":          len(ref_faces),
        "orig_faces":         len(orig_faces),
        "enh_faces":          len(enh_faces),
        "orig_accuracy_pct":  round(orig_accuracy, 1),
        "enh_accuracy_pct":   round(enh_accuracy, 1),
        "orig_time_ms":       round(orig_time, 2),
        "enh_time_ms":        round(enh_time, 2),
        "overhead_ms":        round(enh_time - orig_time, 2),
    }
    return results
