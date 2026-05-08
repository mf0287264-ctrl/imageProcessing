"""
cv_accuracy.py
--------------
Task Two: CV Accuracy Challenge for face detection.

Runs the same OpenCV Haar-cascade detector on:
1. The uploaded low-quality original image.
2. A processed version improved with Task One histogram equalization.
"""

import time

import cv2
import numpy as np

from image_processor import histogram_equalization


def _load_cascades():
    """Load frontal and profile cascades for wider face angle coverage."""
    frontal = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    alt2 = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    profile = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_profileface.xml")
    return frontal, alt2, profile


def _nms_boxes(boxes, overlap_thresh=0.35):
    """Remove duplicate overlapping bounding boxes."""
    if len(boxes) == 0:
        return []

    arr = np.array(boxes, dtype=np.float32)
    x1, y1 = arr[:, 0], arr[:, 1]
    x2, y2 = arr[:, 0] + arr[:, 2], arr[:, 1] + arr[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(areas)[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= overlap_thresh)[0] + 1]

    return [boxes[k] for k in keep]


def detect_faces(image: np.ndarray, frontal=None, alt2=None, profile=None):
    """
    Detect faces with frontal and profile Haar cascades.

    Returns: (faces_list, elapsed_ms, annotated_image)
    """
    if frontal is None:
        frontal, alt2, profile = _load_cascades()

    t0 = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_flip = cv2.flip(gray, 1)
    width = image.shape[1]
    all_faces = []

    detections = [
        frontal.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(40, 40),
            maxSize=(600, 600),
            flags=cv2.CASCADE_SCALE_IMAGE,
        ),
        alt2.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(40, 40),
            maxSize=(600, 600),
        ),
        profile.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(40, 40),
        ),
    ]

    for det in detections:
        if len(det):
            all_faces.extend(det.tolist())

    det = profile.detectMultiScale(
        gray_flip,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(40, 40),
    )
    if len(det):
        for x, y, w, h in det.tolist():
            all_faces.append([width - x - w, y, w, h])

    faces = _nms_boxes(all_faces, overlap_thresh=0.35)
    elapsed_ms = (time.time() - t0) * 1000
    annotated = _annotate_faces(image, faces)

    return faces, elapsed_ms, annotated


def _annotate_faces(image: np.ndarray, faces):
    annotated = image.copy()
    for idx, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 230, 80), 2)
        label = f"Face {idx + 1}"
        label_w = len(label) * 9
        cv2.rectangle(annotated, (x, y - 20), (x + label_w, y),
                      (0, 230, 80), -1)
        cv2.putText(annotated, label, (x + 3, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0),
                    1, cv2.LINE_AA)
    return annotated


def run_accuracy_experiment(image: np.ndarray):
    """
    Baseline vs processed face detection experiment.

    Haar cascades do not provide a ground-truth accuracy score by default, so
    this report uses success rate based on detected face count relative to the
    best count found by either run.
    """
    frontal, alt2, profile = _load_cascades()

    baseline_faces, baseline_time, baseline_annotated = detect_faces(
        image, frontal, alt2, profile)

    t0 = time.time()
    processed = histogram_equalization(image)
    processing_ms = (time.time() - t0) * 1000

    processed_faces, processed_time, processed_annotated = detect_faces(
        processed, frontal, alt2, profile)

    best_count = max(len(baseline_faces), len(processed_faces), 1)
    baseline_accuracy = (len(baseline_faces) / best_count) * 100
    processed_accuracy = (len(processed_faces) / best_count) * 100

    return {
        "original_image": image,
        "processed_image": processed,
        "original_annotated": baseline_annotated,
        "processed_annotated": processed_annotated,
        "original_faces": len(baseline_faces),
        "processed_faces": len(processed_faces),
        "original_accuracy_pct": round(baseline_accuracy, 1),
        "processed_accuracy_pct": round(processed_accuracy, 1),
        "original_time_ms": round(baseline_time, 2),
        "processed_time_ms": round(processed_time, 2),
        "processing_ms": round(processing_ms, 2),
        "total_processed_ms": round(processing_ms + processed_time, 2),
    }
