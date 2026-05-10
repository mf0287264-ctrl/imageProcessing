"""
cv_accuracy.py
--------------
Task Two: CV Accuracy Challenge for face detection.

The core of this task is to prove that "Garbage In = Garbage Out."
Students run the same OpenCV pre-trained face detector on:
1. A low-quality original image.
2. A processed version improved with Task One histogram equalization.

The comparison reports face-only detections, ground-truth count accuracy,
detector confidence, inference speed, processing delay, and total processed time.
"""

import time

import cv2
import numpy as np

from image_processor import histogram_equalization


def _load_cascades():
    """Load frontal face and eye cascades."""
    frontal = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    alt2 = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    eye = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
    if frontal.empty() or alt2.empty() or eye.empty():
        raise RuntimeError("Could not load OpenCV Haar cascade files.")
    return frontal, alt2, eye


def _nms_detections(detections, overlap_thresh=0.35):
    """Remove duplicate overlapping bounding boxes."""
    if len(detections) == 0:
        return []

    boxes = [item["box"] for item in detections]
    arr = np.array(boxes, dtype=np.float32)
    x1, y1 = arr[:, 0], arr[:, 1]
    x2, y2 = arr[:, 0] + arr[:, 2], arr[:, 1] + arr[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    scores = np.array([item["confidence"] for item in detections],
                      dtype=np.float32)
    order = np.argsort(scores)[::-1]
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

    return [detections[k] for k in keep]


def _haar_confidence(weight: float) -> float:
    """Map Haar level weight to a stable 0-100 confidence percentage."""
    return float(np.clip(100.0 / (1.0 + np.exp(-0.85 * weight)), 0.0, 100.0))


def _detect_with_weights(cascade, gray, scale_factor, min_neighbors,
                         min_size, max_size):
    boxes, _reject, weights = cascade.detectMultiScale3(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        maxSize=max_size,
        flags=cv2.CASCADE_SCALE_IMAGE,
        outputRejectLevels=True,
    )
    if len(boxes) == 0:
        return []
    return [
        {"box": tuple(map(int, box)), "confidence": _haar_confidence(weight)}
        for box, weight in zip(boxes, weights)
    ]


def _looks_like_face(gray: np.ndarray, box, eye_cascade,
                     confidence: float) -> bool:
    """Reject common false positives such as hands, clothing, and body parts."""
    x, y, w, h = box
    img_h, img_w = gray.shape[:2]
    aspect = w / max(h, 1)

    if not 0.78 <= aspect <= 1.28:
        return False

    min_side = max(32, int(min(img_w, img_h) * 0.08))
    max_side = int(min(img_w, img_h) * 0.62)
    if w < min_side or h < min_side or w > max_side or h > max_side:
        return False

    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
        return False

    roi = gray[y:y + h, x:x + w]
    if roi.size == 0:
        return False

    upper_face = roi[:max(1, int(h * 0.62)), :]
    eyes = eye_cascade.detectMultiScale(
        upper_face,
        scaleFactor=1.08,
        minNeighbors=3,
        minSize=(max(8, w // 8), max(8, h // 10)),
        maxSize=(max(12, w // 2), max(12, h // 3)),
    )

    # Clear eye evidence is the strongest face-only filter. Very high-confidence
    # Haar hits are still allowed because noisy/low-quality inputs can hide eyes.
    return len(eyes) > 0 or confidence >= 92.0


def detect_faces(image: np.ndarray, frontal=None, alt2=None, eye=None):
    """
    Detect frontal faces only with Haar cascades.

    Returns: (detections, elapsed_ms, annotated_image)
    """
    if frontal is None:
        frontal, alt2, eye = _load_cascades()

    t0 = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape[:2]
    min_side = max(32, int(min(img_w, img_h) * 0.08))
    max_side = max(min_side, int(min(img_w, img_h) * 0.62))

    candidates = []
    for cascade in (frontal, alt2):
        candidates.extend(_detect_with_weights(
            cascade,
            gray,
            scale_factor=1.06,
            min_neighbors=6,
            min_size=(min_side, min_side),
            max_size=(max_side, max_side),
        ))

    faces = [
        item for item in candidates
        if _looks_like_face(gray, item["box"], eye, item["confidence"])
    ]
    faces = _nms_detections(faces, overlap_thresh=0.35)
    elapsed_ms = (time.time() - t0) * 1000
    annotated = _annotate_faces(image, faces)

    return faces, elapsed_ms, annotated


def _annotate_faces(image: np.ndarray, faces):
    annotated = image.copy()
    for idx, face in enumerate(faces):
        x, y, w, h = face["box"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 230, 80), 2)
        label = f"Face {idx + 1}: {face['confidence']:.0f}%"
        label_w = max(70, len(label) * 8)
        label_y = max(20, y)
        cv2.rectangle(annotated, (x, label_y - 20), (x + label_w, label_y),
                      (0, 230, 80), -1)
        cv2.putText(annotated, label, (x + 3, label_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0),
                    1, cv2.LINE_AA)
    return annotated


def _count_accuracy(detected_count: int, expected_count: int) -> float:
    expected_count = max(1, int(expected_count))
    error = abs(detected_count - expected_count)
    return round(max(0.0, 100.0 - (error / expected_count) * 100.0), 1)


def run_accuracy_experiment(image: np.ndarray, expected_faces: int = 1):
    """
    Baseline vs processed face detection experiment.

    Accuracy is calculated against the user-entered ground-truth face count.
    Extra false positives and missed faces both reduce the score.
    """
    expected_faces = max(1, int(expected_faces))
    frontal, alt2, eye = _load_cascades()

    baseline_faces, baseline_time, baseline_annotated = detect_faces(
        image, frontal, alt2, eye)

    t0 = time.time()
    processed = histogram_equalization(image)
    processing_ms = (time.time() - t0) * 1000

    processed_faces, processed_time, processed_annotated = detect_faces(
        processed, frontal, alt2, eye)

    baseline_accuracy = _count_accuracy(len(baseline_faces), expected_faces)
    processed_accuracy = _count_accuracy(len(processed_faces), expected_faces)

    return {
        "original_image": image,
        "processed_image": processed,
        "original_annotated": baseline_annotated,
        "processed_annotated": processed_annotated,
        "expected_faces": expected_faces,
        "original_faces": len(baseline_faces),
        "processed_faces": len(processed_faces),
        "original_accuracy_pct": round(baseline_accuracy, 1),
        "processed_accuracy_pct": round(processed_accuracy, 1),
        "original_confidence_pct": _average_confidence(baseline_faces),
        "processed_confidence_pct": _average_confidence(processed_faces),
        "original_time_ms": round(baseline_time, 2),
        "processed_time_ms": round(processed_time, 2),
        "processing_ms": round(processing_ms, 2),
        "total_processed_ms": round(processing_ms + processed_time, 2),
    }


def _average_confidence(faces) -> float:
    if not faces:
        return 0.0
    return round(
        sum(face["confidence"] for face in faces) / len(faces),
        1,
    )
