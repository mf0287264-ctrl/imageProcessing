"""
cv_accuracy_window.py
---------------------
Pop-up Tkinter window for Task Two: CV Accuracy Challenge.
"""

import threading
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from cv_accuracy import run_accuracy_experiment


def _cv2_to_tk(img_bgr: np.ndarray, max_w=360, max_h=260) -> ImageTk.PhotoImage:
    h, w = img_bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


class CVAccuracyWindow(tk.Toplevel):
    """Task Two popup for baseline vs processed face detection."""

    BG = "#0d0d1a"
    PANEL = "#13132a"
    CARD = "#17172f"
    ACCENT = "#00d4ff"
    GREEN = "#00ff88"
    ORANGE = "#ffb347"
    TEXT = "#e0e0f0"
    DIM = "#6060a0"

    def __init__(self, parent, image: np.ndarray):
        super().__init__(parent)
        self.image = image
        self._refs = []
        self._table = {}
        self.title("Task Two - CV Accuracy Challenge")
        self.configure(bg=self.BG)
        self.resizable(True, False)
        self._build_ui()
        self._run_experiment()

    def _build_ui(self):
        hdr = tk.Frame(self, bg="#090918", pady=12)
        hdr.pack(fill=tk.X)
        tk.Label(
            hdr,
            text="CV Accuracy Challenge",
            bg="#090918",
            fg=self.ACCENT,
            font=("Courier New", 15, "bold"),
        ).pack()
        tk.Label(
            hdr,
            text="Face detection: low-quality original vs histogram-equalized image",
            bg="#090918",
            fg=self.DIM,
            font=("Courier New", 9),
        ).pack(pady=(2, 0))

        img_row = tk.Frame(self, bg=self.BG)
        img_row.pack(fill=tk.X, padx=12, pady=10)

        self._img_labels = []
        self._cap_labels = []
        columns = [
            ("Original Baseline", self.ORANGE),
            ("Processed Result", self.GREEN),
        ]
        for col, (title, color) in enumerate(columns):
            card = tk.Frame(
                img_row,
                bg=self.CARD,
                highlightthickness=1,
                highlightbackground="#24244a",
            )
            card.grid(row=0, column=col, padx=6, sticky="nsew")
            img_row.columnconfigure(col, weight=1)

            tk.Label(
                card,
                text=title,
                bg=self.CARD,
                fg=color,
                font=("Courier New", 10, "bold"),
            ).pack(pady=(10, 6))

            img_label = tk.Label(card, bg="#08080f")
            img_label.pack(padx=8, pady=4)
            self._img_labels.append(img_label)

            cap_label = tk.Label(
                card,
                text="Running...",
                bg=self.CARD,
                fg=self.DIM,
                font=("Courier New", 9),
                justify=tk.CENTER,
            )
            cap_label.pack(pady=(4, 10))
            self._cap_labels.append(cap_label)

        info = tk.Frame(self, bg=self.CARD, padx=14, pady=10,
                        highlightthickness=1,
                        highlightbackground="#24244a")
        info.pack(fill=tk.X, padx=12, pady=(0, 8))
        tk.Label(
            info,
            text="Workflow",
            bg=self.CARD,
            fg=self.ACCENT,
            font=("Courier New", 10, "bold"),
        ).pack(anchor="w")
        tk.Label(
            info,
            text=("1. Run Haar Cascades on the original image   "
                  "2. Improve it with Histogram Equalization   "
                  "3. Run the same detector again"),
            bg=self.CARD,
            fg=self.TEXT,
            font=("Courier New", 8),
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(5, 0))

        table = tk.Frame(self, bg=self.PANEL, padx=14, pady=12)
        table.pack(fill=tk.X, padx=12, pady=(0, 8))
        tk.Label(
            table,
            text="Comparison Table",
            bg=self.PANEL,
            fg=self.ACCENT,
            font=("Courier New", 10, "bold"),
        ).grid(row=0, column=0, columnspan=3, pady=(0, 8))

        headers = ["Metric", "Original", "Processed"]
        colors = [self.TEXT, self.ORANGE, self.GREEN]
        for col, (header, color) in enumerate(zip(headers, colors)):
            tk.Label(
                table,
                text=header,
                bg="#0d0d1a",
                fg=color,
                font=("Courier New", 9, "bold"),
                width=26,
                anchor="center",
                pady=5,
                bd=1,
                relief=tk.GROOVE,
            ).grid(row=1, column=col, sticky="ew")

        metrics = [
            "Faces Detected",
            "Accuracy (%)",
            "Inference Speed (ms)",
            "Processing Delay (ms)",
            "Total Time (ms)",
        ]
        for row, metric in enumerate(metrics, start=2):
            tk.Label(
                table,
                text=metric,
                bg=self.PANEL,
                fg=self.DIM,
                font=("Courier New", 9),
                width=26,
                anchor="w",
                padx=8,
                pady=4,
                bd=1,
                relief=tk.GROOVE,
            ).grid(row=row, column=0, sticky="ew")

            original = tk.Label(
                table,
                text="...",
                bg=self.PANEL,
                fg=self.TEXT,
                font=("Courier New", 9, "bold"),
                width=26,
                anchor="center",
                pady=4,
                bd=1,
                relief=tk.GROOVE,
            )
            original.grid(row=row, column=1, sticky="ew")

            processed = tk.Label(
                table,
                text="...",
                bg=self.PANEL,
                fg=self.TEXT,
                font=("Courier New", 9, "bold"),
                width=26,
                anchor="center",
                pady=4,
                bd=1,
                relief=tk.GROOVE,
            )
            processed.grid(row=row, column=2, sticky="ew")
            self._table[metric] = (original, processed)

        self._conclusion = tk.Label(
            self,
            text="",
            bg=self.BG,
            fg=self.GREEN,
            font=("Courier New", 9, "bold"),
            wraplength=860,
            justify=tk.CENTER,
        )
        self._conclusion.pack(pady=4)

        self._status = tk.Label(
            self,
            text="Running Task Two experiment...",
            bg=self.BG,
            fg="#ffcc00",
            font=("Courier New", 9),
        )
        self._status.pack(pady=(0, 10))

    def _run_experiment(self):
        def worker():
            try:
                result = run_accuracy_experiment(self.image)
                self.after(0, lambda: self._fill_results(result))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror(
                    "Error", str(e), parent=self))

        threading.Thread(target=worker, daemon=True).start()

    def _fill_results(self, result: dict):
        images = [
            result["original_annotated"],
            result["processed_annotated"],
        ]
        captions = [
            (f"Faces: {result['original_faces']} | "
             f"Accuracy: {result['original_accuracy_pct']}%"),
            (f"Faces: {result['processed_faces']} | "
             f"Accuracy: {result['processed_accuracy_pct']}%"),
        ]
        for label, image, caption, cap_label in zip(
                self._img_labels, images, captions, self._cap_labels):
            photo = _cv2_to_tk(image)
            self._refs.append(photo)
            label.configure(image=photo)
            cap_label.configure(text=caption)

        rows = {
            "Faces Detected": (
                str(result["original_faces"]),
                str(result["processed_faces"]),
            ),
            "Accuracy (%)": (
                f"{result['original_accuracy_pct']}%",
                f"{result['processed_accuracy_pct']}%",
            ),
            "Inference Speed (ms)": (
                f"{result['original_time_ms']} ms",
                f"{result['processed_time_ms']} ms",
            ),
            "Processing Delay (ms)": (
                "0 ms",
                f"{result['processing_ms']} ms",
            ),
            "Total Time (ms)": (
                f"{result['original_time_ms']} ms",
                f"{result['total_processed_ms']} ms",
            ),
        }
        for metric, values in rows.items():
            original, processed = self._table[metric]
            original.configure(text=values[0])
            processed.configure(text=values[1])

        improvement = (
            result["processed_accuracy_pct"] - result["original_accuracy_pct"]
        )
        if improvement > 0:
            msg = (f"Processed image improved detection accuracy by "
                   f"{improvement:.1f}%. This supports Garbage In = Garbage Out.")
        elif improvement == 0:
            msg = ("Accuracy stayed the same. The processed image did not change "
                   "the detector result for this input.")
        else:
            msg = ("Processed accuracy was lower on this image. The table still "
                   "shows how input quality changes model output.")
            self._conclusion.configure(fg=self.ORANGE)

        self._conclusion.configure(text=msg)
        self._status.configure(
            text=("Experiment complete | processing delay: "
                  f"{result['processing_ms']} ms"),
            fg=self.GREEN,
        )
