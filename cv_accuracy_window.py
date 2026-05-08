"""
cv_accuracy_window.py
---------------------
Pop-up Tkinter window that runs and displays the CV accuracy experiment (Task 2).
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk
from cv_accuracy import run_accuracy_experiment


def _cv2_to_tk(img_bgr: np.ndarray, max_w=340, max_h=220) -> ImageTk.PhotoImage:
    h, w = img_bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img_bgr, (nw, nh))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


class CVAccuracyWindow(tk.Toplevel):
    """Popup window for Task 2: CV Accuracy Challenge."""

    BG     = "#0d0d1a"
    PANEL  = "#13132a"
    ACCENT = "#00d4ff"
    TEXT   = "#e0e0f0"
    DIM    = "#7070a0"

    def __init__(self, parent, image: np.ndarray):
        super().__init__(parent)
        self.image = image
        self.title("CV Accuracy Challenge — Face Detection")
        self.configure(bg=self.BG)
        self.resizable(False, False)
        self._refs = []          # keep PhotoImage refs alive

        self._build_ui()
        self._run_experiment()

    # ──────────────────────────────────────────────────────────────────
    #  UI BUILD
    # ──────────────────────────────────────────────────────────────────

    def _build_ui(self):
        W = 760
        # ── Header ──
        hdr = tk.Frame(self, bg="#0a0a1a", pady=10)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="⚡  CV Accuracy Challenge",
                 bg="#0a0a1a", fg=self.ACCENT,
                 font=("Courier New", 14, "bold")).pack()
        tk.Label(hdr, text="Proving: Garbage In = Garbage Out",
                 bg="#0a0a1a", fg=self.DIM,
                 font=("Courier New", 9)).pack()

        # ── Image rows ──
        img_frame = tk.Frame(self, bg=self.BG)
        img_frame.pack(fill=tk.X, padx=15, pady=10)

        labels = ["Original", "Degraded (Baseline)", "Enhanced (Post-Process)"]
        self._img_labels = []
        self._cap_labels = []
        for i, lbl in enumerate(labels):
            col = tk.Frame(img_frame, bg=self.PANEL, bd=1, relief=tk.FLAT,
                           padx=6, pady=6)
            col.grid(row=0, column=i, padx=8)
            tk.Label(col, text=lbl, bg=self.PANEL, fg=self.ACCENT,
                     font=("Courier New", 8, "bold")).pack()
            il = tk.Label(col, bg=self.PANEL, width=340, height=220)
            il.pack()
            self._img_labels.append(il)
            cl = tk.Label(col, text="—", bg=self.PANEL, fg=self.DIM,
                          font=("Courier New", 8))
            cl.pack()
            self._cap_labels.append(cl)

        # ── Results table ──
        tbl_frame = tk.Frame(self, bg=self.PANEL, padx=20, pady=12)
        tbl_frame.pack(fill=tk.X, padx=15, pady=5)
        tk.Label(tbl_frame, text="📊  Comparison Table",
                 bg=self.PANEL, fg=self.ACCENT,
                 font=("Courier New", 10, "bold")).grid(row=0, columnspan=3,
                                                         pady=(0, 8))

        headers = ["Metric", "Degraded (Original)", "Enhanced (Processed)"]
        for c, h in enumerate(headers):
            tk.Label(tbl_frame, text=h, bg=self.PANEL, fg=self.TEXT,
                     font=("Courier New", 9, "bold"),
                     width=24, anchor="center",
                     bd=1, relief=tk.GROOVE).grid(row=1, column=c, ipady=4)

        self._table_rows = {}
        metrics = ["Faces Detected", "Accuracy (%)", "Inference Speed (ms)"]
        for r, m in enumerate(metrics, start=2):
            tk.Label(tbl_frame, text=m, bg=self.PANEL, fg=self.DIM,
                     font=("Courier New", 9), width=24,
                     bd=1, relief=tk.GROOVE).grid(row=r, column=0, ipady=3)
            orig_lbl = tk.Label(tbl_frame, text="…", bg=self.PANEL, fg=self.TEXT,
                                font=("Courier New", 9), width=24,
                                bd=1, relief=tk.GROOVE)
            orig_lbl.grid(row=r, column=1, ipady=3)
            enh_lbl = tk.Label(tbl_frame, text="…", bg=self.PANEL, fg=self.TEXT,
                               font=("Courier New", 9), width=24,
                               bd=1, relief=tk.GROOVE)
            enh_lbl.grid(row=r, column=2, ipady=3)
            self._table_rows[m] = (orig_lbl, enh_lbl)

        # ── Status ──
        self._status = tk.Label(self, text="Running experiment…",
                                bg=self.BG, fg="#ffcc00",
                                font=("Courier New", 9))
        self._status.pack(pady=8)

    # ──────────────────────────────────────────────────────────────────
    #  EXPERIMENT
    # ──────────────────────────────────────────────────────────────────

    def _run_experiment(self):
        def worker():
            try:
                results = run_accuracy_experiment(self.image)
                self.after(0, lambda: self._display_results(results))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e),
                                                            parent=self))

        threading.Thread(target=worker, daemon=True).start()

    def _display_results(self, r: dict):
        # Images
        imgs = [r["original_image"], r["degraded_image"], r["enh_annotated"]]
        caps = [
            f"Ref faces: {r['ref_faces']}",
            f"Detected: {r['orig_faces']}  |  Acc: {r['orig_accuracy_pct']}%",
            f"Detected: {r['enh_faces']}   |  Acc: {r['enh_accuracy_pct']}%",
        ]
        for il, img, cap, cl in zip(self._img_labels, imgs, caps, self._cap_labels):
            ph = _cv2_to_tk(img)
            self._refs.append(ph)
            il.configure(image=ph, width=ph.width(), height=ph.height())
            cl.configure(text=cap)

        # Table
        rows_data = {
            "Faces Detected":      (str(r["orig_faces"]),      str(r["enh_faces"])),
            "Accuracy (%)":        (f"{r['orig_accuracy_pct']}%", f"{r['enh_accuracy_pct']}%"),
            "Inference Speed (ms)":(f"{r['orig_time_ms']} ms", f"{r['enh_time_ms']} ms"),
        }
        for metric, (ov, ev) in rows_data.items():
            orig_lbl, enh_lbl = self._table_rows[metric]
            orig_lbl.configure(text=ov)
            enh_lbl.configure(text=ev,
                               fg="#00ff99" if ev >= ov else self.TEXT)

        self._status.configure(
            text=f"✓  Experiment complete — overhead: {r['overhead_ms']} ms",
            fg="#00ff88")
