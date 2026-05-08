"""
histogram_panel.py
------------------
Embeddable Tkinter widget that draws a live BGR histogram using matplotlib.
"""

import tkinter as tk
import numpy as np
import matplotlib
matplotlib.use("Agg")                          # off-screen backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from image_processor import compute_histogram


class HistogramPanel(tk.Frame):
    """A compact matplotlib histogram embedded in a Tkinter frame."""

    COLORS = {"B": "#4FC3F7", "G": "#81C784", "R": "#EF9A9A"}

    def __init__(self, parent, title="Histogram", **kwargs):
        super().__init__(parent, bg="#1a1a2e", **kwargs)
        self.title_text = title

        self.fig, self.ax = plt.subplots(figsize=(3.2, 1.8), facecolor="#1a1a2e")
        self.ax.set_facecolor("#0f0f23")
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.15)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._draw_empty()

    def _style_ax(self):
        self.ax.tick_params(colors="#555577", labelsize=6)
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#333355")
        self.ax.set_title(self.title_text, color="#aaaacc", fontsize=7, pad=3)
        self.ax.set_xlim(0, 255)

    def _draw_empty(self):
        self.ax.clear()
        self.ax.set_facecolor("#0f0f23")
        self.ax.text(0.5, 0.5, "No Image", transform=self.ax.transAxes,
                     ha="center", va="center", color="#444466", fontsize=9)
        self._style_ax()
        self.canvas.draw()

    def update(self, image: np.ndarray):
        if image is None:
            self._draw_empty()
            return
        hists = compute_histogram(image)
        self.ax.clear()
        self.ax.set_facecolor("#0f0f23")
        for ch, color in self.COLORS.items():
            self.ax.plot(hists[ch], color=color, lw=0.8, alpha=0.85)
            self.ax.fill_between(range(256), hists[ch], alpha=0.18, color=color)
        self._style_ax()
        self.canvas.draw()
