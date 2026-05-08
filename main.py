"""
main.py
-------
Vision Editor – Main Application Entry Point.
Run:  python main.py

Layout (mirrors the reference UI screenshots):
┌─────────────────────────────────────────────────────────┐
│  HEADER BAR                                             │
├──────────┬──────────────────────┬──────────────────────┤
│  LEFT    │   INPUT IMAGE        │   OUTPUT IMAGE        │
│  PANEL   │   + Histogram        │   + Histogram         │
│  (tools) │                      │                       │
│          ├──────────────────────┴──────────────────────┤
│  LIGHT   │  FILTER BAR                                  │
│  COLOR   │  (6 filter buttons + kernel size slider)     │
└──────────┴──────────────────────────────────────────────┘
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk

from image_processor import (
    apply_filter,
    adjust_brightness, adjust_contrast, adjust_saturation,
    adjust_warmth,
    rotate_image, zoom_image,
    histogram_equalization, gamma_correction,
)
from histogram_panel import HistogramPanel
from cv_accuracy_window import CVAccuracyWindow


# ──────────────────────────────────────────────────────────────────────────────
#  CONSTANTS / THEME
# ──────────────────────────────────────────────────────────────────────────────

BG       = "#0d0d1a"
PANEL_BG = "#13132a"
SIDE_BG  = "#10101f"
SIDE_EDGE = "#24244a"
CARD_BG  = "#17172f"
TRACK_BG = "#25254a"
ACCENT   = "#00d4ff"
ACCENT2  = "#7c4dff"
TEXT_PRI = "#e0e0f0"
TEXT_DIM = "#6060a0"
BTN_NORM = "#1e1e3a"
BTN_HOV  = "#2a2a50"
FONT_HDR = ("Courier New", 10, "bold")
FONT_LBL = ("Courier New", 9)
FONT_BTN = ("Courier New", 9, "bold")

FILTERS  = ["Laplacian", "Sobel", "Averaging", "Median", "Gaussian", "Bilateral"]

LIGHT_SLIDERS = [
    ("Brightness", -100, 100),
    ("Contrast",   -100, 100),
]
COLOR_SLIDERS = [
    ("Saturation", -100, 100),
    ("Warmth",     -100, 100),
]


# ──────────────────────────────────────────────────────────────────────────────
#  HELPER
# ──────────────────────────────────────────────────────────────────────────────

def _fit(img_bgr: np.ndarray, max_w: int, max_h: int) -> ImageTk.PhotoImage:
    h, w = img_bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN APP CLASS
# ──────────────────────────────────────────────────────────────────────────────

class VisionEditor(tk.Tk):

    START_W = 980
    START_H = 680
    IMG_W = 340
    IMG_H = 240

    def __init__(self):
        super().__init__()
        self.title("Vision Editor — Advanced Image Enhancement & Analysis")
        self.configure(bg=BG)
        self.geometry(f"{self.START_W}x{self.START_H}")
        self.minsize(880, 600)
        self.resizable(True, True)

        # ── State ──────────────────────────────────────────────────────
        self._original_image: np.ndarray | None = None   # never modified
        self._working_image:  np.ndarray | None = None   # after light/color
        self._output_image:   np.ndarray | None = None   # after filter

        self._active_filter: str | None  = None
        self._filter_btns: dict[str, tk.Button] = {}
        self._photo_refs  = []                           # prevent GC

        self._slider_vars: dict[str, tk.DoubleVar] = {}

        # ── Build ──────────────────────────────────────────────────────
        self._build_header()
        self._build_body()
        self._build_filter_bar()

    # ──────────────────────────────────────────────────────────────────
    #  HEADER
    # ──────────────────────────────────────────────────────────────────

    def _build_header(self):
        hdr = tk.Frame(self, bg="#090918", pady=6)
        hdr.pack(fill=tk.X)

        tk.Label(hdr, text="◈  VISION EDITOR",
                 bg="#090918", fg=ACCENT,
                 font=("Courier New", 15, "bold")).pack(side=tk.LEFT, padx=18)

        tk.Button(hdr, text="📂  Upload Image",
                  command=self._upload_image,
                  bg=ACCENT2, fg=TEXT_PRI, font=FONT_BTN,
                  relief=tk.FLAT, padx=12, pady=4,
                  cursor="hand2").pack(side=tk.LEFT, padx=6)

        tk.Button(hdr, text="💾  Save Output",
                  command=self._save_output,
                  bg=BTN_NORM, fg=TEXT_PRI, font=FONT_BTN,
                  relief=tk.FLAT, padx=12, pady=4,
                  cursor="hand2").pack(side=tk.LEFT, padx=6)

        tk.Button(hdr, text="🔁  Reset",
                  command=self._reset,
                  bg="#1e3a1e", fg="#80ff80", font=FONT_BTN,
                  relief=tk.FLAT, padx=12, pady=4,
                  cursor="hand2").pack(side=tk.LEFT, padx=6)

        tk.Button(hdr, text="⚡  CV Accuracy Challenge",
                  command=self._open_cv_window,
                  bg="#3a1e1e", fg="#ff8080", font=FONT_BTN,
                  relief=tk.FLAT, padx=12, pady=4,
                  cursor="hand2").pack(side=tk.RIGHT, padx=18)

    # ──────────────────────────────────────────────────────────────────
    #  BODY  (side panel + two image panels)
    # ──────────────────────────────────────────────────────────────────

    def _build_body(self):
        body = tk.Frame(self, bg=BG)
        body.pack(fill=tk.BOTH, expand=True)

        # Left side panel (light + color sliders)
        self._build_side_panel(body)

        # Two image columns
        img_area = tk.Frame(body, bg=BG)
        img_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._in_panel  = self._build_image_column(img_area, "INPUT")
        self._out_panel = self._build_image_column(img_area, "OUTPUT")

    def _build_side_panel(self, parent):
        side = tk.Frame(parent, bg=SIDE_BG, width=250,
                        highlightthickness=1, highlightbackground=SIDE_EDGE)
        side.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0), pady=8)
        side.pack_propagate(False)

        head = tk.Frame(side, bg=SIDE_BG)
        head.pack(fill=tk.X, padx=14, pady=(12, 8))
        tk.Label(head, text="ADJUSTMENTS", bg=SIDE_BG, fg=ACCENT,
                 font=("Courier New", 12, "bold")).pack(anchor="w")
        tk.Label(head, text="Tone, color, geometry", bg=SIDE_BG, fg=TEXT_DIM,
                 font=("Courier New", 8)).pack(anchor="w", pady=(1, 0))

        canvas = tk.Canvas(side, bg=SIDE_BG, highlightthickness=0)
        scroll = tk.Scrollbar(side, orient="vertical", command=canvas.yview,
                              bg=SIDE_BG, troughcolor=SIDE_BG)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = tk.Frame(canvas, bg=SIDE_BG)
        inner_window = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(
            inner_window, width=e.width))

        def _wheel(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", _wheel))
        canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))

        # LIGHT section
        self._make_section(inner, "LIGHT", LIGHT_SLIDERS)
        # COLOR section
        self._make_section(inner, "COLOR", COLOR_SLIDERS)

        # Geometric tools
        geo_card = self._make_sidebar_card(inner, "GEOMETRY", "Rotate and zoom")

        # Rotation slider
        self._rot_var = tk.DoubleVar(value=0)
        self._make_slider_row(geo_card, "Rotation°", self._rot_var,
                              -180, 180, self._on_rotation)

        # Zoom slider
        self._zoom_var = tk.DoubleVar(value=1.0)
        self._make_slider_row(geo_card, "Zoom", self._zoom_var,
                              0.1, 3.0, self._on_zoom, resolution=0.05)

        # Zoom method
        zm_row = tk.Frame(geo_card, bg=CARD_BG)
        zm_row.pack(fill=tk.X, padx=10, pady=(6, 2))
        tk.Label(zm_row, text="Method", bg=CARD_BG, fg=TEXT_DIM,
                 font=FONT_LBL).pack(anchor="w")
        method_row = tk.Frame(zm_row, bg=CARD_BG)
        method_row.pack(fill=tk.X, pady=(4, 0))
        self._zoom_method = tk.StringVar(value="Bilinear")
        for m in ("Bilinear", "Nearest"):
            tk.Radiobutton(method_row, text=m, variable=self._zoom_method, value=m,
                           indicatoron=False, bg=BTN_NORM, fg=TEXT_PRI,
                           selectcolor=ACCENT2, activebackground=BTN_HOV,
                           activeforeground=TEXT_PRI, font=FONT_LBL,
                           relief=tk.FLAT, padx=8, pady=3,
                           command=self._on_zoom).pack(side=tk.LEFT, fill=tk.X,
                                                       expand=True, padx=(0, 4))

        # Enhancement shortcuts
        enh_card = self._make_sidebar_card(inner, "ENHANCE", "One-step corrections")

        self._make_sidebar_button(enh_card, "Histogram Equalization",
                                  self._hist_eq)

        # Gamma slider
        self._gamma_var = tk.DoubleVar(value=1.0)
        self._make_slider_row(enh_card, "Gamma", self._gamma_var,
                              0.1, 4.0, self._on_gamma, resolution=0.05)

    def _make_sidebar_card(self, parent, title: str, subtitle: str):
        card = tk.Frame(parent, bg=CARD_BG, highlightthickness=1,
                        highlightbackground=SIDE_EDGE)
        card.pack(fill=tk.X, padx=10, pady=(0, 10))

        head = tk.Frame(card, bg=CARD_BG)
        head.pack(fill=tk.X, padx=10, pady=(9, 5))
        tk.Frame(head, bg=ACCENT, width=3, height=24).pack(side=tk.LEFT,
                                                           padx=(0, 8))
        copy = tk.Frame(head, bg=CARD_BG)
        copy.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(copy, text=title, bg=CARD_BG, fg=TEXT_PRI,
                 font=FONT_HDR).pack(anchor="w")
        tk.Label(copy, text=subtitle, bg=CARD_BG, fg=TEXT_DIM,
                 font=("Courier New", 8)).pack(anchor="w")
        return card

    def _make_sidebar_button(self, parent, text: str, command):
        btn = tk.Button(parent, text=text, command=command,
                        bg=BTN_NORM, fg=TEXT_PRI, activebackground=BTN_HOV,
                        activeforeground=TEXT_PRI, font=FONT_LBL,
                        relief=tk.FLAT, padx=8, pady=5, cursor="hand2")
        btn.pack(fill=tk.X, padx=10, pady=(2, 8))
        return btn

    def _make_section(self, parent, title: str, slider_defs: list):
        subtitles = {
            "LIGHT": "Balance image luminance",
            "COLOR": "Tune saturation and warmth",
        }
        card = self._make_sidebar_card(parent, title, subtitles.get(title, ""))
        for name, lo, hi in slider_defs:
            var = tk.DoubleVar(value=0)
            self._slider_vars[name] = var
            self._make_slider_row(card, name, var, lo, hi,
                                  lambda v, n=name: self._on_light_color(n))

    def _make_slider_row(self, parent, label: str, var: tk.DoubleVar,
                         lo: float, hi: float, command,
                         resolution: float = 1.0):
        row = tk.Frame(parent, bg=CARD_BG)
        row.pack(fill=tk.X, padx=10, pady=(2, 7))

        top = tk.Frame(row, bg=CARD_BG)
        top.pack(fill=tk.X)
        tk.Label(top, text=label, bg=CARD_BG, fg=TEXT_DIM,
                 font=FONT_LBL).pack(side=tk.LEFT)

        display_var = tk.StringVar()

        def _format_value(*_):
            value = var.get()
            display_var.set(f"{value:.0f}" if resolution >= 1 else f"{value:.2f}")

        var.trace_add("write", _format_value)
        _format_value()
        val_lbl = tk.Label(top, textvariable=display_var, bg=BTN_NORM,
                           fg=TEXT_PRI, font=("Courier New", 8, "bold"),
                           width=6, padx=3, pady=1)
        val_lbl.pack(side=tk.RIGHT)

        sl = tk.Scale(row, variable=var, from_=lo, to=hi,
                      orient=tk.HORIZONTAL, resolution=resolution,
                      bg=CARD_BG, fg=TEXT_PRI, troughcolor=TRACK_BG,
                      activebackground=ACCENT, sliderrelief=tk.FLAT,
                      highlightthickness=0, showvalue=False,
                      command=lambda v, cmd=command: cmd(v))
        sl.pack(fill=tk.X)

    def _build_image_column(self, parent, label: str) -> dict:
        col = tk.Frame(parent, bg=PANEL_BG, bd=0)
        col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)

        tk.Label(col, text=f"◆ {label}", bg=PANEL_BG, fg=ACCENT,
                 font=FONT_HDR, pady=4).pack()

        placeholder = tk.PhotoImage(width=self.IMG_W, height=self.IMG_H)
        self._photo_refs.append(placeholder)
        img_lbl = tk.Label(col, bg="#08080f", image=placeholder,
                           width=self.IMG_W, height=self.IMG_H)
        img_lbl.pack(padx=6, pady=4)

        hist = HistogramPanel(col,
                              title=f"{label} Histogram",
                              height=120)
        hist.pack(fill=tk.X, padx=6, pady=(0, 6))

        return {"img_lbl": img_lbl, "hist": hist}

    # ──────────────────────────────────────────────────────────────────
    #  FILTER BAR  (bottom)
    # ──────────────────────────────────────────────────────────────────

    def _build_filter_bar(self):
        bar = tk.Frame(self, bg="#0a0a18", pady=8)
        bar.pack(fill=tk.X, side=tk.BOTTOM)

        tk.Label(bar, text="FILTERS:", bg="#0a0a18", fg=ACCENT,
                 font=FONT_HDR).pack(side=tk.LEFT, padx=14)

        for name in FILTERS:
            btn = tk.Button(bar, text=name,
                            command=lambda n=name: self._apply_filter(n),
                            bg=BTN_NORM, fg=TEXT_PRI, font=FONT_BTN,
                            relief=tk.FLAT, padx=12, pady=5,
                            cursor="hand2")
            btn.pack(side=tk.LEFT, padx=4)
            self._filter_btns[name] = btn

        # Kernel size
        tk.Label(bar, text="  Kernel:", bg="#0a0a18", fg=TEXT_DIM,
                 font=FONT_LBL).pack(side=tk.LEFT, padx=(16, 2))

        self._kernel_var = tk.IntVar(value=5)
        kernel_sl = tk.Scale(bar, variable=self._kernel_var, from_=3, to=31,
                             orient=tk.HORIZONTAL, resolution=2,
                             bg="#0a0a18", fg=TEXT_PRI,
                             troughcolor="#222244", highlightthickness=0,
                             showvalue=True, length=120,
                             command=lambda v: self._reapply_filter())
        kernel_sl.pack(side=tk.LEFT)

        # Clear filter
        tk.Button(bar, text="✕ Clear Filter",
                  command=self._clear_filter,
                  bg="#2a1a1a", fg="#ff8080", font=FONT_BTN,
                  relief=tk.FLAT, padx=10, pady=5,
                  cursor="hand2").pack(side=tk.RIGHT, padx=14)

    # ──────────────────────────────────────────────────────────────────
    #  IMAGE DISPLAY
    # ──────────────────────────────────────────────────────────────────

    def _show_input(self, img: np.ndarray):
        if img is None:
            return
        ph = _fit(img, self.IMG_W, self.IMG_H)
        self._photo_refs.append(ph)
        self._in_panel["img_lbl"].configure(image=ph)
        self._in_panel["hist"].update(img)

    def _show_output(self, img: np.ndarray):
        if img is None:
            return
        ph = _fit(img, self.IMG_W, self.IMG_H)
        self._photo_refs.append(ph)
        self._out_panel["img_lbl"].configure(image=ph)
        self._out_panel["hist"].update(img)

    # ──────────────────────────────────────────────────────────────────
    #  UPLOAD / SAVE / RESET
    # ──────────────────────────────────────────────────────────────────

    def _upload_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                       ("All files", "*.*")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"Cannot read: {path}")
            return
        self._original_image = img.copy()
        self._working_image  = img.copy()
        self._output_image   = img.copy()
        self._active_filter  = None
        self._reset_sliders()
        self._show_input(self._original_image)
        self._show_output(self._working_image)

    def _save_output(self):
        if self._output_image is None:
            messagebox.showwarning("No Output", "Process an image first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")])
        if not path:
            return
        cv2.imwrite(path, self._output_image)
        messagebox.showinfo("Saved", f"Output saved to:\n{path}")

    def _reset(self):
        if self._original_image is None:
            return
        self._reset_sliders()
        self._active_filter = None
        self._working_image = self._original_image.copy()
        self._output_image  = self._original_image.copy()
        for btn in self._filter_btns.values():
            btn.configure(bg=BTN_NORM, fg=TEXT_PRI)
        self._show_input(self._original_image)
        self._show_output(self._original_image)

    def _reset_sliders(self):
        for var in self._slider_vars.values():
            var.set(0)
        self._rot_var.set(0)
        self._zoom_var.set(1.0)
        self._gamma_var.set(1.0)

    # ──────────────────────────────────────────────────────────────────
    #  LIGHT / COLOR SLIDERS
    # ──────────────────────────────────────────────────────────────────

    def _on_light_color(self, changed_name):
        """Rebuild working image from original applying all slider values."""
        if self._original_image is None:
            return
        img = self._original_image.copy()
        sv  = self._slider_vars

        img = adjust_brightness(img,  sv["Brightness"].get())
        img = adjust_contrast(img,    sv["Contrast"].get())
        img = adjust_saturation(img,  sv["Saturation"].get())
        img = adjust_warmth(img,      sv["Warmth"].get())

        self._working_image = img
        # reapply active filter on top
        self._reapply_filter()

    # ──────────────────────────────────────────────────────────────────
    #  GEOMETRIC
    # ──────────────────────────────────────────────────────────────────

    def _on_rotation(self, _=None):
        if self._working_image is None:
            return
        angle = self._rot_var.get()
        rotated = rotate_image(self._working_image, angle)
        self._output_image = rotated
        self._show_output(rotated)

    def _on_zoom(self, _=None):
        if self._working_image is None:
            return
        scale  = self._zoom_var.get()
        method = self._zoom_method.get()
        zoomed = zoom_image(self._working_image, scale, method)
        self._output_image = zoomed
        self._show_output(zoomed)

    # ──────────────────────────────────────────────────────────────────
    #  ENHANCEMENT
    # ──────────────────────────────────────────────────────────────────

    def _hist_eq(self):
        if self._working_image is None:
            return
        result = histogram_equalization(self._working_image)
        self._output_image = result
        self._show_output(result)

    def _on_gamma(self, _=None):
        if self._working_image is None:
            return
        g      = self._gamma_var.get()
        result = gamma_correction(self._working_image, g)
        self._output_image = result
        self._show_output(result)

    # ──────────────────────────────────────────────────────────────────
    #  FILTERS
    # ──────────────────────────────────────────────────────────────────

    def _apply_filter(self, name: str):
        if self._working_image is None:
            return
        self._active_filter = name
        # Highlight active button
        for n, btn in self._filter_btns.items():
            if n == name:
                btn.configure(bg=ACCENT, fg="#000000")
            else:
                btn.configure(bg=BTN_NORM, fg=TEXT_PRI)
        self._reapply_filter()

    def _reapply_filter(self):
        if self._working_image is None:
            return
        if self._active_filter is None:
            self._output_image = self._working_image.copy()
            self._show_output(self._output_image)
            return
        ksize  = self._kernel_var.get()
        result = apply_filter(self._working_image, self._active_filter, ksize)
        self._output_image = result
        self._show_output(result)

    def _clear_filter(self):
        self._active_filter = None
        for btn in self._filter_btns.values():
            btn.configure(bg=BTN_NORM, fg=TEXT_PRI)
        if self._working_image is not None:
            self._output_image = self._working_image.copy()
            self._show_output(self._output_image)

    # ──────────────────────────────────────────────────────────────────
    #  CV ACCURACY WINDOW
    # ──────────────────────────────────────────────────────────────────

    def _open_cv_window(self):
        if self._original_image is None:
            messagebox.showwarning("No Image", "Upload an image first.")
            return
        CVAccuracyWindow(self, self._original_image)


# ──────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = VisionEditor()
    app.mainloop()
