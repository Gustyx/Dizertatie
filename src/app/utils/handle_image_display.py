from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tkinter as tk
from tkinter import filedialog
from contextlib import contextmanager

from PIL import Image, ImageTk

try:
    from tkinterdnd2 import DND_FILES

    DND_AVAILABLE = True
except Exception:
    DND_AVAILABLE = False
    DND_FILES = None


IMAGE_FILETYPES = [("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.gif")]


@contextmanager
def _allow_large_images():
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        yield
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit


@dataclass
class ImageUiState:
    selected_image_path: Path | None = None
    selected_image_tk: Any = None
    action_buttons_frame: tk.Frame | None = None
    result_category_frame: tk.Frame | None = None
    result_frame: tk.LabelFrame | None = None
    result_widget: tk.Text | None = None


def display_image(
    path: Path,
    canvas: tk.Canvas,
    state: ImageUiState,
    max_size: tuple[int, int] = (800, 600),
    extra_window_height: int = 180,
) -> None:
    with _allow_large_images():
        with Image.open(path) as img:
            img.thumbnail(max_size)
            preview = img.copy()

    state.selected_image_tk = ImageTk.PhotoImage(preview)

    canvas.delete("all")
    canvas.config(
        width=state.selected_image_tk.width(), height=state.selected_image_tk.height()
    )
    canvas.create_image(0, 0, anchor="nw", image=state.selected_image_tk)
    state.selected_image_path = path

    if (
        state.action_buttons_frame is not None
        and not state.action_buttons_frame.winfo_ismapped()
    ):
        state.action_buttons_frame.pack(pady=(6, 0))
    if (
        state.result_category_frame is not None
        and not state.result_category_frame.winfo_ismapped()
    ):
        state.result_category_frame.pack(pady=(6, 0))
    if state.result_frame is not None and not state.result_frame.winfo_ismapped():
        state.result_frame.pack(fill="both", expand=True, pady=(8, 0))

    try:
        top = canvas.winfo_toplevel()
        top.update_idletasks()
        new_w = state.selected_image_tk.width() + 40
        new_h = state.selected_image_tk.height() + extra_window_height
        if new_w > top.winfo_width() or new_h > top.winfo_height():
            top.geometry(
                f"{max(new_w, top.winfo_width())}x{max(new_h, top.winfo_height())}"
            )
    except Exception:
        pass


def reset_placeholder(canvas: tk.Canvas, text: str) -> None:
    canvas.delete("all")
    padding = 8
    width = int(canvas.cget("width"))
    height = int(canvas.cget("height"))
    canvas.create_rectangle(
        padding,
        padding,
        width - padding,
        height - padding,
        outline="#3a3d55",
        width=2,
        dash=(4, 4),
        tags=("placeholder",),
    )
    canvas.create_text(
        width // 2,
        height // 2,
        text=text,
        fill="#5a5e7a",
        tags=("placeholder",),
    )


def set_result_text(state: ImageUiState, text: str) -> None:
    if state.result_widget is None:
        return

    state.result_widget.config(state="normal")
    state.result_widget.delete("1.0", tk.END)
    state.result_widget.insert("1.0", text)
    state.result_widget.config(state="disabled")
    state.result_widget.yview_moveto(0)


def choose_image(
    canvas: tk.Canvas,
    state: ImageUiState,
    *,
    title: str,
    initialdir: str | None,
    max_size: tuple[int, int] = (800, 600),
    extra_window_height: int = 180,
    filetypes=IMAGE_FILETYPES,
) -> None:
    file = filedialog.askopenfilename(
        title=title,
        initialdir=initialdir,
        filetypes=filetypes,
    )
    if not file:
        return
    display_image(
        Path(file),
        canvas,
        state,
        max_size=max_size,
        extra_window_height=extra_window_height,
    )


def on_drop(
    event,
    canvas: tk.Canvas,
    state: ImageUiState,
    *,
    max_size: tuple[int, int] = (800, 600),
    extra_window_height: int = 180,
) -> None:
    try:
        parts = canvas.master.tk.splitlist(event.data)
        if not parts:
            return
        display_image(
            Path(parts[0]),
            canvas,
            state,
            max_size=max_size,
            extra_window_height=extra_window_height,
        )
    except Exception:
        return


def build_image_box(
    frame,
    state: ImageUiState,
    *,
    placeholder_text: str,
    max_size: tuple[int, int] = (800, 600),
    extra_window_height: int = 180,
) -> tk.Canvas:
    img_box = tk.Canvas(
        frame, bg="#0d0f1a", relief="flat", width=400, height=240, highlightthickness=0
    )
    reset_placeholder(img_box, placeholder_text)
    img_box.pack(padx=12, pady=(12, 0), anchor="n")

    if DND_AVAILABLE:
        try:
            img_box.drop_target_register(DND_FILES)
            img_box.dnd_bind(
                "<<Drop>>",
                lambda e: on_drop(
                    e,
                    img_box,
                    state,
                    max_size=max_size,
                    extra_window_height=extra_window_height,
                ),
            )
        except Exception:
            pass

    return img_box
