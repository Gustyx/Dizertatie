from __future__ import annotations

import tempfile
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

from ..plots.generate_bitplane_plots import _render_bitplane_bar
from ..plots.generate_correlation_plots import _render_correlation_plot
from ..plots.generate_histogram_plots import _render_histogram_pair


def _resolve_related_paths(encrypted_path: Path) -> tuple[Path, Path]:
    if "_enc_" not in encrypted_path.name:
        raise ValueError("Select an encrypted image first.")

    plain_name = encrypted_path.name.split("_enc_", 1)[1]
    dataset_root = encrypted_path.parent.parent

    plain_path = dataset_root / "plain" / plain_name
    plus_one_bit_path = dataset_root / "encrypted_plus_one_bit" / encrypted_path.name

    if not plain_path.exists():
        raise FileNotFoundError(f"Plain image not found: {plain_path}")

    return plain_path, plus_one_bit_path


def _load_preview_image(path: Path, max_size: tuple[int, int]) -> ImageTk.PhotoImage:
    with Image.open(path) as image:
        preview = image.copy()
        preview.thumbnail(max_size)
    return ImageTk.PhotoImage(preview)


def _add_image_panel(
    parent: tk.Widget,
    title: str,
    image_path: Path,
    images: list[ImageTk.PhotoImage],
    *,
    max_size: tuple[int, int] = (1100, 420),
) -> None:
    container = ttk.LabelFrame(parent, text=title, padding=10)
    container.pack(fill="both", expand=True, padx=6, pady=6)

    photo = _load_preview_image(image_path, max_size=max_size)
    images.append(photo)

    label = ttk.Label(container, image=photo)
    label.pack(fill="both", expand=True)


def _build_scrollable_tab(parent: ttk.Notebook) -> tuple[ttk.Frame, ttk.Frame]:
    tab = ttk.Frame(parent)
    tab.pack(fill="both", expand=True)

    canvas = tk.Canvas(tab, highlightthickness=0, borderwidth=0)
    scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
    content = ttk.Frame(canvas)

    content_window = canvas.create_window((0, 0), window=content, anchor="nw")

    def _update_scrollregion(_event=None) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _sync_width(event) -> None:
        canvas.itemconfigure(content_window, width=event.width)

    content.bind("<Configure>", _update_scrollregion)
    canvas.bind("<Configure>", _sync_width)
    canvas.configure(yscrollcommand=scrollbar.set)

    def _on_mousewheel(event) -> str:
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        return "break"

    def _bind_mousewheel(_event=None) -> None:
        tab.bind_all("<MouseWheel>", _on_mousewheel)

    def _unbind_mousewheel(_event=None) -> None:
        tab.unbind_all("<MouseWheel>")

    tab.bind("<Enter>", _bind_mousewheel)
    tab.bind("<Leave>", _unbind_mousewheel)
    content.bind("<Enter>", _bind_mousewheel)
    # content.bind("<Leave>", _unbind_mousewheel)
    canvas.bind("<Enter>", _bind_mousewheel)
    canvas.bind("<Leave>", _unbind_mousewheel)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    return tab, content


def open_plots_window(
    encrypted_image_path: Path | None,
    parent: tk.Misc | None = None,
) -> tk.Toplevel | None:
    if encrypted_image_path is None:
        messagebox.showwarning("No image", "Please choose and encrypt an image first.")
        return None

    try:
        plain_path, plus_one_bit_path = _resolve_related_paths(encrypted_image_path)
    except Exception as exc:
        messagebox.showerror("Plots unavailable", str(exc))
        return None

    if parent is None:
        parent = tk._get_default_root()
    if parent is None:
        messagebox.showerror("Plots unavailable", "No active Tk window found.")
        return None

    window = tk.Toplevel(parent)
    window.title("Image plots")
    window.geometry("1200x900")
    window.minsize(900, 700)

    temp_dir = tempfile.TemporaryDirectory(prefix="image_plot_ui_")
    window._temp_dir = temp_dir  # type: ignore[attr-defined]
    window._plot_images = []  # type: ignore[attr-defined]

    def _cleanup() -> None:
        try:
            temp_dir.cleanup()
        except Exception:
            pass
        try:
            window.destroy()
        except Exception:
            pass

    window.protocol("WM_DELETE_WINDOW", _cleanup)

    root_frame = ttk.Frame(window, padding=12)
    root_frame.pack(fill="both", expand=True)

    header = ttk.Label(
        root_frame,
        text=f"Plots for {encrypted_image_path.name}",
        font=(None, 12, "bold"),
    )
    header.pack(anchor="w", pady=(0, 8))

    notebook = ttk.Notebook(root_frame)
    notebook.pack(fill="both", expand=True)

    correlation_tab, correlation_content = _build_scrollable_tab(notebook)
    histogram_tab, histogram_content = _build_scrollable_tab(notebook)
    bitplane_tab, bitplane_content = _build_scrollable_tab(notebook)

    notebook.add(correlation_tab, text="Correlation")
    notebook.add(histogram_tab, text="Histogram")
    notebook.add(bitplane_tab, text="Bitplane")

    plot_images: list[ImageTk.PhotoImage] = window._plot_images  # type: ignore[attr-defined]

    correlation_path = Path(temp_dir.name) / "correlation_plain_vs_encrypted.png"
    _render_correlation_plot(
        plain_path,
        encrypted_image_path,
        correlation_path,
        title="Plain vs Encrypted correlation",
        max_points=12000,
    )
    _add_image_panel(
        correlation_content,
        "Plain vs encrypted",
        correlation_path,
        plot_images,
    )

    if plus_one_bit_path.exists():
        correlation_plus_path = (
            Path(temp_dir.name) / "correlation_encrypted_vs_plus_one_bit.png"
        )
        _render_correlation_plot(
            encrypted_image_path,
            plus_one_bit_path,
            correlation_plus_path,
            title="Encrypted vs Encrypted plus one bit correlation",
            max_points=12000,
        )
        _add_image_panel(
            correlation_content,
            "Encrypted vs encrypted + one bit",
            correlation_plus_path,
            plot_images,
        )

    histogram_path = Path(temp_dir.name) / "histogram_plain_vs_encrypted.png"
    _render_histogram_pair(
        plain_path,
        encrypted_image_path,
        histogram_path,
        normalize=False,
    )
    _add_image_panel(
        histogram_content,
        "Histogram comparison",
        histogram_path,
        plot_images,
        max_size=(1100, 520),
    )

    bitplane_path = Path(temp_dir.name) / "bitplane_plain_vs_encrypted.png"
    _render_bitplane_bar(plain_path, encrypted_image_path, bitplane_path)
    _add_image_panel(
        bitplane_content,
        "Bitplane comparison",
        bitplane_path,
        plot_images,
        max_size=(1100, 520),
    )

    if plus_one_bit_path.exists():
        bitplane_plus_path = Path(temp_dir.name) / "bitplane_encrypted_vs_plus_one.png"
        _render_bitplane_bar(
            encrypted_image_path, plus_one_bit_path, bitplane_plus_path
        )
        _add_image_panel(
            bitplane_content,
            "Encrypted vs encrypted + one bit",
            bitplane_plus_path,
            plot_images,
            max_size=(1100, 520),
        )

    return window
