from __future__ import annotations

import tempfile
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

from ..plots import (
    render_bitplane_bars_all_algorithms,
    render_histogram_all_algorithms,
    render_entropy_bars,
    render_correlation_bars,
    render_scatter_comparison,
)


def _resolve_plain_path(encrypted_path: Path) -> tuple[Path, Path]:
    if "_enc_" not in encrypted_path.name:
        raise ValueError("Select an encrypted image first.")

    plain_name = encrypted_path.name.split("_enc_", 1)[1]
    dataset_root = encrypted_path.parent.parent

    plain_path = dataset_root / "plain" / plain_name

    if not plain_path.exists():
        raise FileNotFoundError(f"Plain image not found: {plain_path}")

    return plain_path


def _open_image_viewer(image_path: Path, title: str, images: list[ImageTk.PhotoImage]) -> None:
    """Open a standalone resizable window — image scales as you drag the corner."""
    with Image.open(image_path) as _im:
        source = _im.copy()

    win = tk.Toplevel()
    win.title(title)
    orig_w, orig_h = source.size
    win.geometry(f"{min(orig_w, 1400)}x{min(orig_h, 900)}")

    label = ttk.Label(win)
    label.pack(fill="both", expand=True)

    _after_id: list[str] = []

    def _on_resize(_event=None) -> None:
        if _after_id:
            try:
                label.after_cancel(_after_id[0])
            except Exception:
                pass
            _after_id.clear()
        _after_id.append(label.after(60, _do_resize))

    def _do_resize() -> None:
        w = win.winfo_width()
        h = win.winfo_height()
        if w < 10 or h < 10:
            return
        resized = source.copy()
        resized.thumbnail((w, h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(resized)
        images.append(photo)
        label.configure(image=photo)

    win.bind("<Configure>", _on_resize)
    win.after(50, _do_resize)


def _add_image_panel(
    parent: tk.Widget,
    title: str,
    image_path: Path,
    images: list[ImageTk.PhotoImage],
    *,
    max_size: tuple[int, int] = (1100, 420),
) -> None:
    with Image.open(image_path) as _im:
        source = _im.copy()

    init = source.copy()
    init.thumbnail(max_size, Image.LANCZOS)

    photo = ImageTk.PhotoImage(init)
    images.append(photo)

    container = ttk.LabelFrame(parent, text=title, padding=6)
    container.pack(fill="x", padx=6, pady=6)

    label = ttk.Label(container, image=photo, cursor="hand2")
    label.pack(anchor="nw")

    label.bind("<Button-1>", lambda _e: _open_image_viewer(image_path, title, images))


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
    canvas.bind("<Enter>", _bind_mousewheel)
    canvas.bind("<Leave>", _unbind_mousewheel)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    return tab, content


def _find_all_encrypted_variants(encrypted_path: Path) -> dict[str, Path]:
    """Scan the encrypted/ folder for every algorithm variant of the same plain image."""
    _, _, plain_name = encrypted_path.name.partition("_enc_")
    plain_stem = Path(plain_name).stem
    encrypted_dir = encrypted_path.parent

    result: dict[str, Path] = {}
    for candidate in sorted(encrypted_dir.iterdir()):
        if "_enc_" not in candidate.name:
            continue
        _, _, cand_plain = candidate.name.partition("_enc_")
        if Path(cand_plain).stem != plain_stem:
            continue
        alg_prefix = candidate.name.split("_enc_")[0]
        result[alg_prefix] = candidate

    return result


def open_plots_window(
    encrypted_image_path: Path | None,
    parent: tk.Misc | None = None,
    all_encrypted_paths: dict[str, Path] | None = None,
) -> tk.Toplevel | None:
    if encrypted_image_path is None:
        messagebox.showwarning("No image", "Please choose and encrypt an image first.")
        return None

    try:
        plain_path = _resolve_plain_path(encrypted_image_path)
    except Exception as exc:
        messagebox.showerror("Plots unavailable", str(exc))
        return None

    if all_encrypted_paths is None:
        all_encrypted_paths = _find_all_encrypted_variants(encrypted_image_path)

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
    window._temp_dir = temp_dir
    window._plot_images = []

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
        text=f"Plots for {plain_path.name}",
        font=(None, 12, "bold"),
    )
    header.pack(anchor="w", pady=(0, 8))

    notebook = ttk.Notebook(root_frame)
    notebook.pack(fill="both", expand=True)

    correlation_tab, correlation_content = _build_scrollable_tab(notebook)
    histogram_tab, histogram_content = _build_scrollable_tab(notebook)
    bitplane_tab, bitplane_content = _build_scrollable_tab(notebook)
    entropy_tab, entropy_content = _build_scrollable_tab(notebook)

    notebook.add(correlation_tab, text="Correlation")
    notebook.add(histogram_tab, text="Histogram")
    notebook.add(bitplane_tab, text="Bitplane")
    notebook.add(entropy_tab, text="Entropy")

    plot_images: list[ImageTk.PhotoImage] = window._plot_images

    enc_paths_for_plots = all_encrypted_paths or {
        encrypted_image_path.stem: encrypted_image_path
    }

    correlation_bars_path = Path(temp_dir.name) / "correlation_bars.png"
    render_correlation_bars(enc_paths_for_plots, correlation_bars_path)
    _add_image_panel(
        correlation_content,
        "Pixel correlation per algorithm (H / V / D)",
        correlation_bars_path,
        plot_images,
        max_size=(1100, 520),
    )

    scatter_path = Path(temp_dir.name) / "correlation_scatter.png"
    render_scatter_comparison(plain_path, enc_paths_for_plots, scatter_path)
    n_rows = 1 + len([p for p in enc_paths_for_plots.values() if p.exists()])
    _add_image_panel(
        correlation_content,
        "Pixel correlation scatter (Original + all algorithms)",
        scatter_path,
        plot_images,
        max_size=(1100, 400 * n_rows),
    )

    hist_all_path = Path(temp_dir.name) / "histogram_all_algorithms.png"
    render_histogram_all_algorithms(plain_path, enc_paths_for_plots, hist_all_path)
    _add_image_panel(
        histogram_content,
        "Histogram comparison (original vs all encrypted algorithms)",
        hist_all_path,
        plot_images,
        max_size=(1100, 520),
    )

    entropy_bars_path = Path(temp_dir.name) / "entropy_bars.png"
    render_entropy_bars(enc_paths_for_plots, entropy_bars_path)
    _add_image_panel(
        entropy_content,
        "Shannon entropy per algorithm",
        entropy_bars_path,
        plot_images,
        max_size=(1100, 520),
    )

    bitplane_bars_all_path = Path(temp_dir.name) / "bitplane_bars_all_algorithms.png"
    render_bitplane_bars_all_algorithms(plain_path, enc_paths_for_plots, bitplane_bars_all_path)
    n_algs = len([p for p in enc_paths_for_plots.values() if p.exists()])
    _add_image_panel(
        bitplane_content,
        "Bit-plane set-bit percentages — all algorithms",
        bitplane_bars_all_path,
        plot_images,
        max_size=(1100, 400 * n_algs),
    )

    return window
