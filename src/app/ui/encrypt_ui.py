from __future__ import annotations

import json
import time
from pathlib import Path
import traceback
import tkinter as tk
from tkinter import messagebox, ttk

from .styles import (
    C_SIDEBAR, C_SIDEBAR_ITEM, C_ACCENT, C_BTN_FG_OFF,
    C_BG, C_BORDER, C_FG, C_FG_DIM, C_FG_PLACEHOLDER,
    FONT_UI, FONT_BOLD, FONT_SMALL,
    apply_main_theme, make_sidebar_button, make_toolbar_button,
    make_filter_button, set_filter_active, make_result_text,
)

from ..utils import (
    ImageUiState,
    build_image_box,
    choose_image,
    display_image,
    encrypt_image,
    decrypt_image,
    add_one_bit_to_pixel,
)
from ..analysis_methods import (
    correlation_between_images,
    diagonal_pixel_correlation,
    horizontal_pixel_correlation,
    vertical_pixel_correlation,
    pixel_entropy_with_blocks,
    image_histogram,
    number_of_pixel_change_rate,
    mean_squared_error,
    peak_signal_to_noise_ratio,
    unified_average_changing_intensity,
    structural_similarity,
)
from .analyse_ui import (
    format_complete_analysis_result,
    format_entropy_result,
    format_histogram_result,
    format_result,
    build_analyse_ui,
)
from .plots_ui import open_plots_window

try:
    from tkinterdnd2 import TkinterDnD
except Exception:
    TkinterDnD = None

ENABLE_ANALYSE_UI = False

KEY_PHRASE = "encryptionkey"
ALL_ALGORITHMS = [
    "AES_CTR", "AES_CBC", "Triple_DES", "ChaCha20",
    "Logistic_Map", "Henon_Map", "Custom_v1", "Custom_v2",
]


ui_state = ImageUiState()

SELECTED_ALGORITHMS: set[str] = {"AES_CTR"}
ALGORITHM_BUTTONS: dict[str, tk.Button] = {}

LAST_ENCRYPTED_PATHS: dict[str, Path] = {}
LAST_ANALYSIS_DATA: dict[str, list[tuple[str, str, str]]] = {}
LAST_ENCRYPTION_TIMES: dict[str, float] = {}

_result_notebook: ttk.Notebook | None = None
_result_notebook_frame: tk.Frame | None = None
_img_box: tk.Canvas | None = None
_active_category: str = "All"

base_dir = Path(__file__).resolve().parent.parent / "shared"


def _toggle_algorithm(name: str) -> None:
    if name in SELECTED_ALGORITHMS:
        SELECTED_ALGORITHMS.discard(name)
    else:
        SELECTED_ALGORITHMS.add(name)
    _refresh_algorithm_buttons()


def _refresh_algorithm_buttons() -> None:
    for name, btn in ALGORITHM_BUTTONS.items():
        if name in SELECTED_ALGORITHMS:
            btn.config(bg=C_ACCENT, fg="white", relief="flat")
        else:
            btn.config(bg=C_SIDEBAR_ITEM, fg=C_BTN_FG_OFF, relief="flat")


def build_algorithm_menu(parent: tk.Widget) -> None:
    global ALGORITHM_BUTTONS
    ALGORITHM_BUTTONS = {}

    header = tk.Frame(parent, bg=C_SIDEBAR)
    header.pack(fill="x", padx=0, pady=0)
    tk.Label(
        header, text="ALGORITHMS", bg=C_SIDEBAR, fg=C_FG_DIM,
        font=("Segoe UI", 8, "bold"),
    ).pack(anchor="w", padx=14, pady=(16, 8))

    btn_frame = tk.Frame(parent, bg=C_SIDEBAR)
    btn_frame.pack(fill="x", padx=8)

    for name in ALL_ALGORITHMS:
        btn = tk.Button(
            btn_frame,
            text=name.replace("_", " "),
            anchor="w",
            relief="flat",
            bd=0,
            padx=10,
            pady=6,
            font=FONT_SMALL,
            cursor="hand2",
            command=lambda n=name: _toggle_algorithm(n),
        )
        btn.pack(fill="x", pady=2)
        ALGORITHM_BUTTONS[name] = btn

    _refresh_algorithm_buttons()


def _make_section(category: str, title: str, body: str) -> tuple[str, str, str]:
    return (category, title, body)


def _fmt(title: str, result: dict, percent: bool = False) -> str:
    return format_result(title, result, percent=percent).replace(f"{title}\n", "", 1)


def _run_analysis(encrypted_path: Path) -> list[tuple[str, str, str]]:
    encrypted_plus_one_bit_path = (
        encrypted_path.parent.parent / "encrypted_plus_one_bit" / encrypted_path.name
    )
    plain_path = (
        encrypted_path.parent.parent
        / "plain"
        / encrypted_path.name.split("_enc_", 1)[-1]
    )

    e = str(encrypted_path)
    eb = str(encrypted_plus_one_bit_path)
    p = str(plain_path)

    return [
        _make_section("Encrypted", "Horizontal correlation",
            _fmt("Horizontal correlation", horizontal_pixel_correlation(e))),
        _make_section("Encrypted", "Vertical correlation",
            _fmt("Vertical correlation", vertical_pixel_correlation(e))),
        _make_section("Encrypted", "Diagonal correlation",
            _fmt("Diagonal correlation", diagonal_pixel_correlation(e))),
        _make_section("Encrypted", "Entropy results",
            format_entropy_result("Entropy results", pixel_entropy_with_blocks(e))
            .replace("Entropy results\n", "", 1)),
        _make_section("Encrypted", "Histogram results",
            format_histogram_result("Histogram results", image_histogram(e))
            .replace("Histogram results\n", "", 1)),
        _make_section("Encrypted vs Encrypted", "Encrypted vs Encrypted correlation",
            _fmt("Encrypted vs Encrypted correlation", correlation_between_images(e, eb))),
        _make_section("Encrypted vs Encrypted", "NPCR results",
            _fmt("NPCR results", number_of_pixel_change_rate(e, eb), percent=True)),
        _make_section("Encrypted vs Encrypted", "UACI results",
            _fmt("UACI results", unified_average_changing_intensity(e, eb), percent=True)),
        _make_section("Encrypted vs Plain", "Encrypted vs Plain correlation",
            _fmt("Encrypted vs Plain correlation", correlation_between_images(e, p))),
        _make_section("Encrypted vs Plain", "MSE results",
            _fmt("MSE results", mean_squared_error(e, p))),
        _make_section("Encrypted vs Plain", "PSNR results",
            _fmt("PSNR results", peak_signal_to_noise_ratio(e, p))),
        _make_section("Encrypted vs Plain", "SSIM results",
            _fmt("SSIM results", structural_similarity(e, p))),
    ]


def _build_result_tab(
    notebook: ttk.Notebook,
    alg: str,
    sections: list[tuple[str, str, str]],
) -> None:
    global _active_category
    tab = ttk.Frame(notebook)
    notebook.add(tab, text=alg.replace("_", "-"))

    btn_row = tk.Frame(tab, bg=C_BG)
    btn_row.pack(fill="x", padx=8, pady=(6, 4))

    cat_buttons: dict[str, tk.Button] = {}

    text_frame = tk.Frame(tab, bg=C_BG, bd=0)
    text_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
    scrollbar = ttk.Scrollbar(text_frame, orient="vertical")
    scrollbar.pack(side="right", fill="y")
    result_text = make_result_text(text_frame, scrollbar)
    result_text.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=result_text.yview)

    def _render(cat: str) -> None:
        global _active_category
        _active_category = cat
        for name, btn in cat_buttons.items():
            set_filter_active(btn, name == cat)

        visible = sections if cat == "All" else [s for s in sections if s[0] == cat]
        text = (
            format_complete_analysis_result([(t, b) for _, t, b in visible])
            if visible else f"No results for '{cat}'."
        )
        result_text.config(state="normal")
        result_text.delete("1.0", "end")
        result_text.insert("1.0", text)
        result_text.config(state="disabled")

    tab._render = _render  # type: ignore[attr-defined]

    for cat in ("All", "Encrypted", "Encrypted vs Plain", "Encrypted vs Encrypted"):
        btn = make_filter_button(btn_row, cat, command=lambda c=cat: _render(c))
        btn.pack(side="left", padx=2)
        cat_buttons[cat] = btn

    _render(_active_category)


def _rebuild_result_notebook() -> None:
    global _result_notebook
    if _result_notebook_frame is None:
        return

    for widget in _result_notebook_frame.winfo_children():
        widget.destroy()
    _result_notebook = None

    if not LAST_ANALYSIS_DATA:
        return

    nb = ttk.Notebook(_result_notebook_frame)
    nb.pack(fill="both", expand=True, padx=0, pady=0)
    _result_notebook = nb

    for alg, sections in LAST_ANALYSIS_DATA.items():
        _build_result_tab(nb, alg, sections)

    def _on_tab_changed(_event=None) -> None:
        try:
            tab_text = nb.tab(nb.index("current"), "text")
            alg = tab_text.replace("-", "_")
            # Update displayed image
            if _img_box is not None:
                path = LAST_ENCRYPTED_PATHS.get(alg)
                if path and path.exists():
                    display_image(path, _img_box, ui_state, extra_window_height=320)
            # Re-apply the active category on the newly selected tab
            current_tab = nb.nametowidget(nb.select())
            render_fn = getattr(current_tab, "_render", None)
            if render_fn is not None:
                render_fn(_active_category)
        except Exception:
            pass

    nb.bind("<<NotebookTabChanged>>", _on_tab_changed)

    try:
        _result_notebook_frame.master.pack(fill="both", expand=True)
    except Exception:
        pass


def _save_analysis_json(enc_path: Path, sections: list[tuple[str, str, str]]) -> None:
    json_dir = enc_path.parent.parent / "analysis_results"
    json_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_dir / f"{enc_path.stem}_analysis.json"
    data = [{"category": c, "title": t, "body": b} for c, t, b in sections]
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_analysis_json(enc_path: Path) -> list[tuple[str, str, str]] | None:
    json_path = (
        enc_path.parent.parent / "analysis_results"
        / f"{enc_path.stem}_analysis.json"
    )
    if not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return [(item["category"], item["title"], item["body"]) for item in data]
    except Exception:
        return None


def _alg_from_enc_path(path: Path) -> str | None:
    """Detect algorithm name from an encrypted filename prefix."""
    stem = path.stem.lower()
    for alg in ALL_ALGORITHMS:
        if stem.startswith(alg.lower() + "_enc_"):
            return alg
    return None


def _choose_and_auto_load(img_box: tk.Canvas) -> None:
    choose_image(
        img_box, ui_state,
        title="Select image",
        initialdir=str(base_dir / "images"),
        extra_window_height=320,
    )
    path = ui_state.selected_image_path
    if path is None or "_enc_" not in path.name:
        return

    # Scan all encrypted variants of the same plain image in the same folder
    _, _, plain_name = path.name.partition("_enc_")
    plain_stem = Path(plain_name).stem
    encrypted_dir = path.parent

    LAST_ENCRYPTED_PATHS.clear()
    LAST_ANALYSIS_DATA.clear()

    for candidate in sorted(encrypted_dir.iterdir()):
        if "_enc_" not in candidate.name:
            continue
        _, _, cand_plain = candidate.name.partition("_enc_")
        if Path(cand_plain).stem != plain_stem:
            continue
        alg = _alg_from_enc_path(candidate)
        if alg is None:
            alg = candidate.name.split("_enc_")[0]
        sections = _load_analysis_json(candidate)
        if sections is None:
            continue
        LAST_ENCRYPTED_PATHS[alg] = candidate
        LAST_ANALYSIS_DATA[alg] = sections

    if LAST_ANALYSIS_DATA:
        _rebuild_result_notebook()


def on_encrypt(canvas: tk.Canvas) -> dict[str, Path]:
    if ui_state.selected_image_path is None:
        messagebox.showwarning("No image", "Please choose or drop an image first.")
        return {}
    if not SELECTED_ALGORITHMS:
        messagebox.showwarning("No algorithm", "Please select at least one algorithm.")
        return {}

    input_path = ui_state.selected_image_path
    # Block encryption of already-encrypted images
    if "_enc_" in input_path.name:
        messagebox.showwarning("Already encrypted", "The selected image is already encrypted. Please choose a plain image.")
        return {}
    # If the displayed image is a decrypted file, resolve back to the plain image
    if "_dec_" in input_path.name and input_path.parent.name == "decrypted":
        _, _, plain_name = input_path.name.partition("_dec_")
        plain_candidate = input_path.parent.parent / "plain" / plain_name
        if plain_candidate.exists():
            input_path = plain_candidate

    input_path_2 = add_one_bit_to_pixel(input_path)

    results: dict[str, Path] = {}
    errors: list[str] = []

    LAST_ENCRYPTION_TIMES.clear()

    for alg in sorted(SELECTED_ALGORITHMS):
        try:
            encrypted_dir = input_path.parent.parent / "encrypted"
            encrypted_dir.mkdir(parents=True, exist_ok=True)
            prefix = alg.lower()
            output_path = encrypted_dir / f"{prefix}_enc_{input_path.name}"
            t0 = time.perf_counter()
            encrypt_image(input_path, output_path, KEY_PHRASE, algorithm=alg)
            LAST_ENCRYPTION_TIMES[alg] = time.perf_counter() - t0

            plus_one_dir = input_path.parent.parent / "encrypted_plus_one_bit"
            plus_one_dir.mkdir(parents=True, exist_ok=True)
            encrypt_image(input_path_2, plus_one_dir / f"{prefix}_enc_{input_path.name}",
                          KEY_PHRASE, algorithm=alg)

            results[alg] = output_path
        except Exception as exc:
            errors.append(f"{alg}: {exc}")

    if errors:
        messagebox.showerror("Encryption errors", "\n".join(errors))

    if results:
        first_path = next(iter(results.values()))
        try:
            display_image(first_path, canvas, ui_state, extra_window_height=320)
        except Exception:
            pass

    LAST_ENCRYPTED_PATHS.clear()
    LAST_ENCRYPTED_PATHS.update(results)
    return results


def on_complete_analysis(paths: dict[str, Path] | None = None) -> None:
    paths = paths or LAST_ENCRYPTED_PATHS
    if not paths:
        messagebox.showwarning("No image", "Please encrypt an image first.")
        return

    LAST_ANALYSIS_DATA.clear()
    errors: list[str] = []

    for alg, enc_path in paths.items():
        try:
            sections = _run_analysis(enc_path)
            enc_time = LAST_ENCRYPTION_TIMES.get(alg)
            if enc_time is not None:
                time_body = f"Seconds: {enc_time:.3f} s"
                sections = [_make_section("Encrypted", "Encryption time", time_body)] + sections
            LAST_ANALYSIS_DATA[alg] = sections
            try:
                _save_analysis_json(enc_path, sections)
            except Exception:
                pass
        except Exception as exc:
            errors.append(f"{alg}:\n{exc}\n{traceback.format_exc()}")

    if errors:
        messagebox.showerror("Analysis errors", "\n\n".join(errors))

    # Also load any other encrypted variants of the same plain image that
    # have saved JSONs but weren't part of this encryption run
    first_enc = next(iter(paths.values()), None)
    if first_enc is not None:
        _, _, plain_name = first_enc.name.partition("_enc_")
        plain_stem = Path(plain_name).stem
        encrypted_dir = first_enc.parent
        for candidate in sorted(encrypted_dir.iterdir()):
            if "_enc_" not in candidate.name:
                continue
            _, _, cand_plain = candidate.name.partition("_enc_")
            if Path(cand_plain).stem != plain_stem:
                continue
            alg = _alg_from_enc_path(candidate) or candidate.name.split("_enc_")[0]
            if alg in LAST_ANALYSIS_DATA:
                continue  # already computed fresh
            sections = _load_analysis_json(candidate)
            if sections is None:
                continue
            LAST_ENCRYPTED_PATHS[alg] = candidate
            LAST_ANALYSIS_DATA[alg] = sections

    _rebuild_result_notebook()


def on_encrypt_and_analysis(canvas: tk.Canvas) -> None:
    paths = on_encrypt(canvas)
    if paths:
        on_complete_analysis(paths)


def on_decrypt(canvas: tk.Canvas) -> None:
    if ui_state.selected_image_path is None:
        messagebox.showwarning("No image", "Please choose or drop an image first.")
        return

    input_path = ui_state.selected_image_path
    stem = input_path.stem.lower()

    prefix_map = {
        "aes_cbc_": "AES_CBC", "aes-cbc_": "AES_CBC",
        "chacha20_": "ChaCha20", "chacha_": "ChaCha20",
        "aes_ctr_": "AES_CTR", "aes-ctr_": "AES_CTR",
        "triple_des_": "Triple_DES", "des_": "Triple_DES",
        "logistic_map_": "Logistic_Map", "logistic-map_": "Logistic_Map",
        "henon_map_": "Henon_Map", "henon-map_": "Henon_Map",
        "custom_v1_": "Custom_v1", "custom_v2_": "Custom_v2",
    }
    detected_alg = next(
        (v for k, v in prefix_map.items() if stem.startswith(k)),
        next(iter(SELECTED_ALGORITHMS), "AES_CTR"),
    )

    output_name = input_path.name
    if "_enc_" in output_name:
        alg_prefix, _, image_name = output_name.partition("_enc_")
        output_name = f"{alg_prefix}_dec_{image_name}"
    else:
        output_name = f"dec_{output_name}"

    try:
        decrypted_dir = input_path.parent.parent / "decrypted"
        decrypted_dir.mkdir(parents=True, exist_ok=True)
        output_path = decrypted_dir / output_name
        decrypt_image(input_path, output_path, KEY_PHRASE, algorithm=detected_alg)
        try:
            display_image(output_path, canvas, ui_state, extra_window_height=320)
        except Exception:
            pass
    except Exception as exc:
        messagebox.showerror("Error", f"Decryption failed:\n{exc}\n\n{traceback.format_exc()}")


def _on_view_plots(root: tk.Misc) -> None:
    """Open plots for the active notebook tab's algorithm, or fallback to selected image."""
    alg: str | None = None
    if _result_notebook is not None and LAST_ENCRYPTED_PATHS:
        try:
            tab_text = _result_notebook.tab(_result_notebook.index("current"), "text")
            alg = tab_text.replace("-", "_")
        except Exception:
            pass

    path = (
        LAST_ENCRYPTED_PATHS.get(alg)
        if alg and alg in LAST_ENCRYPTED_PATHS
        else (next(iter(LAST_ENCRYPTED_PATHS.values()), None) or ui_state.selected_image_path)
    )
    open_plots_window(
        path,
        parent=root,
        all_encrypted_paths=None,
        encryption_times=dict(LAST_ENCRYPTION_TIMES),
    )


def build_encrypt_ui() -> None:
    global _result_notebook_frame

    root = TkinterDnD.Tk() if TkinterDnD is not None else tk.Tk()
    root.title("Image Encryptor")
    root.minsize(780, 540)
    root.configure(bg=C_SIDEBAR)
    apply_main_theme(root)

    outer = tk.Frame(root, bg=C_SIDEBAR)
    outer.pack(fill="both", expand=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    sidebar = tk.Frame(outer, bg=C_SIDEBAR, width=148)
    sidebar.pack(side="left", fill="y")
    sidebar.pack_propagate(False)
    build_algorithm_menu(sidebar)

    if (ENABLE_ANALYSE_UI):
        tk.Frame(sidebar, bg=C_BORDER, height=1).pack(side="bottom", fill="x")
        make_sidebar_button(
            sidebar, "Analyse", command=lambda: _try_switch(root, build_analyse_ui)
        ).pack(side="bottom", fill="x")

    # ── Main content ─────────────────────────────────────────────────────────
    content = tk.Frame(outer, bg="#0d0f1a")
    content.pack(side="left", fill="both", expand=True)

    global _img_box
    img_box = build_image_box(
        content, ui_state,
        placeholder_text="Drop or choose an image",
        extra_window_height=320,
    )
    _img_box = img_box

    # ── Toolbar ──────────────────────────────────────────────────────────────
    toolbar = tk.Frame(content, bg=C_BG, pady=8)
    toolbar.pack(fill="x")
    btn_inner = tk.Frame(toolbar, bg=C_BG)
    btn_inner.pack(anchor="center")

    make_toolbar_button(btn_inner, "Choose image",
        command=lambda: _choose_and_auto_load(img_box)).pack(side="left", padx=4)
    make_toolbar_button(btn_inner, "Encrypt",
        command=lambda: on_encrypt(img_box)).pack(side="left", padx=4)
    make_toolbar_button(btn_inner, "Encrypt + Analysis",
        command=lambda: on_encrypt_and_analysis(img_box)).pack(side="left", padx=4)
    make_toolbar_button(btn_inner, "Decrypt",
        command=lambda: on_decrypt(img_box)).pack(side="left", padx=4)
    make_toolbar_button(btn_inner, "View plots",
        command=lambda: _on_view_plots(root)).pack(side="left", padx=4)

    # ── Results area ─────────────────────────────────────────────────────────
    results_wrap = tk.Frame(content, bg=C_BG)
    results_wrap.pack(fill="both", expand=True)

    results_header = tk.Frame(results_wrap, bg=C_BG)
    results_header.pack(fill="x", padx=12, pady=(10, 0))
    tk.Label(results_header, text="Results", font=FONT_BOLD,
             bg=C_BG, fg=C_FG).pack(side="left")
    tk.Frame(results_wrap, bg=C_BORDER, height=1).pack(fill="x", padx=12, pady=(4, 0))

    _result_notebook_frame = tk.Frame(results_wrap, bg=C_BG)
    _result_notebook_frame.pack(fill="both", expand=True)

    placeholder = tk.Label(
        _result_notebook_frame,
        text="Run Encrypt + Analysis to see results per algorithm.",
        fg=C_FG_PLACEHOLDER, bg=C_BG, font=FONT_UI,
    )
    placeholder.pack(expand=True)
    _result_notebook_frame._placeholder = placeholder

    root.mainloop()


def _try_switch(root: tk.Misc, builder) -> None:
    try:
        root.destroy()
    except Exception:
        pass
    try:
        builder()
    except Exception as e:
        messagebox.showerror("Error", str(e))


def main() -> None:
    build_encrypt_ui()


if __name__ == "__main__":
    main()
