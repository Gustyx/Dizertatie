from pathlib import Path
import traceback
import tkinter as tk
from tkinter import messagebox, ttk

from .styles import (
    C_SIDEBAR, C_BG, C_BORDER, C_FG, FONT_BOLD,
    apply_main_theme, make_sidebar_button, make_toolbar_button,
    set_filter_active, make_result_text,
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
from ..utils import ImageUiState, build_image_box, choose_image, set_result_text

try:
    from tkinterdnd2 import TkinterDnD
except Exception:
    TkinterDnD = None

ANALYSIS_DEFAULT = "NPCR"
RESULT_CATEGORY_DEFAULT = "All"

ui_state = ImageUiState()
analysis_selection: tk.StringVar | None = None
analysis_buttons: dict[str, tk.Button] = {}
RESULT_CATEGORY_SELECTION: tk.StringVar | None = None
RESULT_CATEGORY_BUTTONS: dict[str, tk.Button] = {}
LAST_ANALYSIS_SECTIONS: list[tuple[str, str, str]] = []


def _set_category_button_state(active_name: str) -> None:
    for name, button in RESULT_CATEGORY_BUTTONS.items():
        set_filter_active(button, name == active_name)


def _render_result_sections() -> None:
    if RESULT_CATEGORY_SELECTION is None:
        return

    category = RESULT_CATEGORY_SELECTION.get()
    if not LAST_ANALYSIS_SECTIONS:
        set_result_text(ui_state, "")
        return

    if category == "All":
        visible_sections = LAST_ANALYSIS_SECTIONS
    else:
        visible_sections = [
            section for section in LAST_ANALYSIS_SECTIONS if section[0] == category
        ]

    if not visible_sections:
        set_result_text(ui_state, f"No results available for {category}.")
        return

    set_result_text(
        ui_state,
        format_complete_analysis_result(
            [(title, body) for _, title, body in visible_sections]
        ),
    )


def _set_result_category(category_name: str) -> None:
    if RESULT_CATEGORY_SELECTION is None:
        return

    RESULT_CATEGORY_SELECTION.set(category_name)
    _set_category_button_state(category_name)
    _render_result_sections()


def _make_section(category: str, title: str, body: str) -> tuple[str, str, str]:
    return (category, title, body)


def _format_result_body(title: str, result: dict, percent: bool = False) -> str:
    return format_result(title, result, percent=percent).replace(f"{title}\n", "", 1)


def format_result(title: str, result: dict, percent: bool = False) -> str:
    suffix = " %" if percent else ""
    lines = [title]
    # Show per-channel values first when available
    if "per_channel" in result:
        lines.append("Per channel:")
        for channel, value in result["per_channel"].items():
            lines.append(f"  {channel}: {value:.6f}{suffix}")

    # Then include grayscale/overall values if present
    if "grayscale" in result:
        lines.append(f"Grayscale: {result['grayscale']:.6f}{suffix}")
    if "overall" in result:
        lines.append(f"Overall: {result['overall']:.6f}{suffix}")
    if "blocks" in result:
        lines.append(
            f"Blocks used: {result['blocks']} ({result['used_size'][0]}x{result['used_size'][1]} area)"
        )
    return "\n".join(lines)


def format_correlation_result(
    title: str,
    encrypted_path: str,
    plain_path: str,
    encrypted_plus_one_bit_path: str,
) -> str:
    sections = [
        (
            "Adjacent pixels inside encrypted image",
            format_complete_analysis_result(
                [
                    (
                        "Horizontal",
                        format_result(
                            "Horizontal",
                            horizontal_pixel_correlation(encrypted_path),
                            percent=False,
                        ).replace("Horizontal\n", "", 1),
                    ),
                    (
                        "Vertical",
                        format_result(
                            "Vertical",
                            vertical_pixel_correlation(encrypted_path),
                            percent=False,
                        ).replace("Vertical\n", "", 1),
                    ),
                    (
                        "Diagonal",
                        format_result(
                            "Diagonal",
                            diagonal_pixel_correlation(encrypted_path),
                            percent=False,
                        ).replace("Diagonal\n", "", 1),
                    ),
                ]
            ),
        ),
        (
            "Plain vs encrypted image",
            format_result(
                "Plain vs encrypted image",
                correlation_between_images(plain_path, encrypted_path),
                percent=False,
            ).replace("Plain vs encrypted image\n", "", 1),
        ),
        (
            "Encrypted vs encrypted plus one bit",
            format_result(
                "Encrypted vs encrypted plus one bit",
                correlation_between_images(encrypted_path, encrypted_plus_one_bit_path),
                percent=False,
            ).replace("Encrypted vs encrypted plus one bit\n", "", 1),
        ),
    ]

    return format_complete_analysis_result(
        [(title, format_complete_analysis_result(sections))]
    )


def format_histogram_result(title: str, result: dict) -> str:
    lines = [title]

    if "per_channel" in result:
        lines.append("Per channel:")
        for channel, summary in result["per_channel"].items():
            lines.append(
                f"  {channel}: peak {summary['peak_intensity']} ({summary['peak_count']} pixels, {summary['peak_percentage']:.2f}% of {summary['total_pixels']})"
            )
            lines.append(f"    non-zero bins: {summary['non_zero_bins']}")
            lines.append(f"    mean intensity: {summary['mean_intensity']:.6f}")
            chi_p = summary.get("chi2_p")
            if chi_p is None:
                lines.append(f"    chi2: {summary.get('chi2'):.3f} (p: N/A)")
            else:
                lines.append(f"    chi2: {summary.get('chi2'):.3f} (p: {chi_p:.3e})")

    if "grayscale" in result:
        summary = result["grayscale"]
        lines.append(
            f"Grayscale peak: {summary['peak_intensity']} ({summary['peak_count']} pixels, {summary['peak_percentage']:.2f}% of {summary['total_pixels']})"
        )
        lines.append(f"Non-zero bins: {summary['non_zero_bins']}")
        lines.append(f"Mean intensity: {summary['mean_intensity']:.6f}")
        chi_p = summary.get("chi2_p")
        if chi_p is None:
            lines.append(f"chi2: {summary.get('chi2'):.3f} (p: N/A)")
        else:
            lines.append(f"chi2: {summary.get('chi2'):.3f} (p: {chi_p:.3e})")

    if "luminance" in result:
        summary = result["luminance"]
        lines.append("Luminance:")
        lines.append(
            f"  peak: {summary['peak_intensity']} ({summary['peak_count']} pixels, {summary['peak_percentage']:.2f}% of {summary['total_pixels']})"
        )
        lines.append(f"  non-zero bins: {summary['non_zero_bins']}")
        lines.append(f"  mean intensity: {summary['mean_intensity']:.6f}")
        chi_p = summary.get("chi2_p")
        if chi_p is None:
            lines.append(f"  chi2: {summary.get('chi2'):.3f} (p: N/A)")
        else:
            lines.append(f"  chi2: {summary.get('chi2'):.3f} (p: {chi_p:.3e})")

    if "overall" in result:
        summary = result["overall"]
        lines.append("Overall:")
        lines.append(f"  mean intensity: {summary['mean_intensity']:.6f}")
        chi_p = summary.get("chi2_p")
        if chi_p is None:
            lines.append(f"  chi2: {summary.get('chi2'):.3f} (p: N/A)")
        else:
            lines.append(f"  chi2: {summary.get('chi2'):.3f} (p: {chi_p:.3e})")

    return "\n".join(lines)


def format_entropy_result(title: str, result: dict) -> str:
    lines = [title]

    if "per_channel" in result:
        lines.append("Per channel:")
        for channel, value in result["per_channel"].items():
            lines.append(f"  {channel}: {value:.6f}")

    if "grayscale" in result:
        lines.append(f"Grayscale: {result['grayscale']:.6f}")
    if "overall" in result:
        lines.append(f"Overall: {result['overall']:.6f}")

    if "block_entropies" in result:
        for block_size in sorted(result["block_entropies"]):
            block_result = result["block_entropies"][block_size]
            lines.append(f"Block entropy ({block_size}x{block_size}):")
            if "error" in block_result:
                lines.append(f"  N/A: {block_result['error']}")
                continue
            if "per_channel" in block_result:
                lines.append("  Per channel:")
                for channel, value in block_result["per_channel"].items():
                    lines.append(f"    {channel}: {value:.6f}")
                lines.append(f"  Overall: {block_result['overall']:.6f}")
            else:
                lines.append(f"  Grayscale: {block_result['grayscale']:.6f}")
            lines.append(
                f"  Blocks used: {block_result['blocks']} ({block_result['used_size'][0]}x{block_result['used_size'][1]} area)"
            )

    return "\n".join(lines)


def format_complete_analysis_result(result_sections: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"{title}\n{body}" for title, body in result_sections)


def on_complete_analysis() -> None:
    if ui_state.selected_image_path is None:
        messagebox.showwarning(
            "No image", "Please choose or drop an encrypted image first."
        )
        return

    try:
        encrypted_path = ui_state.selected_image_path
        encrypted_plus_one_bit_path = (
            encrypted_path.parent.parent
            / "encrypted_plus_one_bit"
            / f"{encrypted_path.name}"
        )
        plain_path = (
            encrypted_path.parent.parent
            / "plain"
            / f"{encrypted_path.name.split('_enc_')[-1]}"
        )

        e_str = str(encrypted_path)
        eb_str = str(encrypted_plus_one_bit_path)
        p_str = str(plain_path)

        sections = [
            _make_section(
                "Encrypted",
                "Horizontal correlation",
                _format_result_body(
                    "Horizontal correlation",
                    horizontal_pixel_correlation(e_str),
                ),
            ),
            _make_section(
                "Encrypted",
                "Vertical correlation",
                _format_result_body(
                    "Vertical correlation",
                    vertical_pixel_correlation(e_str),
                ),
            ),
            _make_section(
                "Encrypted",
                "Diagonal correlation",
                _format_result_body(
                    "Diagonal correlation",
                    diagonal_pixel_correlation(e_str),
                ),
            ),
            _make_section(
                "Encrypted",
                "Entropy results",
                format_entropy_result(
                    "Entropy results", pixel_entropy_with_blocks(e_str)
                ).replace("Entropy results\n", "", 1),
            ),
            _make_section(
                "Encrypted",
                "Histogram results",
                format_histogram_result(
                    "Histogram results", image_histogram(e_str)
                ).replace("Histogram results\n", "", 1),
            ),
            _make_section(
                "Encrypted vs Encrypted",
                "Encrypted vs Encrypted correlation",
                _format_result_body(
                    "Encrypted vs Encrypted correlation",
                    correlation_between_images(e_str, eb_str),
                ),
            ),
            _make_section(
                "Encrypted vs Encrypted",
                "NPCR results",
                _format_result_body(
                    "NPCR results",
                    number_of_pixel_change_rate(e_str, eb_str),
                    percent=True,
                ),
            ),
            _make_section(
                "Encrypted vs Encrypted",
                "UACI results",
                _format_result_body(
                    "UACI results",
                    unified_average_changing_intensity(e_str, eb_str),
                    percent=True,
                ),
            ),
            _make_section(
                "Encrypted vs Plain",
                "Encrypted vs Plain correlation",
                _format_result_body(
                    "Encrypted vs Plain correlation",
                    correlation_between_images(e_str, p_str),
                ),
            ),
            _make_section(
                "Encrypted vs Plain",
                "MSE results",
                _format_result_body(
                    "MSE results",
                    mean_squared_error(e_str, p_str),
                ),
            ),
            _make_section(
                "Encrypted vs Plain",
                "PSNR results",
                _format_result_body(
                    "PSNR results",
                    peak_signal_to_noise_ratio(e_str, p_str),
                ),
            ),
            _make_section(
                "Encrypted vs Plain",
                "SSIM results",
                _format_result_body(
                    "SSIM results",
                    structural_similarity(e_str, p_str),
                ),
            ),
        ]

        LAST_ANALYSIS_SECTIONS[:] = sections
        _set_result_category(
            RESULT_CATEGORY_SELECTION.get() if RESULT_CATEGORY_SELECTION else "All"
        )
    except Exception as exc:
        messagebox.showerror(
            "Error", f"Analysis failed:\n{exc}\n\n{traceback.format_exc()}"
        )


def on_analyse() -> None:
    if ui_state.selected_image_path is None:
        messagebox.showwarning(
            "No image", "Please choose or drop an encrypted image first."
        )
        return

    try:
        metric = analysis_selection.get()
    except Exception:
        metric = ANALYSIS_DEFAULT

    try:
        encrypted_path = ui_state.selected_image_path
        if metric == "Entropy":
            result = pixel_entropy_with_blocks(str(encrypted_path))
            text = format_entropy_result("", result)
        elif metric == "Histogram":
            result = image_histogram(str(encrypted_path))
            text = format_histogram_result("", result)
        else:
            encrypted_plus_one_bit_path = (
                encrypted_path.parent.parent
                / "encrypted_plus_one_bit"
                / f"{encrypted_path.name}"
            )
            plain_path = (
                encrypted_path.parent.parent
                / "plain"
                / f"{encrypted_path.name.split('_enc_')[-1]}"
            )
            # Some analysis functions expect path strings or numpy arrays,
            # not Path objects. Convert to strings for compatibility.
            e_str = str(encrypted_path)
            eb_str = str(encrypted_plus_one_bit_path)
            p_str = str(plain_path)

            if metric == "NPCR":
                result = number_of_pixel_change_rate(e_str, eb_str)
                text = format_result("", result, percent=True)
            elif metric == "UACI":
                result = unified_average_changing_intensity(e_str, eb_str)
                text = format_result("", result, percent=True)
            elif metric == "Correlation":
                text = format_correlation_result("", e_str, p_str, eb_str)
            elif metric == "PSNR":
                result = peak_signal_to_noise_ratio(e_str, p_str)
                text = format_result("", result, percent=False)
            elif metric == "SSIM":
                result = structural_similarity(e_str, p_str)
                text = format_result("", result, percent=False)
            elif metric == "MSE":
                result = mean_squared_error(e_str, p_str)
                text = format_result("", result, percent=False)
            else:
                raise ValueError(f"Unsupported analysis metric: {metric}")

        category = {
            "NPCR": "Encrypted vs Encrypted",
            "UACI": "Encrypted vs Encrypted",
            "Correlation": "Encrypted vs Plain",
            "MSE": "Encrypted vs Plain",
            "PSNR": "Encrypted vs Plain",
            "SSIM": "Encrypted vs Plain",
            "Entropy": "Encrypted",
            "Histogram": "Encrypted",
        }.get(metric, "All")

        LAST_ANALYSIS_SECTIONS[:] = [_make_section(category, f"{metric} results", text)]
        _set_result_category(
            RESULT_CATEGORY_SELECTION.get() if RESULT_CATEGORY_SELECTION else "All"
        )
    except Exception as exc:
        messagebox.showerror(
            "Error", f"Analysis failed:\n{exc}\n\n{traceback.format_exc()}"
        )


def build_analysis_menu(parent) -> None:
    global analysis_selection, analysis_buttons
    analysis_selection = tk.StringVar(value=ANALYSIS_DEFAULT)

    tk.Label(parent, text="Analysis", font=FONT_BOLD,
             bg=C_SIDEBAR, fg=C_FG).pack(pady=(12, 6), padx=10, anchor="w")

    def make_btn(display_name: str) -> None:
        def on_click() -> None:
            analysis_selection.set(display_name)
            for name, button in analysis_buttons.items():
                set_filter_active(button, name == display_name)

        btn = make_sidebar_button(parent, display_name, command=on_click)
        btn.pack(fill="x")
        analysis_buttons[display_name] = btn

    make_btn("NPCR")
    make_btn("UACI")
    make_btn("Correlation")
    make_btn("MSE")
    make_btn("PSNR")
    make_btn("SSIM")
    make_btn("Entropy")
    make_btn("Histogram")

    analysis_selection.set(ANALYSIS_DEFAULT)
    set_filter_active(analysis_buttons[ANALYSIS_DEFAULT], True)


def build_result_category_menu(parent) -> None:
    global RESULT_CATEGORY_SELECTION, RESULT_CATEGORY_BUTTONS
    RESULT_CATEGORY_SELECTION = tk.StringVar(value=RESULT_CATEGORY_DEFAULT)

    ui_state.result_category_frame = tk.Frame(parent, bg=C_SIDEBAR)

    tk.Frame(ui_state.result_category_frame, bg=C_BORDER, height=1).pack(fill="x")
    tk.Label(
        ui_state.result_category_frame, text="Results", font=FONT_BOLD,
        bg=C_SIDEBAR, fg=C_FG,
    ).pack(pady=(10, 6), padx=10, anchor="w")

    def make_btn(display_name: str) -> None:
        btn = make_sidebar_button(
            ui_state.result_category_frame, display_name,
            command=lambda n=display_name: _set_result_category(n),
        )
        btn.pack(fill="x")
        RESULT_CATEGORY_BUTTONS[display_name] = btn

    make_btn("All")
    make_btn("Encrypted")
    make_btn("Encrypted vs Plain")
    make_btn("Encrypted vs Encrypted")

    _set_category_button_state(RESULT_CATEGORY_DEFAULT)
    RESULT_CATEGORY_SELECTION.set(RESULT_CATEGORY_DEFAULT)


def build_analyse_ui() -> None:
    if TkinterDnD is not None:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()

    root.title("Image Analysis")
    root.minsize(800, 500)
    root.configure(bg=C_SIDEBAR)
    apply_main_theme(root)

    outer = tk.Frame(root, bg=C_SIDEBAR)
    outer.pack(fill="both", expand=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    sidebar = tk.Frame(outer, bg=C_SIDEBAR, width=160)
    sidebar.pack(side="left", fill="y")
    sidebar.pack_propagate(False)

    build_analysis_menu(sidebar)
    tk.Frame(sidebar, bg=C_SIDEBAR).pack(fill="both", expand=True)
    build_result_category_menu(sidebar)

    tk.Frame(sidebar, bg=C_BORDER, height=1).pack(side="bottom", fill="x")
    make_sidebar_button(
        sidebar, "Encryption",
        command=lambda: _switch_to_encrypt(root),
    ).pack(side="bottom", fill="x")

    # ── Main content ─────────────────────────────────────────────────────────
    content = tk.Frame(outer, bg="#0d0f1a")
    content.pack(side="left", fill="both", expand=True)

    img_box = build_image_box(
        content,
        ui_state,
        placeholder_text="Drop or choose an encrypted image",
        extra_window_height=180,
    )

    # ── Toolbar ──────────────────────────────────────────────────────────────
    toolbar = tk.Frame(content, bg=C_BG, pady=8)
    toolbar.pack(fill="x")
    btn_inner = tk.Frame(toolbar, bg=C_BG)
    btn_inner.pack(anchor="center")

    base_dir = Path(__file__).resolve().parent
    encrypted_dir = base_dir / "images" / "encrypted"

    make_toolbar_button(
        btn_inner, "Choose encrypted image",
        command=lambda: choose_image(
            img_box, ui_state,
            title="Select encrypted image",
            initialdir=str(encrypted_dir) if encrypted_dir.exists() else None,
            extra_window_height=180,
        ),
    ).pack(side="left", padx=4)

    ui_state.action_buttons_frame = tk.Frame(btn_inner, bg=C_BG)

    make_toolbar_button(
        ui_state.action_buttons_frame, "Analyse", command=on_analyse
    ).pack(side="left", padx=4)
    make_toolbar_button(
        ui_state.action_buttons_frame, "Complete Analysis", command=on_complete_analysis
    ).pack(side="left", padx=4)

    # ── Results area ─────────────────────────────────────────────────────────
    results_wrap = tk.Frame(content, bg=C_BG)
    results_wrap.pack(fill="both", expand=True)

    results_header = tk.Frame(results_wrap, bg=C_BG)
    results_header.pack(fill="x", padx=12, pady=(10, 0))
    tk.Label(results_header, text="Results", font=FONT_BOLD,
             bg=C_BG, fg=C_FG).pack(side="left")
    tk.Frame(results_wrap, bg=C_BORDER, height=1).pack(fill="x", padx=12, pady=(4, 0))

    ui_state.result_frame = tk.Frame(results_wrap, bg=C_BG)

    result_container = tk.Frame(ui_state.result_frame, bg=C_BG)
    result_container.pack(fill="both", expand=True, padx=12, pady=8)

    result_scrollbar = ttk.Scrollbar(result_container, orient="vertical")
    result_scrollbar.pack(side="right", fill="y")

    ui_state.result_widget = make_result_text(result_container, result_scrollbar)
    ui_state.result_widget.pack(side="left", fill="both", expand=True)
    result_scrollbar.config(command=ui_state.result_widget.yview)
    ui_state.result_widget.config(state="normal")
    ui_state.result_widget.insert(
        "1.0",
        "Choose an encrypted image, then run NPCR, UACI, Correlation, MSE, PSNR, Entropy, Histogram, or Complete Analysis.",
    )
    ui_state.result_widget.config(state="disabled")

    ui_state.result_category_frame.pack_forget()
    ui_state.action_buttons_frame.pack_forget()
    ui_state.result_frame.pack_forget()

    root.mainloop()


def _switch_to_encrypt(root: tk.Tk) -> None:
    try:
        root.destroy()
    except Exception:
        pass
    try:
        from .encrypt_ui import build_encrypt_ui
        build_encrypt_ui()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open Encrypt UI:\n{e}")


def main() -> None:
    build_analyse_ui()


if __name__ == "__main__":
    main()
