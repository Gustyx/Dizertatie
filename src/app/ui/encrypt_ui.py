from pathlib import Path
import traceback
import tkinter as tk
from tkinter import messagebox

import numpy as np

from ..utils import (
    ImageUiState,
    build_image_box,
    choose_image,
    display_image,
    set_result_text,
    encrypt_image,
    decrypt_image,
    add_one_bit_to_pixel,
    extract_pixels,
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


KEY_PHRASE = "encryptionkey"
ALGORITHM_DEFAULT = "AES_CTR"

ui_state = ImageUiState()
ALGORITHM_SELECTION: tk.StringVar | None = None
ALGORITHM_BUTTONS: dict[str, tk.Button] = {}
RESULT_CATEGORY_DEFAULT = "All"
RESULT_CATEGORY_SELECTION: tk.StringVar | None = None
RESULT_CATEGORY_BUTTONS: dict[str, tk.Button] = {}
LAST_ANALYSIS_SECTIONS: list[tuple[str, str, str]] = []


base_dir = Path(__file__).resolve().parent.parent / "shared"


def _set_category_button_state(active_name: str) -> None:
    for name, button in RESULT_CATEGORY_BUTTONS.items():
        if name == active_name:
            button.config(relief="sunken", bg="#cfe")
        else:
            button.config(relief="raised", bg=button.master.cget("bg"))


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


def on_encrypt(canvas: tk.Canvas) -> Path | None:
    if encrypt_image is None:
        messagebox.showerror("Error", "encrypt_image function not available")
        return None
    if ui_state.selected_image_path is None:
        messagebox.showwarning("No image", "Please choose or drop an image first.")
        return None

    try:
        input_path = ui_state.selected_image_path

        try:
            alg = ALGORITHM_SELECTION.get()
        except Exception:
            alg = ALGORITHM_DEFAULT

        encrypted_dir = input_path.parent.parent / "encrypted"
        encrypted_dir.mkdir(parents=True, exist_ok=True)
        prefix = alg.lower()
        output_path = encrypted_dir / f"{prefix}_enc_{input_path.name}"

        encrypt_image(input_path, output_path, KEY_PHRASE, algorithm=alg)

        input_path_2 = add_one_bit_to_pixel(input_path)
        encrypted_plus_one_bit_dir = input_path.parent.parent / "encrypted_plus_one_bit"
        encrypted_plus_one_bit_dir.mkdir(parents=True, exist_ok=True)
        output_path_2 = encrypted_plus_one_bit_dir / f"{prefix}_enc_{input_path.name}"

        encrypt_image(input_path_2, output_path_2, KEY_PHRASE, algorithm=alg)
        # replace displayed image with encrypted output
        try:
            display_image(output_path, canvas, ui_state, extra_window_height=320)
        except Exception:
            pass
        return output_path
    except Exception as exc:
        messagebox.showerror(
            "Error", f"Encryption failed:\n{exc}\n\n{traceback.format_exc()}"
        )
        return None


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

        arr, _ = extract_pixels(encrypted_path)
        arr = np.asarray(arr, dtype=np.uint8)

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
                    "Entropy results", pixel_entropy_with_blocks(arr)
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


def on_encrypt_and_analysis(canvas: tk.Canvas) -> None:
    output_path = on_encrypt(canvas)
    if output_path is None:
        return
    on_complete_analysis()


def on_decrypt(canvas: tk.Canvas) -> None:
    if decrypt_image is None:
        messagebox.showerror("Error", "decrypt_image function not available")
        return
    if ui_state.selected_image_path is None:
        messagebox.showwarning("No image", "Please choose or drop an image first.")
        return

    try:
        input_path = ui_state.selected_image_path
        # Decide algorithm from filename prefix if possible
        stem = input_path.stem
        if stem.lower().startswith("aes-cbc_") or stem.lower().startswith("aes_cbc_"):
            detected_alg = "AES_CBC"
            print("Detected AES-CBC from filename prefix")
        elif stem.lower().startswith("chacha20_") or stem.lower().startswith("chacha_"):
            detected_alg = "CHACHA20"
            print("Detected ChaCha20 from filename prefix")
        elif stem.lower().startswith("aes-ctr_") or stem.lower().startswith("aes_ctr_"):
            detected_alg = "AES_CTR"
        elif stem.lower().startswith("des_"):
            detected_alg = "DES"
        elif stem.lower().startswith("custom-aes_") or stem.lower().startswith(
            "custom_aes_"
        ):
            detected_alg = "CUSTOM_AES"
        elif stem.lower().startswith("logistic-map_") or stem.lower().startswith(
            "logistic_map_"
        ):
            detected_alg = "LOGISTIC_MAP"
        elif stem.lower().startswith("henon-map_") or stem.lower().startswith(
            "henon_map_"
        ):
            detected_alg = "HENON_MAP"
        else:
            try:
                detected_alg = ALGORITHM_SELECTION.get()
            except Exception:
                detected_alg = ALGORITHM_DEFAULT

        output_name = input_path.name
        if "_enc_" in output_name:
            alg_prefix, _, image_name = output_name.partition("_enc_")
            output_name = f"{alg_prefix}_dec_{image_name}"
        else:
            output_name = f"dec_{output_name}"

        decrypted_dir = input_path.parent.parent / "decrypted"
        decrypted_dir.mkdir(parents=True, exist_ok=True)
        output_path = decrypted_dir / output_name

        decrypt_image(input_path, output_path, KEY_PHRASE, algorithm=detected_alg)
        # replace displayed image with decrypted output
        try:
            display_image(output_path, canvas, ui_state, extra_window_height=320)
        except Exception:
            pass
    except Exception as exc:
        messagebox.showerror(
            "Error", f"Decryption failed:\n{exc}\n\n{traceback.format_exc()}"
        )


def build_buttons(frame, img_box) -> None:
    choose_btn = tk.Button(
        frame,
        text="Choose image",
        command=lambda: choose_image(
            img_box,
            ui_state,
            title="Select image",
            initialdir=str(base_dir / "images"),
            extra_window_height=320,
        ),
    )
    choose_btn.pack()

    ui_state.action_buttons_frame = tk.Frame(frame)

    encrypt_btn = tk.Button(
        ui_state.action_buttons_frame,
        text="Encrypt",
        width=12,
        command=lambda: on_encrypt(img_box),
    )
    encrypt_analysis_btn = tk.Button(
        ui_state.action_buttons_frame,
        text="Encrypt + Analysis",
        width=18,
        command=lambda: on_encrypt_and_analysis(img_box),
    )
    decrypt_btn = tk.Button(
        ui_state.action_buttons_frame,
        text="Decrypt",
        width=12,
        command=lambda: on_decrypt(img_box),
    )
    plots_btn = tk.Button(
        ui_state.action_buttons_frame,
        text="View plots",
        width=12,
        command=lambda: open_plots_window(
            ui_state.selected_image_path,
            parent=frame.winfo_toplevel(),
        ),
    )
    encrypt_btn.pack(side="left", padx=6)
    encrypt_analysis_btn.pack(side="left", padx=6)
    decrypt_btn.pack(side="left", padx=6)
    plots_btn.pack(side="left", padx=6)


def build_algorithm_menu(parent) -> None:
    """Build vertical algorithm buttons in the left sidebar."""
    global ALGORITHM_SELECTION, ALGORITHM_BUTTONS
    ALGORITHM_SELECTION = tk.StringVar(value=ALGORITHM_DEFAULT)

    tk.Label(parent, text="Algorithms", font=(None, 10, "bold")).pack(pady=(0, 6))

    def make_btn(display_name: str) -> None:
        btn = tk.Button(parent, text=display_name, width=12)

        def on_click():
            set_algorithm(display_name)

        btn.config(command=on_click)
        btn.pack(pady=4, fill="x")
        ALGORITHM_BUTTONS[display_name] = btn

    make_btn("AES_CTR")
    make_btn("AES_CBC")
    make_btn("Triple_DES")
    make_btn("ChaCha20")
    make_btn("Logistic_Map")
    make_btn("Henon_Map")
    make_btn("Custom_AES")

    # initialize visuals
    def set_algorithm(display_name: str):
        ALGORITHM_SELECTION.set(display_name)
        for name, btn in ALGORITHM_BUTTONS.items():
            if name == display_name:
                btn.config(relief="sunken", bg="#cfe")
            else:
                btn.config(relief="raised", bg=parent.cget("bg"))

    set_algorithm(ALGORITHM_DEFAULT)
    ALGORITHM_BUTTONS[ALGORITHM_DEFAULT].config(relief="sunken", bg="#cfe")


def build_result_category_menu(parent) -> None:
    global RESULT_CATEGORY_SELECTION, RESULT_CATEGORY_BUTTONS
    RESULT_CATEGORY_SELECTION = tk.StringVar(value=RESULT_CATEGORY_DEFAULT)

    ui_state.result_category_frame = tk.Frame(parent)

    tk.Label(
        ui_state.result_category_frame, text="Results", font=(None, 10, "bold")
    ).pack(pady=(0, 6))

    def make_btn(display_name: str) -> None:
        btn = tk.Button(ui_state.result_category_frame, text=display_name, width=12)

        def on_click() -> None:
            _set_result_category(display_name)

        btn.config(command=on_click)
        btn.pack(pady=4, fill="x")
        RESULT_CATEGORY_BUTTONS[display_name] = btn

    make_btn("All")
    make_btn("Encrypted")
    make_btn("Encrypted vs Plain")
    make_btn("Encrypted vs Encrypted")

    _set_category_button_state(RESULT_CATEGORY_DEFAULT)
    RESULT_CATEGORY_SELECTION.set(RESULT_CATEGORY_DEFAULT)


def build_encrypt_ui() -> None:
    # create root (use DnD root if available)
    if TkinterDnD is not None:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()

    root.title("Image Encryptor")
    root.minsize(320, 240)

    frame = tk.Frame(root, padx=10, pady=10)
    frame.pack(expand=True, fill="both")

    # Left sidebar for algorithm selection
    alg_sidebar = tk.Frame(frame)
    alg_sidebar.pack(side="left", fill="y", padx=(0, 8))
    build_algorithm_menu(alg_sidebar)
    tk.Frame(alg_sidebar).pack(fill="both", expand=True)
    build_result_category_menu(alg_sidebar)

    # Main content area (image + controls)
    content = tk.Frame(frame)
    content.pack(side="left", fill="both", expand=True)

    img_box = build_image_box(
        content,
        ui_state,
        placeholder_text="Drop or choose an image",
        extra_window_height=320,
    )

    controls_frame = tk.Frame(content)
    controls_frame.pack()

    build_buttons(controls_frame, img_box)

    ui_state.result_frame = tk.LabelFrame(content, text="Results")
    ui_state.result_frame.pack(side="left", fill="both", expand=True)

    result_container = tk.Frame(ui_state.result_frame)
    result_container.pack(fill="both", expand=True)

    result_scrollbar = tk.Scrollbar(result_container, orient="vertical")
    result_scrollbar.pack(side="right", fill="y")

    ui_state.result_widget = tk.Text(
        result_container,
        wrap="word",
        height=14,
        padx=8,
        pady=8,
        yscrollcommand=result_scrollbar.set,
        relief="flat",
        borderwidth=0,
    )
    ui_state.result_widget.pack(side="left", fill="both", expand=True)
    result_scrollbar.config(command=ui_state.result_widget.yview)
    ui_state.result_widget.insert(
        "1.0",
        "Choose an image, then run Encrypt + Analysis to see all metrics in one scrollable results box.",
    )
    ui_state.result_widget.config(state="disabled")

    # Mode switch button to open analysis UI
    def switch_to_analyse():
        try:
            root.destroy()
        except Exception:
            pass
        try:
            build_analyse_ui()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open Analyse UI:\n{e}")

    bottom = tk.Frame(frame)
    bottom.pack(side="bottom", fill="x", pady=(8, 0))
    switch_btn = tk.Button(bottom, text="Analyse", width=12, command=switch_to_analyse)
    switch_btn.pack()

    ui_state.result_frame.pack_forget()

    root.mainloop()


def main() -> None:
    build_encrypt_ui()


if __name__ == "__main__":
    main()
