from pathlib import Path
import traceback
import tkinter as tk
from tkinter import messagebox

from PIL import Image
import numpy as np

from imagePixels import add_one_bit_to_pixel
from entropy import pixel_entropy_with_blocks
from histogram import image_histogram
from handleImage import (
    DND_AVAILABLE,
    ImageUiState,
    build_image_box,
    choose_image,
    display_image,
    set_result_text,
)
from nprc import number_of_pixel_change_rate
from psnr import mean_squared_error, peak_signal_to_noise_ratio
from uaci import unified_average_changing_intensity
from analyse_ui import (
    format_complete_analysis_result,
    format_correlation_result,
    format_entropy_result,
    format_histogram_result,
    format_result,
)
from ssim import structural_similarity

try:
    from tkinterdnd2 import TkinterDnD
except Exception:
    TkinterDnD = None

try:
    import enryptImage
except Exception:
    enryptImage = None

KEY_PHRASE = "encryptionkey"
ALGORITHM_DEFAULT = "AES_CTR"

ui_state = ImageUiState()
ALGORITHM_SELECTION: tk.StringVar | None = None
ALGORITHM_BUTTONS: dict[str, tk.Button] = {}


def on_encrypt(canvas: tk.Canvas) -> Path | None:
    if enryptImage is None:
        messagebox.showerror("Error", "enryptImage module not available")
        return None
    if ui_state.selected_image_path is None:
        messagebox.showwarning("No image", "Please choose or drop an image first.")
        return None

    try:
        input_path = ui_state.selected_image_path
        base_dir = Path(__file__).resolve().parent
        encrypted_dir = base_dir / "images" / "encrypted"
        encrypted_dir.mkdir(parents=True, exist_ok=True)
        encrypted_plus_one_bit_dir = base_dir / "images" / "encrypted_plus_one_bit"
        encrypted_plus_one_bit_dir.mkdir(parents=True, exist_ok=True)
        nonce_dir = base_dir / "nonce_files"
        nonce_dir.mkdir(parents=True, exist_ok=True)

        # Determine algorithm from UI selector if available
        try:
            alg = ALGORITHM_SELECTION.get()
        except Exception:
            alg = ALGORITHM_DEFAULT

        prefix = alg.lower()
        output_path = encrypted_dir / f"{prefix}_enc_{input_path.name}"

        enryptImage.encrypt_image(input_path, output_path, KEY_PHRASE, algorithm=alg)

        input_path_2 = add_one_bit_to_pixel(input_path)
        output_path_2 = encrypted_plus_one_bit_dir / f"{prefix}_enc_{input_path.name}"

        enryptImage.encrypt_image(
            input_path_2, output_path_2, KEY_PHRASE, algorithm=alg
        )
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

        img = Image.open(encrypted_path).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)

        sections = [
            (
                "NPCR results",
                format_result(
                    "NPCR results",
                    number_of_pixel_change_rate(e_str, eb_str),
                    percent=True,
                ).replace("NPCR results\n", "", 1),
            ),
            (
                "UACI results",
                format_result(
                    "UACI results",
                    unified_average_changing_intensity(e_str, eb_str),
                    percent=True,
                ).replace("UACI results\n", "", 1),
            ),
            (
                "Correlation results",
                format_correlation_result(
                    "Correlation results", e_str, p_str, eb_str
                ).replace("Correlation results\n", "", 1),
            ),
            (
                "MSE results",
                format_result(
                    "MSE results",
                    mean_squared_error(e_str, p_str),
                    percent=False,
                ).replace("MSE results\n", "", 1),
            ),
            (
                "PSNR results",
                format_result(
                    "PSNR results",
                    peak_signal_to_noise_ratio(e_str, p_str),
                    percent=False,
                ).replace("PSNR results\n", "", 1),
            ),
            (
                "SSIM results",
                format_result(
                    "SSIM results",
                    structural_similarity(e_str, p_str),
                    percent=False,
                ).replace("SSIM results\n", "", 1),
            ),
            (
                "Entropy results",
                format_entropy_result(
                    "Entropy results", pixel_entropy_with_blocks(arr)
                ).replace("Entropy results\n", "", 1),
            ),
            (
                "Histogram results",
                format_histogram_result(
                    "Histogram results", image_histogram(arr)
                ).replace("Histogram results\n", "", 1),
            ),
        ]

        set_result_text(ui_state, format_complete_analysis_result(sections))
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
    if enryptImage is None:
        messagebox.showerror("Error", "enryptImage module not available")
        return
    if ui_state.selected_image_path is None:
        messagebox.showwarning("No image", "Please choose or drop an image first.")
        return

    try:
        input_path = ui_state.selected_image_path

        base_dir = Path(__file__).resolve().parent
        decrypted_dir = base_dir / "images" / "decrypted"
        decrypted_dir.mkdir(parents=True, exist_ok=True)
        # Decide algorithm from filename prefix if possible
        stem = input_path.stem
        if stem.lower().startswith("aes-cbc_") or stem.lower().startswith("aes_cbc_"):
            detected_alg = "AES_CBC"
            print("Detected AES-CBC from filename prefix")
        elif stem.lower().startswith("aes-gcm_") or stem.lower().startswith("aes_gcm_"):
            detected_alg = "AES_GCM"
            print("Detected AES-GCM from filename prefix")
        elif stem.lower().startswith("chacha20_") or stem.lower().startswith("chacha_"):
            detected_alg = "CHACHA20"
            print("Detected ChaCha20 from filename prefix")
        elif stem.lower().startswith("aes-ccm_") or stem.lower().startswith("aes_ccm_"):
            detected_alg = "AES_CCM"
            print("Detected AES-CCM from filename prefix")
        elif stem.lower().startswith("aes-ctr_") or stem.lower().startswith("aes_ctr_"):
            detected_alg = "AES_CTR"
        elif stem.lower().startswith("des_"):
            detected_alg = "DES"
        elif stem.lower().startswith("custom-aes_") or stem.lower().startswith(
            "custom_aes_"
        ):
            detected_alg = "CUSTOM_AES"
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
        output_path = decrypted_dir / output_name

        enryptImage.decrypt_image(
            input_path, output_path, KEY_PHRASE, algorithm=detected_alg
        )
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
            initialdir=str(Path(__file__).resolve().parent / "images" / "plain"),
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
    encrypt_btn.pack(side="left", padx=6)
    encrypt_analysis_btn.pack(side="left", padx=6)
    decrypt_btn.pack(side="left", padx=6)


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
    make_btn("AES_GCM")
    make_btn("AES_CCM")
    make_btn("DES")
    make_btn("Custom_AES")
    make_btn("ChaCha20")

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


def build_ui() -> None:
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
            import analyse_ui as analyse_ui

            analyse_ui.build_ui()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open Analyse UI:\n{e}")

    bottom = tk.Frame(frame)
    bottom.pack(side="bottom", fill="x", pady=(8, 0))
    switch_btn = tk.Button(bottom, text="Analyse", width=12, command=switch_to_analyse)
    switch_btn.pack()

    ui_state.result_frame.pack_forget()

    root.mainloop()


def main() -> None:
    build_ui()


if __name__ == "__main__":
    main()
