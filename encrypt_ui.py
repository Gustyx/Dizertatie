from pathlib import Path
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

import numpy as np

from imagePixels import add_one_bit_to_pixel
from correlation import correlation_between_images
from entropy import pixel_entropy_with_blocks
from histogram import image_histogram
from nprc import number_of_pixel_change_rate
from psnr import mean_squared_error, peak_signal_to_noise_ratio
from uaci import unified_average_changing_intensity
from analyse_ui import (
    format_complete_analysis_result,
    format_entropy_result,
    format_histogram_result,
    format_result,
)
from ssim import structural_similarity

try:
    import enryptImage
except Exception:
    enryptImage = None

KEY_PHRASE = "encryptionkey"
ALGORITHM_DEFAULT = "AES_CTR"

# state for selected image
selected_image_path: Path | None = None
selected_image_tk = None
action_buttons_frame: tk.Frame | None = None
result_frame: tk.LabelFrame | None = None
result_widget: tk.Text | None = None
ALGORITHM_SELECTION: tk.StringVar | None = None
ALGORITHM_BUTTONS: dict = {}
ALGORITHM_BUTTON_VALUES: dict = {}

# try to enable drag-and-drop support (optional)
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    DND_AVAILABLE = True
except Exception:
    DND_AVAILABLE = False


def display_image(path: Path, canvas: tk.Canvas, max_size=(800, 600)) -> None:
    global selected_image_tk, selected_image_path
    img = Image.open(path)
    img.thumbnail(max_size)
    selected_image_tk = ImageTk.PhotoImage(img)

    # adjust canvas to image pixel size and draw image at top-left
    canvas.delete("all")
    canvas.config(width=selected_image_tk.width(), height=selected_image_tk.height())
    canvas.create_image(0, 0, anchor="nw", image=selected_image_tk)
    selected_image_path = path

    if action_buttons_frame is not None and not action_buttons_frame.winfo_ismapped():
        action_buttons_frame.pack(pady=(6, 0))
    if result_frame is not None and not result_frame.winfo_ismapped():
        result_frame.pack(fill="both", expand=True, pady=(8, 0))

    # Resize top-level window to fit image (only enlarge, don't shrink)
    try:
        top = canvas.winfo_toplevel()
        top.update_idletasks()
        new_w = selected_image_tk.width() + 40
        new_h = selected_image_tk.height() + 320

        cur_w = top.winfo_width()
        cur_h = top.winfo_height()

        # If the image requires a larger window, enlarge; otherwise leave as-is
        if new_w > cur_w or new_h > cur_h:
            target_w = max(new_w, cur_w)
            target_h = max(new_h, cur_h)
            top.geometry(f"{target_w}x{target_h}")
    except Exception:
        pass


def set_result_text(text: str) -> None:
    if result_widget is None:
        return

    result_widget.config(state="normal")
    result_widget.delete("1.0", tk.END)
    result_widget.insert("1.0", text)
    result_widget.config(state="disabled")
    result_widget.yview_moveto(0)


def choose_image(canvas: tk.Canvas) -> None:
    file = filedialog.askopenfilename(
        title="Select image",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif")],
    )
    if not file:
        return
    display_image(Path(file), canvas)


def on_drop(event, canvas: tk.Canvas) -> None:
    # event.data may contain a Tcl list of filenames
    try:
        parts = canvas.master.tk.splitlist(event.data)
        if not parts:
            return
        file = parts[0]
        display_image(Path(file), canvas)
    except Exception:
        return


def on_encrypt(canvas: tk.Canvas) -> Path | None:
    global selected_image_path
    if enryptImage is None:
        messagebox.showerror("Error", "enryptImage module not available")
        return None
    if selected_image_path is None:
        messagebox.showwarning("No image", "Please choose or drop an image first.")
        return None

    try:
        input_path = selected_image_path
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
            display_image(output_path, canvas)
        except Exception:
            pass
        return output_path
    except Exception as exc:
        messagebox.showerror(
            "Error", f"Encryption failed:\n{exc}\n\n{traceback.format_exc()}"
        )
        return None


def on_complete_analysis() -> None:
    global selected_image_path
    if selected_image_path is None:
        messagebox.showwarning(
            "No image", "Please choose or drop an encrypted image first."
        )
        return

    try:
        encrypted_path = selected_image_path
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
                format_result(
                    "Correlation results",
                    correlation_between_images(e_str, eb_str),
                    percent=False,
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

        set_result_text(format_complete_analysis_result(sections))
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
    global selected_image_path
    if enryptImage is None:
        messagebox.showerror("Error", "enryptImage module not available")
        return
    if selected_image_path is None:
        messagebox.showwarning("No image", "Please choose or drop an image first.")
        return

    try:
        input_path = selected_image_path

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
            display_image(output_path, canvas)
        except Exception:
            pass
    except Exception as exc:
        messagebox.showerror(
            "Error", f"Decryption failed:\n{exc}\n\n{traceback.format_exc()}"
        )


def build_image_box(frame) -> None:
    # image display area using a Canvas (starts with placeholder size)
    img_box = tk.Canvas(
        frame, bg="#ffffff", relief="flat", width=400, height=240, highlightthickness=0
    )
    # visible dashed border and placeholder text so users know where to drop
    padding = 8
    w, h = 400, 240
    img_box.create_rectangle(
        padding,
        padding,
        w - padding,
        h - padding,
        outline="#666",
        width=2,
        dash=(4, 4),
        tags=("placeholder",),
    )
    img_box.create_text(
        w // 2,
        h // 2,
        text="Drop or choose an image",
        fill="#666",
        tags=("placeholder",),
    )
    img_box.pack(pady=(0, 8), anchor="n")

    # enable drag-and-drop if available
    if DND_AVAILABLE:
        try:
            img_box.drop_target_register(DND_FILES)
            img_box.dnd_bind("<<Drop>>", lambda e: on_drop(e, img_box))
        except Exception:
            pass

    return img_box


def build_buttons(frame, img_box) -> None:
    choose_btn = tk.Button(
        frame, text="Choose image", command=lambda: choose_image(img_box)
    )
    choose_btn.pack()

    global action_buttons_frame
    action_buttons_frame = tk.Frame(frame)

    encrypt_btn = tk.Button(
        action_buttons_frame,
        text="Encrypt",
        width=12,
        command=lambda: on_encrypt(img_box),
    )
    encrypt_analysis_btn = tk.Button(
        action_buttons_frame,
        text="Encrypt + Analysis",
        width=18,
        command=lambda: on_encrypt_and_analysis(img_box),
    )
    decrypt_btn = tk.Button(
        action_buttons_frame,
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

    label = tk.Label(parent, text="Algorithms", font=(None, 10, "bold"))
    label.pack(pady=(0, 6))

    def make_btn(display_name: str, value: str | None = None):
        if value is None:
            value = display_name

        btn = tk.Button(parent, text=display_name, width=12)

        def on_click():
            set_algorithm(display_name, value)

        btn.config(command=on_click)
        btn.pack(pady=4, fill="x")
        ALGORITHM_BUTTONS[display_name] = btn
        ALGORITHM_BUTTON_VALUES[display_name] = value

    make_btn("AES-CTR", "AES_CTR")
    make_btn("AES-CBC", "AES_CBC")
    make_btn("AES-GCM", "AES_GCM")
    make_btn("AES-CCM", "AES_CCM")
    make_btn("DES")
    make_btn("Custom AES", "CUSTOM_AES")

    make_btn("ChaCha20", "CHACHA20")

    # initialize visuals
    def set_algorithm(display_name: str, value: str | None = None):
        if value is None:
            value = ALGORITHM_BUTTON_VALUES.get(display_name, display_name)

        ALGORITHM_SELECTION.set(value)
        for n, b in ALGORITHM_BUTTONS.items():
            if n == display_name:
                b.config(relief="sunken", bg="#cfe")
            else:
                b.config(relief="raised", bg=parent.cget("bg"))

    set_algorithm(ALGORITHM_DEFAULT)


def build_ui() -> None:
    # create root (use DnD root if available)
    if DND_AVAILABLE:
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

    img_box = build_image_box(content)

    controls_frame = tk.Frame(content)
    controls_frame.pack()

    build_buttons(controls_frame, img_box)

    global result_frame, result_widget
    result_frame = tk.LabelFrame(content, text="Results")
    result_container = tk.Frame(result_frame)
    result_container.pack(fill="both", expand=True)

    result_scrollbar = tk.Scrollbar(result_container, orient="vertical")
    result_scrollbar.pack(side="right", fill="y")

    result_widget = tk.Text(
        result_container,
        wrap="word",
        height=14,
        padx=8,
        pady=8,
        yscrollcommand=result_scrollbar.set,
        relief="flat",
        borderwidth=0,
    )
    result_widget.pack(side="left", fill="both", expand=True)
    result_scrollbar.config(command=result_widget.yview)
    result_widget.insert(
        "1.0",
        "Choose an image, then run Encrypt + Analysis to see all metrics in one scrollable results box.",
    )
    result_widget.config(state="disabled")

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

    result_frame.pack_forget()

    root.mainloop()


def main() -> None:
    build_ui()


if __name__ == "__main__":
    main()
