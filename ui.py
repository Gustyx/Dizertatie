from pathlib import Path
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

try:
    import enryptImage
except Exception:
    enryptImage = None

KEY_PHRASE = "encryptionkey"

# state for selected image
selected_image_path: Path | None = None
selected_image_tk = None
action_buttons_frame: tk.Frame | None = None

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

    # Resize top-level window to fit image (only enlarge, don't shrink)
    try:
        top = canvas.winfo_toplevel()
        top.update_idletasks()
        new_w = selected_image_tk.width() + 40
        new_h = selected_image_tk.height() + 140

        cur_w = top.winfo_width()
        cur_h = top.winfo_height()

        # If the image requires a larger window, enlarge; otherwise leave as-is
        if new_w > cur_w or new_h > cur_h:
            target_w = max(new_w, cur_w)
            target_h = max(new_h, cur_h)
            top.geometry(f"{target_w}x{target_h}")
    except Exception:
        pass


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


def on_encrypt(canvas: tk.Canvas) -> None:
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
        encrypted_dir = base_dir / "images" / "encrypted"
        encrypted_dir.mkdir(parents=True, exist_ok=True)

        output_path = encrypted_dir / f"aes_{input_path.name}"
        nonce_path = encrypted_dir / f"aes_{input_path.stem}.nonce"

        enryptImage.encrypt_image(input_path, output_path, nonce_path, KEY_PHRASE)
        # replace displayed image with encrypted output
        try:
            display_image(output_path, canvas)
        except Exception:
            pass
        messagebox.showinfo(
            "Encrypted",
            f"Encrypted saved:\n{output_path}\nNonce:\n{nonce_path}",
        )
    except Exception as exc:
        messagebox.showerror(
            "Error", f"Encryption failed:\n{exc}\n\n{traceback.format_exc()}"
        )


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
        encrypted_dir = input_path.parent
        nonce_path = encrypted_dir / f"{input_path.stem}.nonce"

        base_dir = Path(__file__).resolve().parent
        decrypted_dir = base_dir / "images" / "decrypted"
        decrypted_dir.mkdir(parents=True, exist_ok=True)
        output_name = input_path.name.replace("aes_", "decrypted_", 1)
        output_path = decrypted_dir / output_name

        enryptImage.decrypt_image(input_path, output_path, nonce_path, KEY_PHRASE)
        # replace displayed image with decrypted output
        try:
            display_image(output_path, canvas)
        except Exception:
            pass
        messagebox.showinfo("Decrypted", f"Decrypted saved:\n{output_path}")
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
    decrypt_btn = tk.Button(
        action_buttons_frame,
        text="Decrypt",
        width=12,
        command=lambda: on_decrypt(img_box),
    )
    encrypt_btn.pack(side="left", padx=6)
    decrypt_btn.pack(side="left", padx=6)



def build_ui() -> None:
    # create root (use DnD root if available)
    if DND_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()

    root.title("Image AES")
    root.minsize(320, 240)

    frame = tk.Frame(root, padx=10, pady=10)
    frame.pack(expand=True, fill="both")

    img_box = build_image_box(frame)

    controls_frame = tk.Frame(frame)
    controls_frame.pack()

    build_buttons(controls_frame, img_box)

    root.mainloop()


def main() -> None:
    pass


if __name__ == "__main__":
    main()