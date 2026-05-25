from pathlib import Path
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk
import numpy as np

from correlation import correlation_between_images
from entropy import pixel_entropy
from histogram import image_histogram
from nprc import number_of_pixel_change_rate
from psnr import mean_squared_error, peak_signal_to_noise_ratio
from uaci import unified_average_changing_intensity
from ssim import structural_similarity

ANALYSIS_DEFAULT = "NPCR"

selected_image_path: Path | None = None
selected_image_tk = None
action_buttons_frame: tk.Frame | None = None
result_text: tk.StringVar | None = None
result_widget: tk.Text | None = None
analysis_selection: tk.StringVar | None = None
analysis_buttons: dict[str, tk.Button] = {}
result_frame: tk.LabelFrame | None = None

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

    canvas.delete("all")
    canvas.config(width=selected_image_tk.width(), height=selected_image_tk.height())
    canvas.create_image(0, 0, anchor="nw", image=selected_image_tk)
    selected_image_path = path

    if action_buttons_frame is not None and not action_buttons_frame.winfo_ismapped():
        action_buttons_frame.pack(pady=(6, 0))
    if result_frame is not None and not result_frame.winfo_ismapped():
        result_frame.pack(fill="both", expand=True, pady=(8, 0))

    try:
        top = canvas.winfo_toplevel()
        top.update_idletasks()
        new_w = selected_image_tk.width() + 40
        new_h = selected_image_tk.height() + 180
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
        outline="#666",
        width=2,
        dash=(4, 4),
        tags=("placeholder",),
    )
    canvas.create_text(
        width // 2,
        height // 2,
        text=text,
        fill="#666",
        tags=("placeholder",),
    )


def set_result_text(text: str) -> None:
    if result_widget is None:
        return

    result_widget.config(state="normal")
    result_widget.delete("1.0", tk.END)
    result_widget.insert("1.0", text)
    result_widget.config(state="disabled")
    result_widget.yview_moveto(0)


def choose_image(canvas: tk.Canvas) -> None:
    base_dir = Path(__file__).resolve().parent
    encrypted_dir = base_dir / "images" / "encrypted"
    file = filedialog.askopenfilename(
        title="Select encrypted image",
        initialdir=str(encrypted_dir) if encrypted_dir.exists() else None,
        filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif")],
    )
    if not file:
        return
    display_image(Path(file), canvas)


def on_drop(event, canvas: tk.Canvas) -> None:
    try:
        parts = canvas.master.tk.splitlist(event.data)
        if not parts:
            return
        display_image(Path(parts[0]), canvas)
    except Exception:
        return


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
            # chi-square
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

    return "\n".join(lines)


def format_complete_analysis_result(result_sections: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"{title}\n{body}" for title, body in result_sections)


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

        sections = []
        sections.append(
            (
                "NPCR results",
                format_result(
                    "NPCR results",
                    number_of_pixel_change_rate(e_str, eb_str),
                    percent=True,
                ).replace("NPCR results\n", "", 1),
            )
        )
        sections.append(
            (
                "UACI results",
                format_result(
                    "UACI results",
                    unified_average_changing_intensity(e_str, eb_str),
                    percent=True,
                ).replace("UACI results\n", "", 1),
            )
        )
        sections.append(
            (
                "Correlation results",
                format_result(
                    "Correlation results",
                    correlation_between_images(e_str, eb_str),
                    percent=False,
                ).replace("Correlation results\n", "", 1),
            )
        )
        sections.append(
            (
                "MSE results",
                format_result(
                    "MSE results",
                    mean_squared_error(e_str, p_str),
                    percent=False,
                ).replace("MSE results\n", "", 1),
            )
        )
        sections.append(
            (
                "PSNR results",
                format_result(
                    "PSNR results",
                    peak_signal_to_noise_ratio(e_str, p_str),
                    percent=False,
                ).replace("PSNR results\n", "", 1),
            )
        )
        sections.append(
            (
                "SSIM results",
                format_result(
                    "SSIM results",
                    structural_similarity(e_str, p_str),
                    percent=False,
                ).replace("SSIM results\n", "", 1),
            )
        )
        sections.append(
            (
                "Entropy results",
                format_result(
                    "Entropy results", pixel_entropy(arr), percent=False
                ).replace("Entropy results\n", "", 1),
            )
        )
        sections.append(
            (
                "Histogram results",
                format_histogram_result(
                    "Histogram results", image_histogram(arr)
                ).replace("Histogram results\n", "", 1),
            )
        )

        set_result_text(format_complete_analysis_result(sections))
    except Exception as exc:
        messagebox.showerror(
            "Error", f"Analysis failed:\n{exc}\n\n{traceback.format_exc()}"
        )


def on_analyse() -> None:
    global selected_image_path
    if selected_image_path is None:
        messagebox.showwarning(
            "No image", "Please choose or drop an encrypted image first."
        )
        return

    try:
        metric = analysis_selection.get()
    except Exception:
        metric = ANALYSIS_DEFAULT

    try:
        encrypted_path = selected_image_path
        if metric == "Entropy":
            # Convert to RGB numpy array so entropy() accepts it regardless
            # of source image mode (RGBA, P, etc.).
            img = Image.open(encrypted_path).convert("RGB")
            arr = np.asarray(img, dtype=np.uint8)
            result = pixel_entropy(arr)
            text = format_result("Entropy results", result, percent=False)
        elif metric == "Histogram":
            img = Image.open(encrypted_path).convert("RGB")
            arr = np.asarray(img, dtype=np.uint8)
            result = image_histogram(arr)
            text = format_histogram_result("Histogram results", result)
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
                text = format_result("NPCR results", result, percent=True)
            elif metric == "UACI":
                result = unified_average_changing_intensity(e_str, eb_str)
                text = format_result("UACI results", result, percent=True)
            elif metric == "Correlation":
                result = correlation_between_images(e_str, eb_str)
                text = format_result("Correlation results", result, percent=False)
            elif metric == "PSNR":
                result = peak_signal_to_noise_ratio(e_str, p_str)
                text = format_result("PSNR results", result, percent=False)
            elif metric == "SSIM":
                result = structural_similarity(e_str, p_str)
                text = format_result("SSIM results", result, percent=False)
            elif metric == "MSE":
                result = mean_squared_error(e_str, p_str)
                text = format_result("MSE results", result, percent=False)
            else:
                raise ValueError(f"Unsupported analysis metric: {metric}")

        set_result_text(text)
    except Exception as exc:
        messagebox.showerror(
            "Error", f"Analysis failed:\n{exc}\n\n{traceback.format_exc()}"
        )


def build_analysis_menu(parent) -> None:
    global analysis_selection, analysis_buttons
    analysis_selection = tk.StringVar(value=ANALYSIS_DEFAULT)

    tk.Label(parent, text="Analysis", font=(None, 10, "bold")).pack(pady=(0, 6))

    def make_btn(display_name: str) -> None:
        btn = tk.Button(parent, text=display_name, width=14)

        def on_click() -> None:
            analysis_selection.set(display_name)
            for name, button in analysis_buttons.items():
                if name == display_name:
                    button.config(relief="sunken", bg="#cfe")
                else:
                    button.config(relief="raised", bg=parent.cget("bg"))

        btn.config(command=on_click)
        btn.pack(pady=4, fill="x")
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
    analysis_buttons[ANALYSIS_DEFAULT].config(relief="sunken", bg="#cfe")


def build_image_box(frame) -> tk.Canvas:
    img_box = tk.Canvas(
        frame, bg="#ffffff", relief="flat", width=400, height=240, highlightthickness=0
    )
    reset_placeholder(img_box, "Drop or choose an encrypted image")
    img_box.pack(pady=(0, 8), anchor="n")

    if DND_AVAILABLE:
        try:
            img_box.drop_target_register(DND_FILES)
            img_box.dnd_bind("<<Drop>>", lambda e: on_drop(e, img_box))
        except Exception:
            pass

    return img_box


def build_ui() -> None:
    global action_buttons_frame, result_frame

    if DND_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()

    root.title("Image Analysis")
    root.minsize(520, 520)

    frame = tk.Frame(root, padx=10, pady=10)
    frame.pack(expand=True, fill="both")

    sidebar = tk.Frame(frame)
    sidebar.pack(side="left", fill="y", padx=(0, 8))
    build_analysis_menu(sidebar)

    content = tk.Frame(frame)
    content.pack(side="left", fill="both", expand=True)

    img_box = build_image_box(content)

    controls = tk.Frame(content)
    controls.pack()

    choose_btn = tk.Button(
        controls,
        text="Choose encrypted image",
        command=lambda: choose_image(img_box),
    )
    choose_btn.pack()

    action_buttons_frame = tk.Frame(controls)

    analyse_btn = tk.Button(
        action_buttons_frame, text="Analyse", width=12, command=on_analyse
    )
    analyse_btn.pack(pady=(6, 0))

    complete_btn = tk.Button(
        action_buttons_frame,
        text="Complete Analysis",
        width=16,
        command=on_complete_analysis,
    )
    complete_btn.pack(pady=(6, 0))

    result_frame = tk.LabelFrame(content, text="Results")

    global result_widget
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
        "Choose an encrypted image, then run NPCR, UACI, Correlation, MSE, PSNR, Entropy, Histogram, or Complete Analysis.",
    )
    result_widget.config(state="disabled")

    # Hide analysis controls until an image is selected.
    action_buttons_frame.pack_forget()
    result_frame.pack_forget()

    # Mode switch button to open encryption UI
    def switch_to_encrypt():
        try:
            root.destroy()
        except Exception:
            pass
        try:
            import encrypt_ui as encrypt_ui

            encrypt_ui.build_ui()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open Encrypt UI:\n{e}")

    bottom = tk.Frame(frame)
    bottom.pack(side="bottom", fill="x", pady=(8, 0))
    switch_btn = tk.Button(
        bottom, text="Encryption", width=12, command=switch_to_encrypt
    )
    switch_btn.pack()

    root.mainloop()


def main() -> None:
    build_ui()


if __name__ == "__main__":
    main()
