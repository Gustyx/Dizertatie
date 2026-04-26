import hashlib
import os
from pathlib import Path

import numpy as np
from PIL import Image
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


def extract_pixels(image_path: Path) -> tuple[np.ndarray, str]:
    """Load an image and return its pixel array plus mode."""

    with Image.open(image_path) as img:
        return np.array(img), img.mode


def reconstruct_image(pixel_array: np.ndarray, mode: str, output_path: Path) -> None:
    """Create and save an image from a pixel array using the original mode."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(pixel_array.astype(np.uint8), mode=mode)
    image.save(output_path)


def derive_key(key_phrase: str) -> bytes:
    """Derive a 256-bit AES key from a passphrase."""

    return hashlib.sha256(key_phrase.encode("utf-8")).digest()


def aes_ctr_transform(pixel_array: np.ndarray, key: bytes, nonce: bytes) -> np.ndarray:
    """Encrypt or decrypt pixel data with AES-CTR (same operation both ways)."""

    flat_pixels = pixel_array.astype(np.uint8).tobytes()
    cipher = Cipher(algorithms.AES(key), modes.CTR(nonce))
    transform = cipher.encryptor()
    transformed_bytes = transform.update(flat_pixels) + transform.finalize()
    transformed_array = np.frombuffer(transformed_bytes, dtype=np.uint8)
    return transformed_array.reshape(pixel_array.shape)


def encrypt_image(
    input_path: Path, output_path: Path, nonce_path: Path, key_phrase: str
) -> None:
    """Encrypt image pixels with AES-CTR and save encrypted image plus nonce file."""

    pixels, mode = extract_pixels(input_path)
    key = derive_key(key_phrase)
    nonce = os.urandom(16)

    encrypted_pixels = aes_ctr_transform(pixels, key, nonce)
    reconstruct_image(encrypted_pixels, mode, output_path)

    nonce_path.parent.mkdir(parents=True, exist_ok=True)
    nonce_path.write_bytes(nonce)


def decrypt_image(
    input_path: Path, output_path: Path, nonce_path: Path, key_phrase: str
) -> None:
    """Decrypt AES-CTR encrypted image pixels using the stored nonce file."""

    if not nonce_path.exists():
        raise FileNotFoundError(f"Nonce file not found: {nonce_path}")

    pixels, mode = extract_pixels(input_path)
    key = derive_key(key_phrase)
    nonce = nonce_path.read_bytes()

    if len(nonce) != 16:
        raise ValueError("Invalid nonce length. Expected 16 bytes for AES-CTR.")

    decrypted_pixels = aes_ctr_transform(pixels, key, nonce)
    reconstruct_image(decrypted_pixels, mode, output_path)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    key_phrase = "encryptionkey"

    plain_dir = base_dir / "images" / "plain"
    encrypted_dir = base_dir / "images" / "encrypted"
    decrypted_dir = base_dir / "images" / "decrypted"

    print("Do you want to 1. encrypt or 2. decrypt?")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        image_files = [
            p
            for p in plain_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"}
        ]
        if not image_files:
            raise FileNotFoundError(f"No image files found in: {plain_dir}")

        input_path = image_files[0]
        output_path = encrypted_dir / f"aes_{input_path.name}"
        nonce_path = encrypted_dir / f"aes_{input_path.stem}.nonce"

        encrypt_image(input_path, output_path, nonce_path, key_phrase)
        print(f"Encrypted image saved to: {output_path}")
        print(f"Nonce saved to: {nonce_path}")
        return

    if choice == "2":
        image_files = [
            p
            for p in encrypted_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"}
            and p.name.startswith("aes_")
        ]
        if not image_files:
            raise FileNotFoundError(
                f"No AES-encrypted image files found in: {encrypted_dir}"
            )

        input_path = image_files[0]
        nonce_path = encrypted_dir / f"{input_path.stem}.nonce"
        output_name = input_path.name.replace("aes_", "decrypted_", 1)
        output_path = decrypted_dir / output_name

        decrypt_image(input_path, output_path, nonce_path, key_phrase)
        print(f"Decrypted image saved to: {output_path}")
        return

    print("Invalid choice. Please run again and enter 1 or 2.")


if __name__ == "__main__":
    main()
