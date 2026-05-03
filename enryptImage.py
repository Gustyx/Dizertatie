import hashlib
import os
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from imagePixels import extract_pixels, reconstruct_image


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
    pass

if __name__ == "__main__":
    main()
