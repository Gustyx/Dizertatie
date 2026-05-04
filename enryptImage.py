import os
from pathlib import Path

from aes import aes_ctr_transform, derive_aes_key
from des import derive_des_key, des_ctr_transform
from imagePixels import extract_pixels, reconstruct_image


def encrypt_image(
    input_path: Path,
    output_path: Path,
    nonce_path: Path,
    key_phrase: str,
    algorithm: str = "AES",
) -> None:
    """Encrypt image pixels with AES-CTR or DES-CTR and save encrypted image plus nonce file.

    algorithm -- case-insensitive string, either 'AES' or 'DES'. Defaults to 'AES'.
    """

    pixels, mode = extract_pixels(input_path)

    match algorithm.upper():
        case "AES":
            key = derive_aes_key(key_phrase)
            nonce = os.urandom(16)
            transformed = aes_ctr_transform(pixels, key, nonce)
        case "DES":
            key = derive_des_key(key_phrase)
            nonce = os.urandom(8)
            transformed = des_ctr_transform(pixels, key, nonce)
        case _:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    reconstruct_image(transformed, mode, output_path)

    nonce_path.parent.mkdir(parents=True, exist_ok=True)
    nonce_path.write_bytes(nonce)


def decrypt_image(
    input_path: Path,
    output_path: Path,
    nonce_path: Path,
    key_phrase: str,
    algorithm: str = "AES",
) -> None:
    """Decrypt AES-CTR or DES-CTR encrypted image pixels using the stored nonce file."""

    if not nonce_path.exists():
        raise FileNotFoundError(f"Nonce file not found: {nonce_path}")

    pixels, mode = extract_pixels(input_path)

    match algorithm.upper():
        case "AES":
            key = derive_aes_key(key_phrase)
            nonce = nonce_path.read_bytes()
            if len(nonce) != 16:
                raise ValueError("Invalid nonce length. Expected 16 bytes for AES-CTR.")
            transformed = aes_ctr_transform(pixels, key, nonce)
        case "DES":
            key = derive_des_key(key_phrase)
            nonce = nonce_path.read_bytes()
            if len(nonce) != 8:
                raise ValueError("Invalid nonce length. Expected 8 bytes for DES-CTR.")
            transformed = des_ctr_transform(pixels, key, nonce)
        case _:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    reconstruct_image(transformed, mode, output_path)


def main() -> None:
    pass


if __name__ == "__main__":
    main()
