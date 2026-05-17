import hashlib
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


def derive_aes_key(key_phrase: str) -> bytes:
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
