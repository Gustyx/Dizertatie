import hashlib
import numpy as np
from cryptography.hazmat.decrepit.ciphers import algorithms as decrepit_alg
from cryptography.hazmat.primitives.ciphers import Cipher, modes


def derive_des_key(key_phrase: str) -> bytes:
    """Derive a 24-byte TripleDES key from a passphrase (not secure for production)."""

    # Use SHA-256 and take the first 24 bytes as the 3DES key
    return hashlib.sha256(key_phrase.encode("utf-8")).digest()[:24]


def des_ctr_transform(pixel_array: np.ndarray, key: bytes, nonce: bytes) -> np.ndarray:
    """Encrypt or decrypt pixel data with TripleDES-CTR (manual CTR using ECB).

    Always uses TripleDES for the DES option.
    """

    flat_pixels = pixel_array.astype(np.uint8).tobytes()

    algo = decrepit_alg.TripleDES(key)

    # Manual CTR using ECB: encrypt successive counter blocks and XOR with data
    block_size = algo.block_size // 8
    if len(nonce) != block_size:
        raise ValueError(
            f"Invalid nonce length. Expected {block_size} bytes for 3DES CTR."
        )

    cipher = Cipher(algo, modes.ECB())
    transform = cipher.encryptor()
    transformed_bytes = bytearray(len(flat_pixels))
    counter = int.from_bytes(nonce, "big")
    mask = (1 << (block_size * 8)) - 1

    for i in range(0, len(flat_pixels), block_size):
        counter_block = (counter & mask).to_bytes(block_size, "big")
        keystream = transform.update(counter_block)
        chunk = flat_pixels[i : i + block_size]
        for j in range(len(chunk)):
            transformed_bytes[i + j] = chunk[j] ^ keystream[j]
        counter = (counter + 1) & mask

    try:
        transform.finalize()
    except Exception:
        pass

    transformed_array = np.frombuffer(bytes(transformed_bytes), dtype=np.uint8)
    return transformed_array.reshape(pixel_array.shape)
