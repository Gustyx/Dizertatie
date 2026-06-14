import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms


def _validate_key_nonce(key: bytes, nonce: bytes) -> None:
    if not isinstance(key, (bytes, bytearray)):
        raise TypeError("key must be bytes")
    if not isinstance(nonce, (bytes, bytearray)):
        raise TypeError("nonce must be bytes")
    if len(key) != 32:
        raise ValueError("ChaCha20 key must be 32 bytes")
    if len(nonce) != 16:
        raise ValueError("ChaCha20 nonce must be 16 bytes")


def chacha20_encrypt_bytes(data: bytes, key: bytes, nonce: bytes) -> bytes:
    _validate_key_nonce(key, nonce)
    cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None)
    encryptor = cipher.encryptor()
    return encryptor.update(data) + encryptor.finalize()


def chacha20_decrypt_bytes(data: bytes, key: bytes, nonce: bytes) -> bytes:
    return chacha20_encrypt_bytes(data, key, nonce)


def chacha20_encrypt_array(
    pixel_array: np.ndarray, key: bytes, nonce: bytes
) -> np.ndarray:
    flat = pixel_array.astype(np.uint8).tobytes()
    encrypted = chacha20_encrypt_bytes(flat, key, nonce)
    return np.frombuffer(encrypted, dtype=np.uint8)


def chacha20_decrypt_array(
    encrypted_array: np.ndarray, key: bytes, nonce: bytes
) -> np.ndarray:
    encrypted_bytes = encrypted_array.tobytes()
    decrypted = chacha20_decrypt_bytes(encrypted_bytes, key, nonce)
    return np.frombuffer(decrypted, dtype=np.uint8)
