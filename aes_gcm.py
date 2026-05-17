import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def aes_gcm_encrypt_bytes(
    data: bytes, key: bytes, nonce: bytes, aad: bytes | None = None
) -> bytes:
    """Encrypt bytes using AES-GCM.

    Returns the ciphertext with the authentication tag appended (as produced
    by `AESGCM.encrypt`). `key` must be 16/24/32 bytes and `nonce` is typically 12 bytes.
    """

    aesgcm = AESGCM(key)
    return aesgcm.encrypt(nonce, data, aad)


def aes_gcm_decrypt_bytes(
    encrypted: bytes, key: bytes, nonce: bytes, aad: bytes | None = None
) -> bytes:
    """Decrypt AES-GCM bytes produced by `aes_gcm_encrypt_bytes`.

    Raises `cryptography.exceptions.InvalidTag` if authentication fails.
    """

    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, encrypted, aad)


def aes_gcm_encrypt_array(
    pixel_array: np.ndarray, key: bytes, nonce: bytes, aad: bytes | None = None
) -> np.ndarray:
    """Encrypt a numpy pixel array with AES-GCM and return encrypted bytes as a 1-D uint8 array.

    The returned array contains ciphertext||tag. Callers should persist the nonce and any AAD
    required for decryption.
    """

    flat = pixel_array.astype(np.uint8).tobytes()
    encrypted = aes_gcm_encrypt_bytes(flat, key, nonce, aad)
    return np.frombuffer(encrypted, dtype=np.uint8)


def aes_gcm_decrypt_array(
    encrypted_array: np.ndarray, key: bytes, nonce: bytes, aad: bytes | None = None
) -> np.ndarray:
    """Decrypt a 1-D uint8 encrypted array produced by `aes_gcm_encrypt_array` and
    return the original plaintext bytes as a 1-D uint8 array.
    """

    encrypted_bytes = encrypted_array.tobytes()
    decrypted = aes_gcm_decrypt_bytes(encrypted_bytes, key, nonce, aad)
    return np.frombuffer(decrypted, dtype=np.uint8)
