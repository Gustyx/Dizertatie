import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESCCM


def aes_ccm_encrypt_bytes(
    data: bytes,
    key: bytes,
    nonce: bytes,
    aad: bytes | None = None,
    tag_length: int = 16,
) -> bytes:
    """Encrypt bytes using AES-CCM.

    Returns the ciphertext with the authentication tag appended (as produced
    by `AESCCM.encrypt`). `key` must be 16/24/32 bytes and `nonce` should be
    7..13 bytes for AES-CCM. `tag_length` controls the length of the
    authentication tag (commonly 16).
    """

    aesccm = AESCCM(key, tag_length=tag_length)
    return aesccm.encrypt(nonce, data, aad)


def aes_ccm_decrypt_bytes(
    encrypted: bytes,
    key: bytes,
    nonce: bytes,
    aad: bytes | None = None,
    tag_length: int = 16,
) -> bytes:
    """Decrypt AES-CCM bytes produced by `aes_ccm_encrypt_bytes`.

    Raises `cryptography.exceptions.InvalidTag` if authentication fails.
    """

    aesccm = AESCCM(key, tag_length=tag_length)
    return aesccm.decrypt(nonce, encrypted, aad)


def aes_ccm_encrypt_array(
    pixel_array: np.ndarray,
    key: bytes,
    nonce: bytes,
    aad: bytes | None = None,
    tag_length: int = 16,
) -> np.ndarray:
    """Encrypt a numpy pixel array with AES-CCM and return ciphertext||tag
    as a 1-D uint8 array.

    The returned array contains ciphertext||tag. Callers should persist the nonce
    and any AAD required for decryption.
    """

    flat = pixel_array.astype(np.uint8).tobytes()
    encrypted = aes_ccm_encrypt_bytes(flat, key, nonce, aad, tag_length)
    return np.frombuffer(encrypted, dtype=np.uint8)


def aes_ccm_decrypt_array(
    encrypted_array: np.ndarray,
    key: bytes,
    nonce: bytes,
    aad: bytes | None = None,
    tag_length: int = 16,
) -> np.ndarray:
    """Decrypt a 1-D uint8 encrypted array produced by `aes_ccm_encrypt_array`
    and return the original plaintext bytes as a 1-D uint8 array.
    """

    encrypted_bytes = encrypted_array.tobytes()
    decrypted = aes_ccm_decrypt_bytes(encrypted_bytes, key, nonce, aad, tag_length)
    return np.frombuffer(decrypted, dtype=np.uint8)
