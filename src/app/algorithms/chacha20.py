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
    """Encrypt bytes using the ChaCha20 stream cipher.

    This uses the low-level ChaCha20 primitive (non-AEAD). The same function
    can be used to decrypt since ChaCha20 is a stream cipher (XOR with keystream).

    `key` must be 32 bytes and `nonce` must be 16 bytes (as accepted by
    cryptography's ChaCha20 implementation).
    """

    _validate_key_nonce(key, nonce)
    cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None)
    encryptor = cipher.encryptor()
    return encryptor.update(data) + encryptor.finalize()


def chacha20_decrypt_bytes(data: bytes, key: bytes, nonce: bytes) -> bytes:
    """Decrypt bytes encrypted with `chacha20_encrypt_bytes`.

    Since ChaCha20 is a stream cipher, decryption is identical to encryption.
    """

    return chacha20_encrypt_bytes(data, key, nonce)


def chacha20_encrypt_array(
    pixel_array: np.ndarray, key: bytes, nonce: bytes
) -> np.ndarray:
    """Encrypt a numpy pixel array with ChaCha20 and return ciphertext as a
    1-D uint8 array.
    """

    flat = pixel_array.astype(np.uint8).tobytes()
    encrypted = chacha20_encrypt_bytes(flat, key, nonce)
    return np.frombuffer(encrypted, dtype=np.uint8)


def chacha20_decrypt_array(
    encrypted_array: np.ndarray, key: bytes, nonce: bytes
) -> np.ndarray:
    """Decrypt a 1-D uint8 encrypted array produced by
    `chacha20_encrypt_array` and return the original plaintext bytes as a
    1-D uint8 array.
    """

    encrypted_bytes = encrypted_array.tobytes()
    decrypted = chacha20_decrypt_bytes(encrypted_bytes, key, nonce)
    return np.frombuffer(decrypted, dtype=np.uint8)
