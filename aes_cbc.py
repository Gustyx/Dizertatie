import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding


def aes_cbc_encrypt_bytes(data: bytes, key: bytes, iv: bytes) -> bytes:
    """Encrypt bytes using AES-CBC with PKCS7 padding."""

    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded = padder.update(data) + padder.finalize()
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    encryptor = cipher.encryptor()
    encrypted = encryptor.update(padded) + encryptor.finalize()
    return encrypted


def aes_cbc_decrypt_bytes(encrypted: bytes, key: bytes, iv: bytes) -> bytes:
    """Decrypt AES-CBC bytes and remove PKCS7 padding."""

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    padded = decryptor.update(encrypted) + decryptor.finalize()
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    data = unpadder.update(padded) + unpadder.finalize()
    return data


def aes_cbc_encrypt_array(pixel_array: np.ndarray, key: bytes, iv: bytes) -> np.ndarray:
    """Encrypt a numpy pixel array with AES-CBC and return encrypted bytes as a 1-D uint8 array.

    Note: AES-CBC with padding will typically change the byte length. The returned array
    is a flat 1-D array of encrypted bytes; callers should store the original shape/length
    if they need to reconstruct the image after decryption.
    """

    flat = pixel_array.astype(np.uint8).tobytes()
    encrypted = aes_cbc_encrypt_bytes(flat, key, iv)
    return np.frombuffer(encrypted, dtype=np.uint8)


def aes_cbc_decrypt_array(encrypted_array: np.ndarray, key: bytes, iv: bytes) -> np.ndarray:
    """Decrypt a 1-D uint8 encrypted array produced by `aes_cbc_encrypt_array` and
    return the original plaintext bytes as a 1-D uint8 array (unpadded).
    """

    encrypted_bytes = encrypted_array.tobytes()
    decrypted = aes_cbc_decrypt_bytes(encrypted_bytes, key, iv)
    return np.frombuffer(decrypted, dtype=np.uint8)
