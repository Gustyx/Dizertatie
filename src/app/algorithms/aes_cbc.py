import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding


def aes_cbc_encrypt_bytes(data: bytes, key: bytes, iv: bytes) -> bytes:
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded = padder.update(data) + padder.finalize()
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    encryptor = cipher.encryptor()
    encrypted = encryptor.update(padded) + encryptor.finalize()
    return encrypted


def aes_cbc_decrypt_bytes(encrypted: bytes, key: bytes, iv: bytes) -> bytes:
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    padded = decryptor.update(encrypted) + decryptor.finalize()
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    data = unpadder.update(padded) + unpadder.finalize()
    return data


def aes_cbc_encrypt_array(pixel_array: np.ndarray, key: bytes, iv: bytes) -> np.ndarray:
    flat = pixel_array.astype(np.uint8).tobytes()
    encrypted = aes_cbc_encrypt_bytes(flat, key, iv)
    return np.frombuffer(encrypted, dtype=np.uint8)


def aes_cbc_decrypt_array(encrypted_array: np.ndarray, key: bytes, iv: bytes) -> np.ndarray:
    encrypted_bytes = encrypted_array.tobytes()
    decrypted = aes_cbc_decrypt_bytes(encrypted_bytes, key, iv)
    return np.frombuffer(decrypted, dtype=np.uint8)
