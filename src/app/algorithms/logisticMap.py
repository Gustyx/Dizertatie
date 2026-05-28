from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

_BURN_IN_STEPS = 256


def _seed_from_key(key_phrase: str, label: str) -> int:
    digest = hashlib.sha256(f"{label}:{key_phrase}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _initial_state(key_phrase: str, label: str) -> tuple[float, float]:
    seed = _seed_from_key(key_phrase, label)
    x0 = ((seed & ((1 << 53) - 1)) + 1) / float(1 << 53)

    r_seed = _seed_from_key(key_phrase, f"{label}:r")
    r = 3.9 + (r_seed / float(1 << 64)) * 0.09999999999999964
    return x0, r


def _logistic_sequence(length: int, key_phrase: str, label: str) -> np.ndarray:
    if length <= 0:
        return np.empty(0, dtype=np.float64)

    x, r = _initial_state(key_phrase, label)

    for _ in range(_BURN_IN_STEPS):
        x = r * x * (1.0 - x)

    values = np.empty(length, dtype=np.float64)
    for index in range(length):
        x = r * x * (1.0 - x)
        if x <= 0.0 or x >= 1.0:
            x = np.nextafter(0.5, 1.0)
        values[index] = x

    return values


def _permutation(length: int, key_phrase: str) -> np.ndarray:
    sequence = _logistic_sequence(length, key_phrase, "perm")
    indices = np.arange(length, dtype=np.int64)
    return np.lexsort((indices, sequence))


def _keystream(length: int, key_phrase: str) -> np.ndarray:
    sequence = _logistic_sequence(length, key_phrase, "stream")
    return np.floor(sequence * 256.0).astype(np.uint8)


def _transform_bytes(data: bytes, key_phrase: str, *, decrypt: bool) -> bytes:
    if not data:
        return b""

    pixels = np.frombuffer(data, dtype=np.uint8)
    permutation = _permutation(len(pixels), key_phrase)
    keystream = _keystream(len(pixels), key_phrase)

    if decrypt:
        inverse = np.empty_like(permutation)
        inverse[permutation] = np.arange(len(permutation), dtype=np.int64)
        unxored = np.bitwise_xor(pixels, keystream)
        restored = unxored[inverse]
        return restored.tobytes()

    permuted = pixels[permutation]
    encrypted = np.bitwise_xor(permuted, keystream)
    return encrypted.tobytes()


def logistic_map_encrypt_bytes(data: bytes, key_phrase: str) -> bytes:
    """Encrypt bytes using a reversible logistic-map permutation and keystream."""

    return _transform_bytes(data, key_phrase, decrypt=False)


def logistic_map_decrypt_bytes(data: bytes, key_phrase: str) -> bytes:
    """Decrypt bytes produced by `logistic_map_encrypt_bytes`."""

    return _transform_bytes(data, key_phrase, decrypt=True)


def logistic_map_encrypt_array(pixel_array: np.ndarray, key_phrase: str) -> np.ndarray:
    """Encrypt a numpy pixel array and return an array with the same shape."""

    flat = pixel_array.astype(np.uint8, copy=False).tobytes()
    encrypted = logistic_map_encrypt_bytes(flat, key_phrase)
    return np.frombuffer(encrypted, dtype=np.uint8).reshape(pixel_array.shape)


def logistic_map_decrypt_array(
    encrypted_array: np.ndarray, key_phrase: str
) -> np.ndarray:
    """Decrypt a numpy pixel array produced by `logistic_map_encrypt_array`."""

    encrypted = encrypted_array.astype(np.uint8, copy=False).tobytes()
    decrypted = logistic_map_decrypt_bytes(encrypted, key_phrase)
    return np.frombuffer(decrypted, dtype=np.uint8).reshape(encrypted_array.shape)


def apply_logistic_map_encrypt(pixels: Sequence[int], key_phrase: str) -> list[int]:
    """Encrypt a flat pixel sequence and return a list of uint8 values."""

    flat = np.asarray(list(pixels), dtype=np.uint8)
    encrypted = logistic_map_encrypt_bytes(flat.tobytes(), key_phrase)
    return np.frombuffer(encrypted, dtype=np.uint8).tolist()


def apply_logistic_map_decrypt(pixels: Sequence[int], key_phrase: str) -> list[int]:
    """Decrypt a flat pixel sequence and return a list of uint8 values."""

    flat = np.asarray(list(pixels), dtype=np.uint8)
    decrypted = logistic_map_decrypt_bytes(flat.tobytes(), key_phrase)
    return np.frombuffer(decrypted, dtype=np.uint8).tolist()
