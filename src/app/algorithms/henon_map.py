from __future__ import annotations

import hashlib
import struct
from typing import Sequence

import numpy as np

try:
    from numba import njit as _njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

    def _njit(fn):  # type: ignore[misc]
        return fn


_BURN_IN_STEPS: int = 256
_HASH_CHUNK: int = 64   # Henon floats hashed per SHA-256 call -> 32 key bytes


def _seed_from_key(key_phrase: str, label: str) -> int:
    digest = hashlib.sha256(f"{label}:{key_phrase}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _initial_state(key_phrase: str, label: str) -> tuple[float, float, float, float]:
    seed = _seed_from_key(key_phrase, label)
    x0 = ((seed & ((1 << 53) - 1)) + 1) / float(1 << 53)
    y0 = (((seed >> 8) & ((1 << 53) - 1)) + 1) / float(1 << 53)
    a_seed = _seed_from_key(key_phrase, f"{label}:a")
    b_seed = _seed_from_key(key_phrase, f"{label}:b")
    a = 1.2 + (a_seed / float(1 << 64)) * 0.6   # in [1.2, 1.8]
    b = 0.1 + (b_seed / float(1 << 64)) * 0.4   # in [0.1, 0.5]
    return x0, y0, a, b


@_njit
def _henon_loop(
    length: int,
    x: float,
    y: float,
    a: float,
    b: float,
    burn_in: int,
) -> np.ndarray:
    """
    Burn-in + generate *length* values, interleaving x and y.
    Fractional-part wrapping prevents divergence without constant fallback.
    """
    for _ in range(burn_in):
        xn = 1.0 - a * x * x + y
        yn = b * x
        x = abs(xn) - np.floor(abs(xn))
        y = abs(yn) - np.floor(abs(yn))

    values = np.empty(length, dtype=np.float64)
    for i in range(length):
        xn = 1.0 - a * x * x + y
        yn = b * x
        x = abs(xn) - np.floor(abs(xn))
        y = abs(yn) - np.floor(abs(yn))
        v = x if (i & 1) == 0 else y
        if v == 0.0:
            v = 1.1102230246251565e-16
        values[i] = v

    return values


def _henon_sequence(length: int, key_phrase: str, label: str) -> np.ndarray:
    if length <= 0:
        return np.empty(0, dtype=np.float64)
    x0, y0, a, b = _initial_state(key_phrase, label)
    return _henon_loop(length, x0, y0, a, b, _BURN_IN_STEPS)


def _permutation(length: int, key_phrase: str, label: str = "perm") -> np.ndarray:
    seq = _henon_sequence(length, key_phrase, label)
    idx = np.arange(length, dtype=np.int64)
    return np.lexsort((idx, seq))


def _keystream(length: int, key_phrase: str, label: str = "stream") -> np.ndarray:
    """
    FIX: Direct floor(v*256) of Henon values clusters around attractor
    bounds, producing color bias.  Instead, pack chunks of raw float64
    values into SHA-256, yielding cryptographically uniform bytes.
    """
    if length <= 0:
        return np.empty(0, dtype=np.uint8)

    seq = _henon_sequence(max(length, _HASH_CHUNK), key_phrase, label)

    out = np.empty(length, dtype=np.uint8)
    filled = 0
    chunk_idx = 0

    while filled < length:
        chunk_end = min(chunk_idx + _HASH_CHUNK, len(seq))
        chunk = seq[chunk_idx:chunk_end]
        raw = struct.pack(f"{len(chunk)}d", *chunk.tolist())
        hashed = hashlib.sha256(raw).digest()   # 32 uniform bytes
        take = min(len(hashed), length - filled)
        out[filled:filled + take] = np.frombuffer(hashed[:take], dtype=np.uint8)
        filled += take
        chunk_idx = chunk_end

        if chunk_idx >= len(seq) and filled < length:
            seq = _henon_sequence(
                length - filled + _HASH_CHUNK,
                key_phrase,
                f"{label}:{chunk_idx}",
            )
            chunk_idx = 0

    return out


def henon_map_encrypt_array(pixel_array: np.ndarray, key_phrase: str) -> np.ndarray:
    """
    Encrypt a H x W x C (or H x W grayscale) image array.

    Pipeline:
      1. Row permutation    - shuffle rows
      2. Column permutation - shuffle columns
      3. Global pixel permutation - flatten and shuffle all pixels
      4. XOR diffusion      - XOR each channel with hashed keystream
    """
    img = pixel_array.astype(np.uint8, copy=True)
    shape = img.shape

    if img.ndim == 2:
        img = img[:, :, np.newaxis]

    H, W, C = img.shape

    # Stage 1: row permutation
    row_perm = _permutation(H, key_phrase, "row_perm")
    img = img[row_perm, :, :]

    # Stage 2: column permutation
    col_perm = _permutation(W, key_phrase, "col_perm")
    img = img[:, col_perm, :]

    # Stage 3: global pixel permutation
    flat_pixels = img.reshape(-1, C)
    px_perm = _permutation(H * W, key_phrase, "px_perm")
    flat_pixels = flat_pixels[px_perm, :]

    # Stage 4: XOR diffusion per channel
    for c in range(C):
        ks = _keystream(H * W, key_phrase, f"ks_ch{c}")
        flat_pixels[:, c] = np.bitwise_xor(flat_pixels[:, c], ks)

    result = flat_pixels.reshape(H, W, C)

    if len(shape) == 2:
        result = result[:, :, 0]

    return result


def henon_map_decrypt_array(encrypted_array: np.ndarray, key_phrase: str) -> np.ndarray:
    """
    Exact inverse of henon_map_encrypt_array.

    Pipeline (reversed):
      4. Un-XOR diffusion
      3. Inverse global pixel permutation
      2. Inverse column permutation
      1. Inverse row permutation
    """
    img = encrypted_array.astype(np.uint8, copy=True)
    shape = img.shape

    if img.ndim == 2:
        img = img[:, :, np.newaxis]

    H, W, C = img.shape
    flat_pixels = img.reshape(-1, C)

    # Stage 4 inverse: un-XOR
    for c in range(C):
        ks = _keystream(H * W, key_phrase, f"ks_ch{c}")
        flat_pixels[:, c] = np.bitwise_xor(flat_pixels[:, c], ks)

    # Stage 3 inverse: inverse pixel permutation
    px_perm = _permutation(H * W, key_phrase, "px_perm")
    inv_px = np.empty(H * W, dtype=np.int64)
    inv_px[px_perm] = np.arange(H * W, dtype=np.int64)
    flat_pixels = flat_pixels[inv_px, :]

    img = flat_pixels.reshape(H, W, C)

    # Stage 2 inverse: inverse column permutation
    col_perm = _permutation(W, key_phrase, "col_perm")
    inv_col = np.empty(W, dtype=np.int64)
    inv_col[col_perm] = np.arange(W, dtype=np.int64)
    img = img[:, inv_col, :]

    # Stage 1 inverse: inverse row permutation
    row_perm = _permutation(H, key_phrase, "row_perm")
    inv_row = np.empty(H, dtype=np.int64)
    inv_row[row_perm] = np.arange(H, dtype=np.int64)
    img = img[inv_row, :, :]

    if len(shape) == 2:
        img = img[:, :, 0]

    return img


def henon_map_encrypt_bytes(data: bytes, key_phrase: str) -> bytes:
    """Encrypt raw bytes (no pixel-awareness)."""
    if not data:
        return b""
    arr = np.frombuffer(data, dtype=np.uint8)
    perm = _permutation(len(arr), key_phrase, "byte_perm")
    ks = _keystream(len(arr), key_phrase, "byte_stream")
    return np.bitwise_xor(arr[perm], ks).tobytes()


def henon_map_decrypt_bytes(data: bytes, key_phrase: str) -> bytes:
    if not data:
        return b""
    arr = np.frombuffer(data, dtype=np.uint8)
    perm = _permutation(len(arr), key_phrase, "byte_perm")
    ks = _keystream(len(arr), key_phrase, "byte_stream")
    inv = np.empty(len(arr), dtype=np.int64)
    inv[perm] = np.arange(len(arr), dtype=np.int64)
    return np.bitwise_xor(arr, ks)[inv].tobytes()


def apply_henon_map_encrypt(pixels: Sequence[int], key_phrase: str) -> list[int]:
    flat = np.asarray(pixels, dtype=np.uint8)
    enc = henon_map_encrypt_bytes(flat.tobytes(), key_phrase)
    return np.frombuffer(enc, dtype=np.uint8).tolist()


def apply_henon_map_decrypt(pixels: Sequence[int], key_phrase: str) -> list[int]:
    flat = np.asarray(pixels, dtype=np.uint8)
    dec = henon_map_decrypt_bytes(flat.tobytes(), key_phrase)
    return np.frombuffer(dec, dtype=np.uint8).tolist()