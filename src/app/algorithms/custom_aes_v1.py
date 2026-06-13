from typing import List

import numpy as np

from .custom_aes_core import (
    _NUMBA, njit,
    _key_schedule, _aes_encrypt_block, _aes_decrypt_block,
    generate_round_keys, aes_encrypt, aes_decrypt,
)


def _validate_key(key: str) -> None:
    if len(key) != 16:
        raise ValueError(f"Key must be exactly 16 characters, got {len(key)}.")


# ---------------------------------------------------------------------------
# ECB processing loop (Numba JIT)
# ---------------------------------------------------------------------------

@njit
def _process_blocks(data, W, encrypt):
    n = len(data)
    out = np.empty(n, dtype=np.uint8)
    full = (n // 16) * 16
    i = 0
    while i < full:
        blk = data[i:i + 16]
        if encrypt:
            res = _aes_encrypt_block(blk, W)
        else:
            res = _aes_decrypt_block(blk, W)
        out[i:i + 16] = res
        i += 16
    while i < n:
        out[i] = data[i]
        i += 1
    return out


if _NUMBA:
    _dummy_W = _key_schedule(np.zeros(16, dtype=np.uint8))
    _process_blocks(np.zeros(16, dtype=np.uint8), _dummy_W, True)
    _process_blocks(np.zeros(16, dtype=np.uint8), _dummy_W, False)
    del _dummy_W


# ---------------------------------------------------------------------------
# Pure-Python ECB fallback
# ---------------------------------------------------------------------------

def _encrypt_pixels_py(pixels: List[int], round_keys) -> List[int]:
    out = []
    for start in range(0, len(pixels), 16):
        chunk = pixels[start:start + 16]
        if len(chunk) == 16:
            out.extend(aes_encrypt(chunk, round_keys))
        else:
            out.extend(chunk)
    return out


def _decrypt_pixels_py(pixels: List[int], round_keys) -> List[int]:
    out = []
    for start in range(0, len(pixels), 16):
        chunk = pixels[start:start + 16]
        if len(chunk) == 16:
            out.extend(aes_decrypt(chunk, round_keys))
        else:
            out.extend(chunk)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_custom_aes_v1_encrypt(pixels: List[int], key: str) -> List[int]:
    _validate_key(key)
    if _NUMBA:
        key_bytes = np.frombuffer(key.encode("latin-1"), dtype=np.uint8)
        W = _key_schedule(key_bytes)
        arr = np.asarray(pixels, dtype=np.uint8)
        return _process_blocks(arr, W, True).tolist()
    round_keys = generate_round_keys(key)
    return _encrypt_pixels_py(pixels, round_keys)


def apply_custom_aes_v1_decrypt(pixels: List[int], key: str) -> List[int]:
    _validate_key(key)
    if _NUMBA:
        key_bytes = np.frombuffer(key.encode("latin-1"), dtype=np.uint8)
        W = _key_schedule(key_bytes)
        arr = np.asarray(pixels, dtype=np.uint8)
        return _process_blocks(arr, W, False).tolist()
    round_keys = generate_round_keys(key)
    return _decrypt_pixels_py(pixels, round_keys)


