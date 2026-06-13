"""
Custom AES v3 — CBC mode on top of the shared FIPS 197 AES-128 block cipher.
Full 16-byte blocks: XOR with previous ciphertext block, then AES encrypt.
Partial last block: XOR-only (no AES), so output length == input length.
"""
from typing import List

import numpy as np

from .custom_aes_core import (
    _NUMBA, njit,
    _key_schedule, _aes_encrypt_block, _aes_decrypt_block,
    generate_round_keys, aes_encrypt, aes_decrypt,
)


# ---------------------------------------------------------------------------
# CBC processing loop (Numba JIT)
# ---------------------------------------------------------------------------

@njit
def _process_blocks_cbc(data, W, encrypt):
    n = len(data)
    out = np.empty(n, dtype=np.uint8)
    prev = np.zeros(16, dtype=np.uint8)  # IV = 0…0
    full = (n // 16) * 16
    i = 0
    while i < full:
        blk = data[i:i + 16].copy()
        if encrypt:
            for k in range(16):
                blk[k] ^= prev[k]
            res = _aes_encrypt_block(blk, W)
            out[i:i + 16] = res
            prev = res
        else:
            res = _aes_decrypt_block(blk, W)
            for k in range(16):
                out[i + k] = res[k] ^ prev[k]
            prev = blk
        i += 16
    # Partial last block: XOR-only (preserves output length == input length)
    r = n - full
    for k in range(r):
        out[full + k] = data[full + k] ^ prev[k]
    return out


if _NUMBA:
    _dummy_W = _key_schedule(np.zeros(16, dtype=np.uint8))
    _process_blocks_cbc(np.zeros(16, dtype=np.uint8), _dummy_W, True)
    _process_blocks_cbc(np.zeros(16, dtype=np.uint8), _dummy_W, False)
    del _dummy_W


# ---------------------------------------------------------------------------
# Pure-Python CBC fallback
# ---------------------------------------------------------------------------

def _encrypt_pixels_py(pixels: List[int], round_keys) -> List[int]:
    out = []
    prev = [0] * 16
    for start in range(0, len(pixels), 16):
        chunk = pixels[start:start + 16]
        if len(chunk) == 16:
            xored = [chunk[i] ^ prev[i] for i in range(16)]
            enc = aes_encrypt(xored, round_keys)
            out.extend(enc)
            prev = enc
        else:
            r = len(chunk)
            out.extend(chunk[i] ^ prev[i] for i in range(r))
    return out


def _decrypt_pixels_py(pixels: List[int], round_keys) -> List[int]:
    out = []
    prev = [0] * 16
    for start in range(0, len(pixels), 16):
        chunk = pixels[start:start + 16]
        if len(chunk) == 16:
            dec = aes_decrypt(chunk, round_keys)
            out.extend(dec[i] ^ prev[i] for i in range(16))
            prev = list(chunk)
        else:
            r = len(chunk)
            out.extend(chunk[i] ^ prev[i] for i in range(r))
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_custom_aes_v2_encrypt(pixels: List[int], key: str) -> List[int]:
    if _NUMBA:
        key_bytes = np.frombuffer(key.encode("latin-1"), dtype=np.uint8)
        W = _key_schedule(key_bytes)
        arr = np.asarray(pixels, dtype=np.uint8)
        return _process_blocks_cbc(arr, W, True).tolist()
    round_keys = generate_round_keys(key)
    return _encrypt_pixels_py(pixels, round_keys)


def apply_custom_aes_v2_decrypt(pixels: List[int], key: str, original_length: int) -> List[int]:
    if _NUMBA:
        key_bytes = np.frombuffer(key.encode("latin-1"), dtype=np.uint8)
        W = _key_schedule(key_bytes)
        arr = np.asarray(pixels, dtype=np.uint8)
        return _process_blocks_cbc(arr, W, False).tolist()
    round_keys = generate_round_keys(key)
    return _decrypt_pixels_py(pixels, round_keys)
