from __future__ import annotations

"""
Custom AES-128 implementation — corrected & Numba-accelerated.

Fixes over the original:
  1. Key schedule: proper RotWord → SubWord → XOR Rcon → XOR W[i-4] sequence.
  2. Full 10 rounds applied to every 16-byte block (not 1 round cycled).
  3. InvMixColumns applied correctly (key XOR happens outside Galois mult).
  4. Key validation (must be exactly 16 bytes / chars).
  5. Numba JIT on the hot Galois-field arithmetic and round functions.
"""

from datetime import datetime
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Optional Numba
# ---------------------------------------------------------------------------
try:
    from numba import njit as _njit, uint8 as _u8
    _NUMBA = True
except ImportError:
    _NUMBA = False
    def _njit(fn=None, **kw):  # type: ignore
        return (lambda f: f)(fn) if fn else (lambda f: f)

# ---------------------------------------------------------------------------
# AES tables as flat NumPy arrays (Numba works best with 1-D arrays)
# ---------------------------------------------------------------------------

S_BOX = np.array([
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
], dtype=np.uint8)

INV_S_BOX = np.array([
    0x52,0x09,0x6a,0xd5,0x30,0x36,0xa5,0x38,0xbf,0x40,0xa3,0x9e,0x81,0xf3,0xd7,0xfb,
    0x7c,0xe3,0x39,0x82,0x9b,0x2f,0xff,0x87,0x34,0x8e,0x43,0x44,0xc4,0xde,0xe9,0xcb,
    0x54,0x7b,0x94,0x32,0xa6,0xc2,0x23,0x3d,0xee,0x4c,0x95,0x0b,0x42,0xfa,0xc3,0x4e,
    0x08,0x2e,0xa1,0x66,0x28,0xd9,0x24,0xb2,0x76,0x5b,0xa2,0x49,0x6d,0x8b,0xd1,0x25,
    0x72,0xf8,0xf6,0x64,0x86,0x68,0x98,0x16,0xd4,0xa4,0x5c,0xcc,0x5d,0x65,0xb6,0x92,
    0x6c,0x70,0x48,0x50,0xfd,0xed,0xb9,0xda,0x5e,0x15,0x46,0x57,0xa7,0x8d,0x9d,0x84,
    0x90,0xd8,0xab,0x00,0x8c,0xbc,0xd3,0x0a,0xf7,0xe4,0x58,0x05,0xb8,0xb3,0x45,0x06,
    0xd0,0x2c,0x1e,0x8f,0xca,0x3f,0x0f,0x02,0xc1,0xaf,0xbd,0x03,0x01,0x13,0x8a,0x6b,
    0x3a,0x91,0x11,0x41,0x4f,0x67,0xdc,0xea,0x97,0xf2,0xcf,0xce,0xf0,0xb4,0xe6,0x73,
    0x96,0xac,0x74,0x22,0xe7,0xad,0x35,0x85,0xe2,0xf9,0x37,0xe8,0x1c,0x75,0xdf,0x6e,
    0x47,0xf1,0x1a,0x71,0x1d,0x29,0xc5,0x89,0x6f,0xb7,0x62,0x0e,0xaa,0x18,0xbe,0x1b,
    0xfc,0x56,0x3e,0x4b,0xc6,0xd2,0x79,0x20,0x9a,0xdb,0xc0,0xfe,0x78,0xcd,0x5a,0xf4,
    0x1f,0xdd,0xa8,0x33,0x88,0x07,0xc7,0x31,0xb1,0x12,0x10,0x59,0x27,0x80,0xec,0x5f,
    0x60,0x51,0x7f,0xa9,0x19,0xb5,0x4a,0x0d,0x2d,0xe5,0x7a,0x9f,0x93,0xc9,0x9c,0xef,
    0xa0,0xe0,0x3b,0x4d,0xae,0x2a,0xf5,0xb0,0xc8,0xeb,0xbb,0x3c,0x83,0x53,0x99,0x61,
    0x17,0x2b,0x04,0x7e,0xba,0x77,0xd6,0x26,0xe1,0x69,0x14,0x63,0x55,0x21,0x0c,0x7d,
], dtype=np.uint8)

RCON = np.array([
    0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1B,0x36,
], dtype=np.uint8)

# ---------------------------------------------------------------------------
# Galois-field helpers  (Numba JIT)
# ---------------------------------------------------------------------------

@_njit
def _gmul(a: int, b: int) -> int:
    """Multiply two bytes in GF(2^8) mod 0x11B."""
    p = np.uint8(0)
    a = np.uint8(a)
    b = np.uint8(b)
    for _ in range(8):
        if b & np.uint8(1):
            p ^= a
        hi = a & np.uint8(0x80)
        a = np.uint8((a << np.uint8(1)) & np.uint8(0xFF))
        if hi:
            a ^= np.uint8(0x1B)
        b >>= np.uint8(1)
    return p


# ---------------------------------------------------------------------------
# Key schedule  (corrected)
# ---------------------------------------------------------------------------

@_njit
def _key_schedule(key: np.ndarray) -> np.ndarray:
    """
    Standard AES-128 key expansion.
    Input : 16-byte key array.
    Output: 44 words × 4 bytes = (44, 4) uint8 array.

    Correct sequence per FIPS 197 §5.2:
      W[i] = W[i-4] XOR SubWord(RotWord(W[i-1])) XOR Rcon[i/4 - 1]  (i % 4 == 0)
      W[i] = W[i-4] XOR W[i-1]                                         (otherwise)
    """
    W = np.zeros((44, 4), dtype=np.uint8)

    # First 4 words directly from the key
    for i in range(4):
        for b in range(4):
            W[i, b] = key[i * 4 + b]

    for i in range(4, 44):
        temp = W[i - 1].copy()

        if i % 4 == 0:
            # RotWord: [a0,a1,a2,a3] -> [a1,a2,a3,a0]
            t = temp[0]
            temp[0] = temp[1]
            temp[1] = temp[2]
            temp[2] = temp[3]
            temp[3] = t
            # SubWord
            for b in range(4):
                temp[b] = S_BOX[temp[b]]
            # XOR Rcon
            temp[0] ^= RCON[i // 4 - 1]

        for b in range(4):
            W[i, b] = W[i - 4, b] ^ temp[b]

    return W


# ---------------------------------------------------------------------------
# AES round operations  (Numba JIT)
# ---------------------------------------------------------------------------

@_njit
def _add_round_key(state: np.ndarray, W: np.ndarray, round_idx: int) -> np.ndarray:
    """XOR state (4×4) with 4 words starting at W[round_idx*4]."""
    out = state.copy()
    base = round_idx * 4
    for col in range(4):
        for row in range(4):
            out[row, col] ^= W[base + col, row]
    return out


@_njit
def _sub_bytes(state: np.ndarray) -> np.ndarray:
    out = np.empty((4, 4), dtype=np.uint8)
    for i in range(4):
        for j in range(4):
            out[i, j] = S_BOX[state[i, j]]
    return out


@_njit
def _inv_sub_bytes(state: np.ndarray) -> np.ndarray:
    out = np.empty((4, 4), dtype=np.uint8)
    for i in range(4):
        for j in range(4):
            out[i, j] = INV_S_BOX[state[i, j]]
    return out


@_njit
def _shift_rows(state: np.ndarray) -> np.ndarray:
    out = np.empty((4, 4), dtype=np.uint8)
    for i in range(4):
        for j in range(4):
            out[i, j] = state[i, (j + i) % 4]
    return out


@_njit
def _inv_shift_rows(state: np.ndarray) -> np.ndarray:
    out = np.empty((4, 4), dtype=np.uint8)
    for i in range(4):
        for j in range(4):
            out[i, j] = state[i, (j - i) % 4]
    return out


@_njit
def _mix_columns(state: np.ndarray) -> np.ndarray:
    out = np.zeros((4, 4), dtype=np.uint8)
    for col in range(4):
        s0 = state[0, col]
        s1 = state[1, col]
        s2 = state[2, col]
        s3 = state[3, col]
        out[0, col] = _gmul(0x02, s0) ^ _gmul(0x03, s1) ^ s2 ^ s3
        out[1, col] = s0 ^ _gmul(0x02, s1) ^ _gmul(0x03, s2) ^ s3
        out[2, col] = s0 ^ s1 ^ _gmul(0x02, s2) ^ _gmul(0x03, s3)
        out[3, col] = _gmul(0x03, s0) ^ s1 ^ s2 ^ _gmul(0x02, s3)
    return out


@_njit
def _inv_mix_columns(state: np.ndarray) -> np.ndarray:
    out = np.zeros((4, 4), dtype=np.uint8)
    for col in range(4):
        s0 = state[0, col]
        s1 = state[1, col]
        s2 = state[2, col]
        s3 = state[3, col]
        out[0, col] = _gmul(0x0e,s0)^_gmul(0x0b,s1)^_gmul(0x0d,s2)^_gmul(0x09,s3)
        out[1, col] = _gmul(0x09,s0)^_gmul(0x0e,s1)^_gmul(0x0b,s2)^_gmul(0x0d,s3)
        out[2, col] = _gmul(0x0d,s0)^_gmul(0x09,s1)^_gmul(0x0e,s2)^_gmul(0x0b,s3)
        out[3, col] = _gmul(0x0b,s0)^_gmul(0x0d,s1)^_gmul(0x09,s2)^_gmul(0x0e,s3)
    return out


# ---------------------------------------------------------------------------
# Full block encrypt / decrypt  (all 10 rounds, corrected)
# ---------------------------------------------------------------------------

@_njit
def _aes_encrypt_block(block: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Encrypt a single 16-byte block using the full AES-128 pipeline.

    Standard FIPS 197 structure:
      AddRoundKey(0)
      for round 1..9:  SubBytes → ShiftRows → MixColumns → AddRoundKey
      round 10:        SubBytes → ShiftRows → AddRoundKey  (no MixColumns)
    """
    # Load block into column-major state
    state = np.zeros((4, 4), dtype=np.uint8)
    for col in range(4):
        for row in range(4):
            state[row, col] = block[col * 4 + row]

    state = _add_round_key(state, W, 0)

    for rnd in range(1, 10):
        state = _sub_bytes(state)
        state = _shift_rows(state)
        state = _mix_columns(state)
        state = _add_round_key(state, W, rnd)

    # Final round — no MixColumns
    state = _sub_bytes(state)
    state = _shift_rows(state)
    state = _add_round_key(state, W, 10)

    out = np.empty(16, dtype=np.uint8)
    for col in range(4):
        for row in range(4):
            out[col * 4 + row] = state[row, col]
    return out


@_njit
def _aes_decrypt_block(block: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Decrypt a single 16-byte block (equivalent inverse cipher).

    Standard FIPS 197 §5.3:
      AddRoundKey(10)
      for round 9..1:  InvShiftRows → InvSubBytes → AddRoundKey → InvMixColumns
      round 0:         InvShiftRows → InvSubBytes → AddRoundKey
    """
    state = np.zeros((4, 4), dtype=np.uint8)
    for col in range(4):
        for row in range(4):
            state[row, col] = block[col * 4 + row]

    state = _add_round_key(state, W, 10)

    for rnd in range(9, 0, -1):
        state = _inv_shift_rows(state)
        state = _inv_sub_bytes(state)
        state = _add_round_key(state, W, rnd)
        state = _inv_mix_columns(state)

    # Final round
    state = _inv_shift_rows(state)
    state = _inv_sub_bytes(state)
    state = _add_round_key(state, W, 0)

    out = np.empty(16, dtype=np.uint8)
    for col in range(4):
        for row in range(4):
            out[col * 4 + row] = state[row, col]
    return out


# ---------------------------------------------------------------------------
# Batch encrypt / decrypt over pixel arrays
# ---------------------------------------------------------------------------

@_njit
def _process_blocks(data: np.ndarray, W: np.ndarray, encrypt: bool) -> np.ndarray:
    """Process all 16-byte blocks in *data* (padded to multiple of 16)."""
    n = len(data)
    # Round up to next multiple of 16
    padded_len = n + (16 - n % 16) % 16
    padded = np.zeros(padded_len, dtype=np.uint8)
    for i in range(n):
        padded[i] = data[i]

    out = np.empty(padded_len, dtype=np.uint8)
    for start in range(0, padded_len, 16):
        block = padded[start:start + 16]
        if encrypt:
            result = _aes_encrypt_block(block, W)
        else:
            result = _aes_decrypt_block(block, W)
        for k in range(16):
            out[start + k] = result[k]

    return out[:n]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _validate_key(key: str) -> np.ndarray:
    if len(key) != 16:
        raise ValueError(f"AES-128 key must be exactly 16 characters, got {len(key)}")
    return np.frombuffer(key.encode("latin-1"), dtype=np.uint8)


def apply_custom_aes_v2_encrypt(pixels: List[int], key: str) -> List[int]:
    """Encrypt a flat pixel list with AES-128 (ECB, PKCS-style zero-pad)."""
    t = datetime.now()
    print(t, "Starting encryption")
    key_bytes = _validate_key(key)
    W = _key_schedule(key_bytes)
    arr = np.asarray(pixels, dtype=np.uint8)
    enc = _process_blocks(arr, W, encrypt=True)
    print(datetime.now(), "Finished encryption")
    print(f"Encryption took {(datetime.now() - t).total_seconds()} seconds")
    return enc.tolist()


def apply_custom_aes_v2_decrypt(pixels: List[int], key: str) -> List[int]:
    """Decrypt a flat pixel list produced by apply_custom_aes_encrypt."""
    key_bytes = _validate_key(key)
    W = _key_schedule(key_bytes)
    arr = np.asarray(pixels, dtype=np.uint8)
    dec = _process_blocks(arr, W, encrypt=False)
    return dec.tolist()


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    enc = apply_custom_aes_v2_encrypt([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                       33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                       49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64], "encryptionkey123")
    print("Encrypted:", len(enc), enc)
    dec = apply_custom_aes_v2_decrypt(enc, "encryptionkey123")
    print("Decrypted:", len(dec), dec)