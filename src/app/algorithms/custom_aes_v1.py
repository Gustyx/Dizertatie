from typing import List

from .custom_aes_core import generate_round_keys, aes_encrypt, aes_decrypt


def _validate_key(key: str) -> None:
    if len(key) != 16:
        raise ValueError(f"Key must be exactly 16 characters, got {len(key)}.")


def apply_custom_aes_v1_encrypt(pixels: List[int], key: str) -> List[int]:
    _validate_key(key)
    round_keys = generate_round_keys(key)
    out = []
    for start in range(0, len(pixels), 16):
        chunk = pixels[start:start + 16]
        if len(chunk) == 16:
            out.extend(aes_encrypt(chunk, round_keys))
        else:
            out.extend(chunk)
    return out


def apply_custom_aes_v1_decrypt(pixels: List[int], key: str) -> List[int]:
    _validate_key(key)
    round_keys = generate_round_keys(key)
    out = []
    for start in range(0, len(pixels), 16):
        chunk = pixels[start:start + 16]
        if len(chunk) == 16:
            out.extend(aes_decrypt(chunk, round_keys))
        else:
            out.extend(chunk)
    return out
