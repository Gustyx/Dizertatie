from typing import List

from .custom_aes_core import generate_round_keys, aes_encrypt, aes_decrypt


def apply_custom_aes_v2_encrypt(pixels: List[int], key: str) -> List[int]:
    round_keys = generate_round_keys(key)
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
            out.extend(chunk[i] ^ prev[i] for i in range(len(chunk)))
    return out


def apply_custom_aes_v2_decrypt(pixels: List[int], key: str, original_length: int) -> List[int]:
    round_keys = generate_round_keys(key)
    out = []
    prev = [0] * 16
    for start in range(0, len(pixels), 16):
        chunk = pixels[start:start + 16]
        if len(chunk) == 16:
            dec = aes_decrypt(chunk, round_keys)
            out.extend(dec[i] ^ prev[i] for i in range(16))
            prev = list(chunk)
        else:
            out.extend(chunk[i] ^ prev[i] for i in range(len(chunk)))
    return out
