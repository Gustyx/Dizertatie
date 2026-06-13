from typing import List

import numpy as np

try:
    from numba import njit
    _NUMBA = True
except ImportError:
    _NUMBA = False
    def njit(fn=None, **kw):
        return (lambda f: f)(fn) if fn else (lambda f: f)


s_box = [
    [0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76],
    [0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0],
    [0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15],
    [0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75],
    [0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84],
    [0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf],
    [0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8],
    [0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2],
    [0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73],
    [0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb],
    [0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79],
    [0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08],
    [0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a],
    [0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e],
    [0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf],
    [0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16],
]

inv_s_box = [
    [0x52,0x09,0x6a,0xd5,0x30,0x36,0xa5,0x38,0xbf,0x40,0xa3,0x9e,0x81,0xf3,0xd7,0xfb],
    [0x7c,0xe3,0x39,0x82,0x9b,0x2f,0xff,0x87,0x34,0x8e,0x43,0x44,0xc4,0xde,0xe9,0xcb],
    [0x54,0x7b,0x94,0x32,0xa6,0xc2,0x23,0x3d,0xee,0x4c,0x95,0x0b,0x42,0xfa,0xc3,0x4e],
    [0x08,0x2e,0xa1,0x66,0x28,0xd9,0x24,0xb2,0x76,0x5b,0xa2,0x49,0x6d,0x8b,0xd1,0x25],
    [0x72,0xf8,0xf6,0x64,0x86,0x68,0x98,0x16,0xd4,0xa4,0x5c,0xcc,0x5d,0x65,0xb6,0x92],
    [0x6c,0x70,0x48,0x50,0xfd,0xed,0xb9,0xda,0x5e,0x15,0x46,0x57,0xa7,0x8d,0x9d,0x84],
    [0x90,0xd8,0xab,0x00,0x8c,0xbc,0xd3,0x0a,0xf7,0xe4,0x58,0x05,0xb8,0xb3,0x45,0x06],
    [0xd0,0x2c,0x1e,0x8f,0xca,0x3f,0x0f,0x02,0xc1,0xaf,0xbd,0x03,0x01,0x13,0x8a,0x6b],
    [0x3a,0x91,0x11,0x41,0x4f,0x67,0xdc,0xea,0x97,0xf2,0xcf,0xce,0xf0,0xb4,0xe6,0x73],
    [0x96,0xac,0x74,0x22,0xe7,0xad,0x35,0x85,0xe2,0xf9,0x37,0xe8,0x1c,0x75,0xdf,0x6e],
    [0x47,0xf1,0x1a,0x71,0x1d,0x29,0xc5,0x89,0x6f,0xb7,0x62,0x0e,0xaa,0x18,0xbe,0x1b],
    [0xfc,0x56,0x3e,0x4b,0xc6,0xd2,0x79,0x20,0x9a,0xdb,0xc0,0xfe,0x78,0xcd,0x5a,0xf4],
    [0x1f,0xdd,0xa8,0x33,0x88,0x07,0xc7,0x31,0xb1,0x12,0x10,0x59,0x27,0x80,0xec,0x5f],
    [0x60,0x51,0x7f,0xa9,0x19,0xb5,0x4a,0x0d,0x2d,0xe5,0x7a,0x9f,0x93,0xc9,0x9c,0xef],
    [0xa0,0xe0,0x3b,0x4d,0xae,0x2a,0xf5,0xb0,0xc8,0xeb,0xbb,0x3c,0x83,0x53,0x99,0x61],
    [0x17,0x2b,0x04,0x7e,0xba,0x77,0xd6,0x26,0xe1,0x69,0x14,0x63,0x55,0x21,0x0c,0x7d],
]

fixed_matrix = [
    [0x02, 0x03, 0x01, 0x01],
    [0x01, 0x02, 0x03, 0x01],
    [0x01, 0x01, 0x02, 0x03],
    [0x03, 0x01, 0x01, 0x02],
]

inv_fixed_matrix = [
  [0x0e, 0x0b, 0x0d, 0x09],
  [0x09, 0x0e, 0x0b, 0x0d],
  [0x0d, 0x09, 0x0e, 0x0b],
  [0x0b, 0x0d, 0x09, 0x0e],
];

round_constants = [
    [0x01,0x00,0x00,0x00],
    [0x02,0x00,0x00,0x00],
    [0x04,0x00,0x00,0x00],
    [0x08,0x00,0x00,0x00],
    [0x10,0x00,0x00,0x00],
    [0x20,0x00,0x00,0x00],
    [0x40,0x00,0x00,0x00],
    [0x80,0x00,0x00,0x00],
    [0x1B,0x00,0x00,0x00],
    [0x36,0x00,0x00,0x00],
]


# ---------------------------------------------------------------------------
# Flat NumPy arrays for Numba (1-D uint8, no Python lists)
# ---------------------------------------------------------------------------

_S_BOX_FLAT     = np.array([v for row in s_box        for v in row], dtype=np.uint8)
_INV_S_BOX_FLAT = np.array([v for row in inv_s_box    for v in row], dtype=np.uint8)
_FIXED_FLAT     = np.array([v for row in fixed_matrix     for v in row], dtype=np.uint8)
_INV_FIXED_FLAT = np.array([v for row in inv_fixed_matrix for v in row], dtype=np.uint8)
_RCON_FLAT      = np.array([row[0] for row in round_constants], dtype=np.uint8)

# ---------------------------------------------------------------------------
# Numba JIT core
# ---------------------------------------------------------------------------

@njit
def _gmul(a, b):
    p = np.uint8(0)
    a = np.uint8(a); b = np.uint8(b)
    for _ in range(8):
        if b & np.uint8(1):
            p ^= a
        hi = a & np.uint8(0x80)
        a = np.uint8((a << np.uint8(1)) & np.uint8(0xFF))
        if hi:
            a ^= np.uint8(0x1B)
        b >>= np.uint8(1)
    return p


@njit
def _key_schedule(key):
    W = np.zeros((44, 4), dtype=np.uint8)
    for i in range(4):
        for b in range(4):
            W[i, b] = key[i * 4 + b]
    for i in range(4, 44):
        temp = W[i - 1].copy()
        if i % 4 == 0:
            t = temp[0]; temp[0] = temp[1]; temp[1] = temp[2]; temp[2] = temp[3]; temp[3] = t
            for b in range(4):
                temp[b] = _S_BOX_FLAT[temp[b]]
            temp[0] ^= _RCON_FLAT[i // 4 - 1]
        for b in range(4):
            W[i, b] = W[i - 4, b] ^ temp[b]
    return W


@njit
def _add_round_key(state, W, rnd):
    out = state.copy()
    base = rnd * 4
    for col in range(4):
        for row in range(4):
            out[row, col] ^= W[base + col, row]
    return out


@njit
def _aes_encrypt_block(block, W):
    state = np.zeros((4, 4), dtype=np.uint8)
    for col in range(4):
        for row in range(4):
            state[row, col] = block[col * 4 + row]
    state = _add_round_key(state, W, 0)
    for rnd in range(1, 10):
        new_state = np.zeros((4, 4), dtype=np.uint8)
        for i in range(4):
            for j in range(4):
                new_state[i, (j - i) % 4] = _S_BOX_FLAT[state[i, j]]
        state = new_state
        mixed = np.zeros((4, 4), dtype=np.uint8)
        for i in range(4):
            for j in range(4):
                mixed[i, j] = (
                    _gmul(_FIXED_FLAT[i * 4 + 0], state[0, j]) ^
                    _gmul(_FIXED_FLAT[i * 4 + 1], state[1, j]) ^
                    _gmul(_FIXED_FLAT[i * 4 + 2], state[2, j]) ^
                    _gmul(_FIXED_FLAT[i * 4 + 3], state[3, j]) ^
                    W[rnd * 4 + j, i]
                )
        state = mixed
    new_state = np.zeros((4, 4), dtype=np.uint8)
    for i in range(4):
        for j in range(4):
            new_state[i, (j - i) % 4] = _S_BOX_FLAT[state[i, j]]
    state = _add_round_key(new_state, W, 10)
    out = np.empty(16, dtype=np.uint8)
    for col in range(4):
        for row in range(4):
            out[col * 4 + row] = state[row, col]
    return out


@njit
def _aes_decrypt_block(block, W):
    state = np.zeros((4, 4), dtype=np.uint8)
    for col in range(4):
        for row in range(4):
            state[row, col] = block[col * 4 + row]
    state = _add_round_key(state, W, 10)
    for rnd in range(9, 0, -1):
        new_state = np.zeros((4, 4), dtype=np.uint8)
        for i in range(4):
            for j in range(4):
                new_state[i, (j + i) % 4] = _INV_S_BOX_FLAT[state[i, j]]
        state = new_state
        mixed = np.zeros((4, 4), dtype=np.uint8)
        for i in range(4):
            for j in range(4):
                mixed[i, j] = (
                    _gmul(_INV_FIXED_FLAT[i * 4 + 0], state[0, j] ^ W[rnd * 4 + j, 0]) ^
                    _gmul(_INV_FIXED_FLAT[i * 4 + 1], state[1, j] ^ W[rnd * 4 + j, 1]) ^
                    _gmul(_INV_FIXED_FLAT[i * 4 + 2], state[2, j] ^ W[rnd * 4 + j, 2]) ^
                    _gmul(_INV_FIXED_FLAT[i * 4 + 3], state[3, j] ^ W[rnd * 4 + j, 3])
                )
        state = mixed
    new_state = np.zeros((4, 4), dtype=np.uint8)
    for i in range(4):
        for j in range(4):
            new_state[i, (j + i) % 4] = _INV_S_BOX_FLAT[state[i, j]]
    state = _add_round_key(new_state, W, 0)
    out = np.empty(16, dtype=np.uint8)
    for col in range(4):
        for row in range(4):
            out[col * 4 + row] = state[row, col]
    return out


@njit
def _process_blocks_cbc(data, W, encrypt):
    n = len(data)
    out = np.zeros(n, dtype=np.uint8)
    prev = np.zeros(16, dtype=np.uint8)
    start = 0
    while start < n:
        remaining = n - start
        if remaining >= 16:
            if encrypt:
                block = np.zeros(16, dtype=np.uint8)
                for k in range(16):
                    block[k] = data[start + k] ^ prev[k]
                result = _aes_encrypt_block(block, W)
                for k in range(16):
                    out[start + k] = result[k]
                    prev[k] = result[k]
            else:
                enc_block = np.zeros(16, dtype=np.uint8)
                for k in range(16):
                    enc_block[k] = data[start + k]
                result = _aes_decrypt_block(enc_block, W)
                for k in range(16):
                    out[start + k] = result[k] ^ prev[k]
                for k in range(16):
                    prev[k] = enc_block[k]
        else:
            r = remaining
            for k in range(r):
                out[start + k] = data[start + k] ^ prev[k]
        start += 16
    return out


# ---------------------------------------------------------------------------
# Pure-Python fallback helpers
# ---------------------------------------------------------------------------

def galois_multiplication(a: int, b: int) -> int:
    p = 0

    while b:
        if b & 1:
            p ^= a

        high_bit_set = a & 0x80

        a = (a << 1) & 0xFF

        if high_bit_set:
            a ^= 0x1B

        b >>= 1

    return p


def sub_byte(value: int) -> int:
    row = (value >> 4) & 0x0F
    col = value & 0x0F
    return s_box[row][col]


def inv_sub_byte(value: int) -> int:
    row = (value >> 4) & 0x0F
    col = value & 0x0F
    return inv_s_box[row][col]


def expand_key(byte_block: List[int], round_index: int) -> List[int]:
    expanded = byte_block.copy()

    for i in range(4):
        sbyte = sub_byte(byte_block[i])
        expanded[(i + 3) % 4] = (
            sbyte ^ round_constants[round_index][(i + 3) % 4]
        )

    return expanded


def generate_round_keys(key: str):
    round_key_blocks = []

    for i in range(0, 16, 4):
        block = [ord(key[i + j]) for j in range(4)]
        round_key_blocks.append(block)

    block_index = 3

    for rnd in range(10):
        expanded = expand_key(round_key_blocks[block_index], rnd)

        new_block = [
            round_key_blocks[block_index - 3][j] ^ expanded[j]
            for j in range(4)
        ]

        round_key_blocks.append(new_block)
        block_index += 1

        for _ in range(3):
            new_block = [
                round_key_blocks[block_index - 3][j] ^
                round_key_blocks[block_index][j]
                for j in range(4)
            ]

            round_key_blocks.append(new_block)
            block_index += 1

    return round_key_blocks


def add_round_key(state, keys):
    result = [[0] * 4 for _ in range(4)]

    for i in range(4):
        for j in range(4):
            result[i][j] = state[i][j] ^ keys[j][i]

    return result


def sub_bytes_and_shift_rows(state):
    new_state = [[0] * 4 for _ in range(4)]

    for i in range(4):
        for j in range(4):
            sub = sub_byte(state[i][j])
            new_state[i][(j - i) % 4] = sub

    return new_state


def inv_sub_bytes_and_shift_rows(state):
    new_state = [[0] * 4 for _ in range(4)]

    for i in range(4):
        for j in range(4):
            sub = inv_sub_byte(state[i][j])
            new_state[i][(j + i) % 4] = sub

    return new_state


def mix_columns_and_add_round_key(mat1, mat2, keys):
    mixed = [[0] * 4 for _ in range(4)]

    for i in range(4):
        for j in range(4):
            mixed[i][j] = (
                galois_multiplication(mat1[i][0], mat2[0][j]) ^
                galois_multiplication(mat1[i][1], mat2[1][j]) ^
                galois_multiplication(mat1[i][2], mat2[2][j]) ^
                galois_multiplication(mat1[i][3], mat2[3][j]) ^
                keys[j][i]
            )

    return mixed


def inv_mix_columns_and_add_round_key(mat1, mat2, keys):
    mixed = [[0] * 4 for _ in range(4)]

    for i in range(4):
        for j in range(4):
            mixed[i][j] = (
                galois_multiplication(mat1[i][0], mat2[0][j] ^ keys[j][0]) ^
                galois_multiplication(mat1[i][1], mat2[1][j] ^ keys[j][1]) ^
                galois_multiplication(mat1[i][2], mat2[2][j] ^ keys[j][2]) ^
                galois_multiplication(mat1[i][3], mat2[3][j] ^ keys[j][3])
            )

    return mixed


def aes_encrypt(block: List[int], round_keys):
    state = [[0] * 4 for _ in range(4)]
    cipher = []

    for i in range(4):
        for j in range(4):
            state[j][i] = block[i * 4 + j]

    state = add_round_key(state, round_keys[:4])

    for round in range(1, 11):
        state = sub_bytes_and_shift_rows(state)

        current_round_keys = round_keys[
            round * 4:(round + 1) * 4
        ]

        if round < 10:
            state = mix_columns_and_add_round_key(
                fixed_matrix,
                state,
                current_round_keys
            )
        else:
            state = add_round_key(state, current_round_keys)

    for i in range(4):
        for j in range(4):
            cipher.append(state[j][i])

    return cipher


def aes_decrypt(block: List[int], round_keys):
    state = [[0] * 4 for _ in range(4)]
    plain = []

    for i in range(4):
        for j in range(4):
            state[j][i] = block[i * 4 + j]

    reversed_keys = []
    for i in range(len(round_keys) - 4, -1, -4):
        reversed_keys.extend(round_keys[i:i + 4])

    state = add_round_key(state, reversed_keys[:4])

    for round in range(1, 11):
        current_round_key = reversed_keys[
            round * 4:(round + 1) * 4
        ]

        state = inv_sub_bytes_and_shift_rows(state)

        if round < 10:
            state = inv_mix_columns_and_add_round_key(
                inv_fixed_matrix,
                state,
                current_round_key
            )
        else:
            state = add_round_key(state, current_round_key)

    for i in range(4):
        for j in range(4):
            plain.append(state[j][i])

    return plain


def encrypt_pixels(pixels: List[int], round_keys) -> List[int]:
    encrypted_pixels = []
    prev_enc_block = [0] * 16

    for start in range(0, len(pixels), 16):
        plain_chunk = pixels[start:start + 16]
        if len(plain_chunk) == 16:
            block = [plain_chunk[i] ^ prev_enc_block[i] for i in range(16)]
            encrypted_block = aes_encrypt(block, round_keys)
            encrypted_pixels.extend(encrypted_block)
            prev_enc_block = encrypted_block
        else:
            # Last partial block: XOR with previous encrypted block, no AES (output same size as input)
            r = len(plain_chunk)
            encrypted_pixels.extend(plain_chunk[i] ^ prev_enc_block[i] for i in range(r))

    return encrypted_pixels


def decrypt_pixels(pixels: List[int], round_keys, original_length: int) -> List[int]:
    decrypted_pixels = []
    prev_enc_block = [0] * 16

    for start in range(0, len(pixels), 16):
        block = pixels[start:start + 16]
        if len(block) == 16:
            decrypted_block = aes_decrypt(block, round_keys)
            plain_chunk = [decrypted_block[i] ^ prev_enc_block[i] for i in range(16)]
            decrypted_pixels.extend(plain_chunk)
            prev_enc_block = block
        else:
            # Last partial block: XOR back with previous encrypted block
            r = len(block)
            decrypted_pixels.extend(block[i] ^ prev_enc_block[i] for i in range(r))

    return decrypted_pixels[:original_length]


def apply_custom_aes_v3_encrypt(pixels: List[int], key: str) -> List[int]:
    if _NUMBA:
        key_bytes = np.frombuffer(key.encode("latin-1"), dtype=np.uint8)
        W = _key_schedule(key_bytes)
        arr = np.asarray(pixels, dtype=np.uint8)
        return _process_blocks_cbc(arr, W, True).tolist()
    round_keys = generate_round_keys(key)
    return encrypt_pixels(pixels, round_keys)


def apply_custom_aes_v3_decrypt(pixels: List[int], key: str, original_length: int) -> List[int]:
    if _NUMBA:
        key_bytes = np.frombuffer(key.encode("latin-1"), dtype=np.uint8)
        W = _key_schedule(key_bytes)
        arr = np.asarray(pixels, dtype=np.uint8)
        return _process_blocks_cbc(arr, W, False).tolist()
    round_keys = generate_round_keys(key)
    return decrypt_pixels(pixels, round_keys, original_length)


if __name__ == "__main__":
    pixels = np.arange(1, 65).tolist()
    pixels[0] += 1
    enc = apply_custom_aes_v3_encrypt(pixels, "encryptionkey123")
    dec = apply_custom_aes_v3_decrypt(enc, "encryptionkey123", 64)
    print("Encrypted:", len(enc))
    print(enc)
    print("Decrypted:", len(dec))
    print(dec)
