import os
from pathlib import Path
import numpy as np

from aes_ctr import aes_ctr_transform, derive_aes_key
from aes_cbc import (
    aes_cbc_encrypt_bytes,
    aes_cbc_decrypt_bytes,
)
from aes_gcm import (
    aes_gcm_encrypt_bytes,
    aes_gcm_decrypt_bytes,
)
from aes_ccm import (
    aes_ccm_encrypt_bytes,
    aes_ccm_decrypt_bytes,
)
from chacha20 import (
    chacha20_encrypt_bytes,
    chacha20_decrypt_bytes,
    chacha20_encrypt_array,
    chacha20_decrypt_array,
)
from custom_aes import apply_custom_aes_decrypt, apply_custom_aes_encrypt
from des import derive_des_key, des_ctr_transform
from imagePixels import extract_pixels, reconstruct_image


def encrypt_image(
    input_path: Path,
    output_path: Path,
    nonce_path: Path,
    key_phrase: str,
    algorithm: str = "AES_CTR",
) -> None:
    """Encrypt image pixels with AES-CTR or DES-CTR and save encrypted image plus nonce file.

    algorithm -- case-insensitive string, either 'AES_CTR' or 'DES'. Defaults to 'AES_CTR'.
    If a nonce file already exists, it will be reused instead of generating a new one.
    """

    pixels, mode = extract_pixels(input_path)

    match algorithm.upper():
        case "AES_CTR":
            key = derive_aes_key(key_phrase)
            if nonce_path.exists():
                nonce = nonce_path.read_bytes()
                if len(nonce) != 16:
                    raise ValueError(
                        "Invalid nonce length. Expected 16 bytes for AES-CTR."
                    )
            else:
                nonce = os.urandom(16)
            transformed = aes_ctr_transform(pixels, key, nonce)
        case "DES":
            key = derive_des_key(key_phrase)
            if nonce_path.exists():
                nonce = nonce_path.read_bytes()
                if len(nonce) != 8:
                    raise ValueError(
                        "Invalid nonce length. Expected 8 bytes for DES-CTR."
                    )
            else:
                nonce = os.urandom(8)
            transformed = des_ctr_transform(pixels, key, nonce)
        case "CUSTOM_AES":
            # Use the provided key phrase (padded/truncated to 16 chars) for the custom AES
            key = (key_phrase or "").ljust(16)[:16]

            # Convert numpy pixel array to a flat list[int] as expected by custom AES
            flat_pixels = pixels.flatten().tolist()

            transformed_list = apply_custom_aes_encrypt(flat_pixels, key)

            # Convert back to a numpy array with original shape and dtype for reconstruction
            transformed = np.array(transformed_list, dtype=np.uint8).reshape(
                pixels.shape
            )

            # Custom AES does not use a nonce in this implementation; write an empty nonce
            nonce = b""
        case "AES_CBC":
            # AES-CBC: store IV in nonce file and full ciphertext in a .cbc file.
            key = derive_aes_key(key_phrase)

            if nonce_path.exists():
                iv = nonce_path.read_bytes()
                if len(iv) != 16:
                    raise ValueError(
                        "Invalid IV length. Expected 16 bytes for AES-CBC."
                    )
            else:
                iv = os.urandom(16)

            # Flatten pixels and encrypt bytes
            flat = pixels.astype(np.uint8).tobytes()
            encrypted_bytes = aes_cbc_encrypt_bytes(flat, key, iv)

            # Save full ciphertext alongside the image so we can decrypt later
            ciphertext_path = output_path.with_suffix(".cbc")
            ciphertext_path.parent.mkdir(parents=True, exist_ok=True)
            ciphertext_path.write_bytes(encrypted_bytes)

            # Save metadata (shape and mode and original length) for reconstruction on decrypt
            meta = {
                "shape": list(pixels.shape),
                "mode": mode,
                "length": len(flat),
            }
            import json

            meta_path = output_path.with_suffix(".meta")
            meta_path.write_text(json.dumps(meta))

            # For the displayed/saved image file, write the first N bytes (or pad/truncate)
            orig_len = len(flat)
            disp_bytes = (
                encrypted_bytes[:orig_len]
                if len(encrypted_bytes) >= orig_len
                else encrypted_bytes + bytes(orig_len - len(encrypted_bytes))
            )
            transformed = np.frombuffer(disp_bytes, dtype=np.uint8).reshape(
                pixels.shape
            )

            # nonce file will contain the IV only
            nonce = iv

        case "AES_GCM":
            # AES-GCM: store nonce in nonce file and full ciphertext in a .gcm file.
            key = derive_aes_key(key_phrase)

            if nonce_path.exists():
                nonce = nonce_path.read_bytes()
                if len(nonce) not in (12, 16):
                    # prefer 12-byte nonces, but accept other lengths
                    raise ValueError(
                        "Invalid nonce length. Expected 12 bytes for AES-GCM."
                    )
            else:
                nonce = os.urandom(12)

            flat = pixels.astype(np.uint8).tobytes()
            encrypted_bytes = aes_gcm_encrypt_bytes(flat, key, nonce)

            ciphertext_path = output_path.with_suffix(".gcm")
            ciphertext_path.parent.mkdir(parents=True, exist_ok=True)
            ciphertext_path.write_bytes(encrypted_bytes)

            meta = {
                "shape": list(pixels.shape),
                "mode": mode,
                "length": len(flat),
            }
            import json

            meta_path = output_path.with_suffix(".meta")
            meta_path.write_text(json.dumps(meta))

            orig_len = len(flat)
            disp_bytes = (
                encrypted_bytes[:orig_len]
                if len(encrypted_bytes) >= orig_len
                else encrypted_bytes + bytes(orig_len - len(encrypted_bytes))
            )
            transformed = np.frombuffer(disp_bytes, dtype=np.uint8).reshape(
                pixels.shape
            )
        case "AES_CCM":
            # AES-CCM: store nonce in nonce file and full ciphertext in a .ccm file.
            key = derive_aes_key(key_phrase)

            if nonce_path.exists():
                nonce = nonce_path.read_bytes()
                if not (7 <= len(nonce) <= 13):
                    raise ValueError(
                        "Invalid nonce length. Expected 7..13 bytes for AES-CCM."
                    )
            else:
                # Use an 11-byte nonce by default so AES-CCM can support large images
                # (q = 15 - nonce_len = 4 -> max message ~ 2^(8*4)-1 = ~4GB)
                nonce = os.urandom(11)

            flat = pixels.astype(np.uint8).tobytes()

            # Check AES-CCM message length capacity for the nonce length.
            # q = 15 - nonce_len; max_size = 2^(8*q)-1
            q = 15 - len(nonce)
            max_size = (1 << (8 * q)) - 1
            if len(flat) > max_size:
                if nonce_path.exists():
                    raise ValueError(
                        f"Image too large for existing AES-CCM nonce (len={len(nonce)}). "
                        f"Max message size for this nonce is {max_size} bytes; image is {len(flat)} bytes. "
                        "Delete the .nonce file to regenerate a larger-capacity nonce."
                    )
                else:
                    raise ValueError(
                        f"Image too large for generated AES-CCM nonce (len={len(nonce)}). "
                        f"Max message size is {max_size} bytes; image is {len(flat)} bytes."
                    )
            encrypted_bytes = aes_ccm_encrypt_bytes(flat, key, nonce)

            ciphertext_path = output_path.with_suffix(".ccm")
            ciphertext_path.parent.mkdir(parents=True, exist_ok=True)
            ciphertext_path.write_bytes(encrypted_bytes)

            meta = {
                "shape": list(pixels.shape),
                "mode": mode,
                "length": len(flat),
            }
            import json

            meta_path = output_path.with_suffix(".meta")
            meta_path.write_text(json.dumps(meta))

            orig_len = len(flat)
            disp_bytes = (
                encrypted_bytes[:orig_len]
                if len(encrypted_bytes) >= orig_len
                else encrypted_bytes + bytes(orig_len - len(encrypted_bytes))
            )
            transformed = np.frombuffer(disp_bytes, dtype=np.uint8).reshape(
                pixels.shape
            )
        case "CHACHA20":
            # ChaCha20 stream cipher (no auth). Store full ciphertext in .chacha file
            key = derive_aes_key(key_phrase)

            if nonce_path.exists():
                nonce = nonce_path.read_bytes()
                if len(nonce) != 16:
                    raise ValueError(
                        "Invalid nonce length. Expected 16 bytes for ChaCha20."
                    )
            else:
                nonce = os.urandom(16)

            flat = pixels.astype(np.uint8).tobytes()
            encrypted_bytes = chacha20_encrypt_bytes(flat, key, nonce)

            ciphertext_path = output_path.with_suffix(".chacha")
            ciphertext_path.parent.mkdir(parents=True, exist_ok=True)
            ciphertext_path.write_bytes(encrypted_bytes)

            meta = {
                "shape": list(pixels.shape),
                "mode": mode,
                "length": len(flat),
            }
            import json

            meta_path = output_path.with_suffix(".meta")
            meta_path.write_text(json.dumps(meta))

            orig_len = len(flat)
            disp_bytes = (
                encrypted_bytes[:orig_len]
                if len(encrypted_bytes) >= orig_len
                else encrypted_bytes + bytes(orig_len - len(encrypted_bytes))
            )
            transformed = np.frombuffer(disp_bytes, dtype=np.uint8).reshape(
                pixels.shape
            )
        case _:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    reconstruct_image(transformed, mode, output_path)

    nonce_path.parent.mkdir(parents=True, exist_ok=True)
    nonce_path.write_bytes(nonce)


def decrypt_image(
    input_path: Path,
    output_path: Path,
    nonce_path: Path,
    key_phrase: str,
    algorithm: str = "AES_CTR",
) -> None:
    """Decrypt AES-CTR or DES-CTR encrypted image pixels using the stored nonce file."""

    if not nonce_path.exists():
        raise FileNotFoundError(f"Nonce file not found: {nonce_path}")

    pixels, mode = extract_pixels(input_path)

    match algorithm.upper():
        case "AES_CTR":
            key = derive_aes_key(key_phrase)
            nonce = nonce_path.read_bytes()
            if len(nonce) != 16:
                raise ValueError("Invalid nonce length. Expected 16 bytes for AES-CTR.")
            transformed = aes_ctr_transform(pixels, key, nonce)
        case "DES":
            key = derive_des_key(key_phrase)
            nonce = nonce_path.read_bytes()
            if len(nonce) != 8:
                raise ValueError("Invalid nonce length. Expected 8 bytes for DES-CTR.")
            transformed = des_ctr_transform(pixels, key, nonce)
        case "AES_CBC":
            key = derive_aes_key(key_phrase)
            iv = nonce_path.read_bytes()
            if len(iv) != 16:
                raise ValueError("Invalid IV length. Expected 16 bytes for AES-CBC.")

            ciphertext_path = input_path.with_suffix(".cbc")
            meta_path = input_path.with_suffix(".meta")
            if not ciphertext_path.exists():
                raise FileNotFoundError(f"Ciphertext file not found: {ciphertext_path}")
            if not meta_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {meta_path}")

            encrypted_bytes = ciphertext_path.read_bytes()
            import json

            meta = json.loads(meta_path.read_text())
            shape = tuple(meta.get("shape", []))
            mode = meta.get("mode", mode)

            decrypted = aes_cbc_decrypt_bytes(encrypted_bytes, key, iv)
            transformed = np.frombuffer(decrypted, dtype=np.uint8).reshape(shape)
        case "AES_GCM":
            key = derive_aes_key(key_phrase)
            nonce = nonce_path.read_bytes()
            if len(nonce) not in (12, 16):
                raise ValueError("Invalid nonce length. Expected 12 bytes for AES-GCM.")

            ciphertext_path = input_path.with_suffix(".gcm")
            meta_path = input_path.with_suffix(".meta")
            if not ciphertext_path.exists():
                raise FileNotFoundError(f"Ciphertext file not found: {ciphertext_path}")
            if not meta_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {meta_path}")

            encrypted_bytes = ciphertext_path.read_bytes()
            import json

            meta = json.loads(meta_path.read_text())
            shape = tuple(meta.get("shape", []))
            mode = meta.get("mode", mode)

            decrypted = aes_gcm_decrypt_bytes(encrypted_bytes, key, nonce)
            transformed = np.frombuffer(decrypted, dtype=np.uint8).reshape(shape)
        case "AES_CCM":
            key = derive_aes_key(key_phrase)
            nonce = nonce_path.read_bytes()
            if not (7 <= len(nonce) <= 13):
                raise ValueError(
                    "Invalid nonce length. Expected 7..13 bytes for AES-CCM."
                )

            ciphertext_path = input_path.with_suffix(".ccm")
            meta_path = input_path.with_suffix(".meta")
            if not ciphertext_path.exists():
                raise FileNotFoundError(f"Ciphertext file not found: {ciphertext_path}")
            if not meta_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {meta_path}")

            encrypted_bytes = ciphertext_path.read_bytes()
            import json

            meta = json.loads(meta_path.read_text())
            shape = tuple(meta.get("shape", []))
            mode = meta.get("mode", mode)

            decrypted = aes_ccm_decrypt_bytes(encrypted_bytes, key, nonce)
            transformed = np.frombuffer(decrypted, dtype=np.uint8).reshape(shape)
        case "CHACHA20":
            key = derive_aes_key(key_phrase)
            nonce = nonce_path.read_bytes()
            if len(nonce) != 16:
                raise ValueError(
                    "Invalid nonce length. Expected 16 bytes for ChaCha20."
                )

            ciphertext_path = input_path.with_suffix(".chacha")
            meta_path = input_path.with_suffix(".meta")
            if not ciphertext_path.exists():
                raise FileNotFoundError(f"Ciphertext file not found: {ciphertext_path}")
            if not meta_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {meta_path}")

            encrypted_bytes = ciphertext_path.read_bytes()
            import json

            meta = json.loads(meta_path.read_text())
            shape = tuple(meta.get("shape", []))
            mode = meta.get("mode", mode)

            decrypted = chacha20_decrypt_bytes(encrypted_bytes, key, nonce)
            transformed = np.frombuffer(decrypted, dtype=np.uint8).reshape(shape)
        case "CUSTOM_AES":
            # Use the provided key phrase (padded/truncated to 16 chars) for the custom AES
            key = (key_phrase or "").ljust(16)[:16]

            # Convert numpy pixel array to a flat list[int] as expected by custom AES
            flat_pixels = pixels.flatten().tolist()

            transformed_list = apply_custom_aes_decrypt(flat_pixels, key)

            # Convert back to a numpy array with original shape and dtype for reconstruction
            transformed = np.array(transformed_list, dtype=np.uint8).reshape(
                pixels.shape
            )

        case _:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    reconstruct_image(transformed, mode, output_path)


def main() -> None:
    pass


if __name__ == "__main__":
    main()
