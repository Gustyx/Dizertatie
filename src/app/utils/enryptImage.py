import os
import json
from pathlib import Path
import numpy as np
from PIL import Image

from ..algorithms import (
    aes_ctr_transform,
    derive_aes_key,
    aes_cbc_encrypt_bytes,
    aes_cbc_decrypt_bytes,
    aes_gcm_encrypt_bytes,
    aes_gcm_decrypt_bytes,
    aes_ccm_encrypt_bytes,
    aes_ccm_decrypt_bytes,
    chacha20_encrypt_bytes,
    chacha20_decrypt_bytes,
    apply_custom_aes_decrypt,
    apply_custom_aes_encrypt,
    apply_custom_aes_v2_decrypt,
    apply_custom_aes_v2_encrypt,
    apply_custom_aes_v4_decrypt,
    apply_custom_aes_v4_encrypt,
    apply_custom_aes_v5_decrypt,
    apply_custom_aes_v5_encrypt,
    apply_logistic_map_decrypt,
    apply_logistic_map_encrypt,
    apply_henon_map_decrypt,
    apply_henon_map_encrypt,
    derive_des_key,
    des_ctr_transform,
)
from .handle_image_pixels import extract_pixels, reconstruct_image

_ROOT_DIR = Path(__file__).resolve().parent.parent / "shared"
_META_DIR = _ROOT_DIR / "meta_files"
_NONCE_DIR = _ROOT_DIR / "nonce_files"


def _meta_path_for_image_path(image_path: Path) -> Path:
    # Keep metadata in a single folder while preserving a unique, reversible name.
    meta_name = f"{image_path.stem}.meta"
    return _META_DIR / meta_name


def _nonce_path_for_image_path(image_path: Path) -> Path:
    # Keep nonce files in a single folder while preserving a unique, reversible name.
    nonce_name = f"{image_path.stem}.nonce"
    return _NONCE_DIR / nonce_name


def _read_or_create_nonce(
    output_path: Path,
    expected_lengths: tuple[int, ...],
    default_length: int,
    label: str,
) -> bytes:
    nonce_path = _nonce_path_for_image_path(output_path)
    print(f"Using nonce path: {nonce_path}")
    if nonce_path.exists():
        nonce = nonce_path.read_bytes()
        if len(nonce) not in expected_lengths:
            expected = " or ".join(str(length) for length in expected_lengths)
            raise ValueError(f"Invalid {label} length. Expected {expected} bytes.")
        return nonce

    nonce = os.urandom(default_length)
    nonce_path.parent.mkdir(parents=True, exist_ok=True)
    nonce_path.write_bytes(nonce)
    return nonce


def _write_ciphertext_and_meta(
    output_path: Path,
    ciphertext_suffix: str | None,
    encrypted_bytes: bytes,
    pixels: np.ndarray,
    mode: str,
    original_length: int,
) -> None:
    # Optionally write ciphertext to a separate file when a suffix is provided.
    if ciphertext_suffix:
        ciphertext_path = output_path.with_suffix(ciphertext_suffix)
        ciphertext_path.parent.mkdir(parents=True, exist_ok=True)
        ciphertext_path.write_bytes(encrypted_bytes)

    meta = {
        "shape": list(pixels.shape),
        "mode": mode,
        "length": original_length,
        "ciphertext_length": len(encrypted_bytes),
    }
    _META_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = _meta_path_for_image_path(output_path)
    meta_path.write_text(json.dumps(meta))


def _load_ciphertext_and_meta(
    input_path: Path, ciphertext_suffix: str
) -> tuple[bytes, tuple[int, ...], str]:
    ciphertext_path = input_path.with_suffix(ciphertext_suffix)
    meta_path = _meta_path_for_image_path(input_path)
    legacy_meta_path = input_path.with_suffix(".meta")
    if not ciphertext_path.exists():
        raise FileNotFoundError(f"Ciphertext file not found: {ciphertext_path}")
    if not meta_path.exists() and legacy_meta_path.exists():
        meta_path = legacy_meta_path
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    encrypted_bytes = ciphertext_path.read_bytes()
    meta = json.loads(meta_path.read_text())
    shape = tuple(meta.get("shape", []))
    mode = meta.get("mode")
    return encrypted_bytes, shape, mode


def _display_bytes(encrypted_bytes: bytes, original_length: int) -> bytes:
    return (
        encrypted_bytes[:original_length]
        if len(encrypted_bytes) >= original_length
        else encrypted_bytes + bytes(original_length - len(encrypted_bytes))
    )


def _ciphertext_canvas_shape(
    ciphertext_length: int, source_shape: tuple[int, ...]
) -> tuple[int, int]:
    """Choose a 2D canvas shape that preserves the source image aspect ratio roughly."""

    if len(source_shape) < 2:
        return 1, ciphertext_length

    source_height = max(1, int(source_shape[0]))
    source_width = max(1, int(source_shape[1]))
    aspect_ratio = source_width / source_height

    canvas_height = max(1, int(np.ceil(np.sqrt(ciphertext_length / aspect_ratio))))
    canvas_width = max(1, int(np.ceil(ciphertext_length / canvas_height)))
    return canvas_height, canvas_width


def _save_ciphertext_image(
    output_path: Path, encrypted_bytes: bytes, source_shape: tuple[int, ...]
) -> None:
    enc_arr = np.frombuffer(encrypted_bytes, dtype=np.uint8)
    height, width = _ciphertext_canvas_shape(len(enc_arr), source_shape)
    padded = np.zeros((height, width), dtype=np.uint8)
    padded.flat[: len(enc_arr)] = enc_arr

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(padded, mode="L").save(output_path)


def _delete_legacy_ciphertext_sidecars(
    output_path: Path, suffixes: tuple[str, ...]
) -> None:
    for suffix in suffixes:
        sidecar = output_path.with_suffix(suffix)
        if sidecar.exists():
            sidecar.unlink()


def _load_ciphertext_from_image(input_path: Path) -> bytes:
    meta_path = _meta_path_for_image_path(input_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    meta = json.loads(meta_path.read_text())
    ciphertext_length = int(meta.get("ciphertext_length", 0))
    encrypted_pixels, _ = extract_pixels(input_path)
    return encrypted_pixels.tobytes()[:ciphertext_length]


def encrypt_image(
    input_path: Path,
    output_path: Path,
    key_phrase: str,
    algorithm: str = "AES_CTR",
) -> None:
    pixels, mode = extract_pixels(input_path)
    flat = pixels.astype(np.uint8).tobytes()

    match algorithm.upper():
        case "AES_CTR":
            key = derive_aes_key(key_phrase)
            nonce = _read_or_create_nonce(output_path, (16,), 16, "nonce")
            transformed = aes_ctr_transform(pixels, key, nonce)
        case "TRIPLE_DES":
            key = derive_des_key(key_phrase)
            nonce = _read_or_create_nonce(output_path, (8,), 8, "nonce")
            transformed = des_ctr_transform(pixels, key, nonce)
        case "CUSTOM_AES":
            # Use the provided key phrase (padded/truncated to 16 chars) for the custom AES
            key = (key_phrase or "").ljust(16)[:16]

            # Convert numpy pixel array to a flat list[int] as expected by custom AES
            flat_pixels = pixels.flatten().tolist()
            print(len(flat_pixels))
            transformed_list = apply_custom_aes_v4_encrypt(flat_pixels, key)
            print(len(transformed_list))

            if len(transformed_list) % 4 != 0:
                transformed_list.extend([0] * (4 - len(transformed_list) % 4))

            # Convert back to a numpy array with original shape and dtype for reconstruction
            print(pixels.shape)
            transformed = np.array(transformed_list, dtype=np.uint8).reshape(
                (304, 921, 4)
            )

            # Custom AES does not use a nonce in this implementation; write an empty nonce
            nonce = b""
        case "LOGISTIC_MAP":
            # Logistic map chaotic encryption uses the key phrase directly.
            flat_pixels = pixels.flatten().tolist()
            transformed_list = apply_logistic_map_encrypt(flat_pixels, key_phrase)
            transformed = np.array(transformed_list, dtype=np.uint8).reshape(
                pixels.shape
            )
            nonce = b""
        case "HENON_MAP":
            flat_pixels = pixels.flatten().tolist()
            transformed_list = apply_henon_map_encrypt(flat_pixels, key_phrase)
            transformed = np.array(transformed_list, dtype=np.uint8).reshape(
                pixels.shape
            )
            nonce = b""
        case "AES_CBC":
            # AES-CBC: store IV in nonce file and full ciphertext in a .cbc file.
            key = derive_aes_key(key_phrase)

            iv = _read_or_create_nonce(output_path, (16,), 16, "IV")

            # Flatten pixels and encrypt bytes
            encrypted_bytes = aes_cbc_encrypt_bytes(flat, key, iv)
            # Write metadata (including ciphertext length). Do not write a separate .cbc file;
            # instead embed the full ciphertext into the saved image as an uncompressed single-channel image.
            _write_ciphertext_and_meta(
                output_path, None, encrypted_bytes, pixels, mode, len(flat)
            )

            # Create a single-channel image that contains the full ciphertext bytes.
            # Save the ciphertext image directly as a single-channel image.
            # Do not pass it through reconstruct_image because the original mode/shape no longer apply.
            _save_ciphertext_image(output_path, encrypted_bytes, pixels.shape)
            return

            # nonce file will contain the IV only
            nonce = iv

        case "AES_GCM":
            # AES-GCM: store nonce in nonce file and embed ciphertext in the output image.
            key = derive_aes_key(key_phrase)

            nonce = _read_or_create_nonce(output_path, (12, 16), 12, "nonce")

            encrypted_bytes = aes_gcm_encrypt_bytes(flat, key, nonce)

            _write_ciphertext_and_meta(
                output_path, None, encrypted_bytes, pixels, mode, len(flat)
            )

            _delete_legacy_ciphertext_sidecars(output_path, (".gcm",))
            _save_ciphertext_image(output_path, encrypted_bytes, pixels.shape)
            return
        case "AES_CCM":
            # AES-CCM: store nonce in nonce file and embed ciphertext in the output image.
            key = derive_aes_key(key_phrase)

            nonce = _read_or_create_nonce(output_path, tuple(range(7, 14)), 11, "nonce")

            # Check AES-CCM message length capacity for the nonce length.
            # q = 15 - nonce_len; max_size = 2^(8*q)-1
            q = 15 - len(nonce)
            max_size = (1 << (8 * q)) - 1
            if len(flat) > max_size:
                raise ValueError(
                    f"Image too large for existing AES-CCM nonce (len={len(nonce)}). "
                    f"Max message size for this nonce is {max_size} bytes; image is {len(flat)} bytes. "
                    "Delete the .nonce file to regenerate a larger-capacity nonce."
                )
            encrypted_bytes = aes_ccm_encrypt_bytes(flat, key, nonce)

            _write_ciphertext_and_meta(
                output_path, None, encrypted_bytes, pixels, mode, len(flat)
            )

            _delete_legacy_ciphertext_sidecars(output_path, (".ccm",))
            _save_ciphertext_image(output_path, encrypted_bytes, pixels.shape)
            return
        case "CHACHA20":
            # ChaCha20 stream cipher (no auth). Embed ciphertext in the output image.
            key = derive_aes_key(key_phrase)

            nonce = _read_or_create_nonce(output_path, (16,), 16, "nonce")

            encrypted_bytes = chacha20_encrypt_bytes(flat, key, nonce)

            _write_ciphertext_and_meta(
                output_path, None, encrypted_bytes, pixels, mode, len(flat)
            )

            _delete_legacy_ciphertext_sidecars(output_path, (".chacha",))
            _save_ciphertext_image(output_path, encrypted_bytes, pixels.shape)
            return
        case _:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    reconstruct_image(transformed, mode, output_path)


def decrypt_image(
    input_path: Path,
    output_path: Path,
    key_phrase: str,
    algorithm: str = "AES_CTR",
) -> None:
    """Decrypt AES-CTR or DES-CTR encrypted image pixels using the stored nonce file."""

    nonce_path = _nonce_path_for_image_path(input_path)
    # if not nonce_path.exists():
    # raise FileNotFoundError(f"Nonce file not found: {nonce_path}")

    pixels, mode = extract_pixels(input_path)

    match algorithm.upper():
        case "AES_CTR":
            key = derive_aes_key(key_phrase)
            nonce = nonce_path.read_bytes()
            print(nonce_path)
            if len(nonce) != 16:
                raise ValueError("Invalid nonce length. Expected 16 bytes for AES-CTR.")
            transformed = aes_ctr_transform(pixels, key, nonce)
        case "TRIPLE_DES":
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

            meta_path = _meta_path_for_image_path(input_path)
            if not meta_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {meta_path}")
            meta = json.loads(meta_path.read_text())
            target_shape = tuple(meta.get("shape", [])) or pixels.shape
            mode = meta.get("mode", mode)

            # Try loading a separate ciphertext file; if missing, read the ciphertext from the image pixels.
            try:
                encrypted_bytes, shape, loaded_mode = _load_ciphertext_and_meta(
                    input_path, ".cbc"
                )
                mode = loaded_mode or mode
                if shape:
                    target_shape = shape
            except FileNotFoundError:
                # Load ciphertext length from metadata and read pixels from the image file.
                ciphertext_length = int(meta.get("ciphertext_length", 0))

                enc_pixels, enc_mode = extract_pixels(input_path)
                encrypted_bytes = enc_pixels.tobytes()[:ciphertext_length]

            decrypted = aes_cbc_decrypt_bytes(encrypted_bytes, key, iv)
            transformed = np.frombuffer(decrypted, dtype=np.uint8).reshape(target_shape)
        case "AES_GCM":
            key = derive_aes_key(key_phrase)
            nonce = nonce_path.read_bytes()
            if len(nonce) not in (12, 16):
                raise ValueError("Invalid nonce length. Expected 12 bytes for AES-GCM.")

            meta_path = _meta_path_for_image_path(input_path)
            if not meta_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {meta_path}")
            meta = json.loads(meta_path.read_text())
            target_shape = tuple(meta.get("shape", [])) or pixels.shape
            mode = meta.get("mode", mode)

            try:
                encrypted_bytes, shape, loaded_mode = _load_ciphertext_and_meta(
                    input_path, ".gcm"
                )
                mode = loaded_mode or mode
                if shape:
                    target_shape = shape
            except FileNotFoundError:
                encrypted_bytes = _load_ciphertext_from_image(input_path)

            decrypted = aes_gcm_decrypt_bytes(encrypted_bytes, key, nonce)
            transformed = np.frombuffer(decrypted, dtype=np.uint8).reshape(target_shape)
        case "AES_CCM":
            key = derive_aes_key(key_phrase)
            nonce = nonce_path.read_bytes()
            if not (7 <= len(nonce) <= 13):
                raise ValueError(
                    "Invalid nonce length. Expected 7..13 bytes for AES-CCM."
                )

            meta_path = _meta_path_for_image_path(input_path)
            if not meta_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {meta_path}")
            meta = json.loads(meta_path.read_text())
            target_shape = tuple(meta.get("shape", [])) or pixels.shape
            mode = meta.get("mode", mode)

            try:
                encrypted_bytes, shape, loaded_mode = _load_ciphertext_and_meta(
                    input_path, ".ccm"
                )
                mode = loaded_mode or mode
                if shape:
                    target_shape = shape
            except FileNotFoundError:
                encrypted_bytes = _load_ciphertext_from_image(input_path)

            decrypted = aes_ccm_decrypt_bytes(encrypted_bytes, key, nonce)
            transformed = np.frombuffer(decrypted, dtype=np.uint8).reshape(target_shape)
        case "CHACHA20":
            key = derive_aes_key(key_phrase)
            nonce = nonce_path.read_bytes()
            if len(nonce) != 16:
                raise ValueError(
                    "Invalid nonce length. Expected 16 bytes for ChaCha20."
                )

            meta_path = _meta_path_for_image_path(input_path)
            if not meta_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {meta_path}")
            meta = json.loads(meta_path.read_text())
            target_shape = tuple(meta.get("shape", [])) or pixels.shape
            mode = meta.get("mode", mode)

            try:
                encrypted_bytes, shape, loaded_mode = _load_ciphertext_and_meta(
                    input_path, ".chacha"
                )
                mode = loaded_mode or mode
                if shape:
                    target_shape = shape
            except FileNotFoundError:
                encrypted_bytes = _load_ciphertext_from_image(input_path)

            decrypted = chacha20_decrypt_bytes(encrypted_bytes, key, nonce)
            transformed = np.frombuffer(decrypted, dtype=np.uint8).reshape(target_shape)
        case "CUSTOM_AES":
            # Use the provided key phrase (padded/truncated to 16 chars) for the custom AES
            key = (key_phrase or "").ljust(16)[:16]

            # Convert numpy pixel array to a flat list[int] as expected by custom AES
            flat_pixels = pixels.flatten().tolist()

            transformed_list = apply_custom_aes_v4_decrypt(flat_pixels, key)

            # Convert back to a numpy array with original shape and dtype for reconstruction
            transformed = np.array(transformed_list, dtype=np.uint8).reshape(
                pixels.shape
            )

        case "LOGISTIC_MAP":
            flat_pixels = pixels.flatten().tolist()
            transformed_list = apply_logistic_map_decrypt(flat_pixels, key_phrase)
            transformed = np.array(transformed_list, dtype=np.uint8).reshape(
                pixels.shape
            )

        case "HENON_MAP":
            flat_pixels = pixels.flatten().tolist()
            transformed_list = apply_henon_map_decrypt(flat_pixels, key_phrase)
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
