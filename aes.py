import base64
import hashlib
import os
from typing import Optional
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

key = "encryptionkey"

def encrypt_text(plaintext: str, key_phrase: str) -> str:
    """Encrypt plaintext with AES-GCM and return a Base64 string.

    The provided key phrase is hashed to a 256-bit AES key because
    "encryptionkey" is not a valid raw AES key length.
    """

    key = hashlib.sha256(key_phrase.encode("utf-8")).digest()
    nonce = os.urandom(12)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext.encode("utf-8"), None)
    return base64.b64encode(nonce + ciphertext).decode("ascii")


def try_decrypt_text(encoded_value: str, key_phrase: str) -> Optional[str]:
    """Try to decrypt a Base64 AES-GCM value and return plaintext on success."""

    try:
        raw = base64.b64decode(encoded_value.encode("ascii"), validate=True)
        if len(raw) < 13:
            return None

        key = hashlib.sha256(key_phrase.encode("utf-8")).digest()
        nonce = raw[:12]
        ciphertext = raw[12:]
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)
        return plaintext.decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None


def main() -> None:
    print("Do you want to 1. encrypt or 2. decrypt?")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        plaintext = input("Enter text to encrypt: ")
        encrypted_value = encrypt_text(plaintext, key)
        print(encrypted_value)
        return

    if choice == "2":
        encoded_value = input("Enter text to decrypt: ")
        decrypted_value = try_decrypt_text(encoded_value, key)
        if decrypted_value is None:
            print("Could not decrypt the input.")
            return

        print(decrypted_value)
        return

    print("Invalid choice. Please run the program again and enter 1 or 2.")


if __name__ == "__main__":
    main()
