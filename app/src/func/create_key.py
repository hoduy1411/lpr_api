import secrets

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def generate_ecc_keypair():
    private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
    public_key = private_key.public_key()
    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    public_key_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    private_key_base64 = private_key_pem
    public_key_base64 = public_key_pem
    return private_key_base64, public_key_base64


if __name__ == "__main__":
    # Generate ECC key pair
    admin_private_key_base64, admin_public_key_base64 = generate_ecc_keypair()
    user_private_key_base64, user_public_key_base64 = generate_ecc_keypair()

    derived_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b'ACV-API',
        backend=default_backend()
    ).derive(b'ACV-API')


    # Decrypt the ciphertext with AES-GCM
    aesgcm = AESGCM(derived_key)
    
    f = open("key.txt", "w")
    # This part to initial encrypt the secret keys
    nonce = secrets.token_bytes(12)
    print(f'nonce = {nonce}')
    f.write(f'nonce = {nonce}\n')
    ciphertext = aesgcm.encrypt(nonce, admin_private_key_base64, None)
    print(f'encrypted_admin_private_key_base64 = {ciphertext}')
    f.write(f'encrypted_admin_private_key_base64 = {ciphertext}\n')
    ciphertext = aesgcm.encrypt(nonce, admin_public_key_base64, None)
    print(f'encrypted_admin_public_key_base64 = {ciphertext}')
    f.write(f'encrypted_admin_public_key_base64 = {ciphertext}\n')
    ciphertext = aesgcm.encrypt(nonce, user_private_key_base64, None)
    print(f'encrypted_user_private_key_base64 = {ciphertext}')
    f.write(f'encrypted_user_private_key_base64 = {ciphertext}\n')
    ciphertext = aesgcm.encrypt(nonce, user_public_key_base64, None)
    print(f'encrypted_user_public_key_base64 = {ciphertext}')
    f.write(f'encrypted_user_public_key_base64 = {ciphertext}\n')
    
    f.close()
    
