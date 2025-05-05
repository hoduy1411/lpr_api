import base64
import sys
from typing import cast

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

def decrypt_buffer(nonce:bytes, ciphertext:bytes):

    nonce = b'\x84\xc33J\x18\x8d\xf9W\x9e\xdes\xa3'
    encrypted_admin_public_key_base64 = b"\xb6{~%J\x80\x93cz\xdc\xa9/\x8c\xf87\xc7\xff\x81\x18^\x9c\xc68\xe4a\xba\x04\xa5\x1f(N\xe3h\xe8\xa0\xa0\xbfjNb\xfa\x07N\x16\xeci\xed5\xaf>\x92\x04\x0f\xb3\x1b\xa6Y\xc1=\xcb\x89f\x92\x90\t\x9b\xb6\xfaQ\xa9\xbd\x10M{\xae\xfb\xdb\x84\t\xc7_\xdb>\xf9\xd5\xbar3y\xda$\x14\x0e\x9f\xd7\xff\x10 ]\xad\xb5\x90\x15\xa3\xba\x9f\x0c4\xebr\xa0\x94\xbac\x1f\xad\xe1s\r\xb7\xc3!\xc5\xef\xfb\x13\x9c0\x9dy;\x8d\xb9\xe5\xd1\xa9d\xdf\xeb\x99\xc7\x12\x91&Y\xcf\xf9Y\x84\x97\xe5e\xfe\xb1\x10\xb7\x8eA,m)\xcaM\x03\x16\x03\\\xd3O\xcfe\x87a\x00\xea\x81\xa7Qb\x12\xca@\xfb\xe1w<'\xbd.\xf4\x0bNa\xf4"
    encrypted_user_private_key_base64 = b"\xb6{~%J\x80\x93cz\xdc\xa9/\x8b\xf3-\xcf\xe8\xe4sP\x80\xb28\xe4a\xba#\xe2\x14\n~\xee^\xd6\xad\xaa\x9drJ_\xd2N|\x10\xee}\x90G\x81\x03\x9e9&\xc0.\xa0[\xefM\xa3\xafP\x92\xab\x13\xa6\xca\xef^\xe7\x9c\x0b:\x1d\xbe\xff\xfd\x87\x1e\xdae\xe89\xdd\xfb\xdff7Y\xff0}}\x93\x89\x97:`\x7f\xcc\xd3\xa7c\x8a\x80\x827\x0e\xd7u\x96\xee\xb0^\x0e\x96\xef<2\xb7\xdd\x1b\xef\xf6\xfd\x17\x97V\xaab-\xac\xbc\xef\xa6\x94Y\xb0\x8b\xca\xe6q\xe8'9\xc9\xa3m\x90\xf5\xaa=\x92\xa4l\xf8\xebf1\x14)\x8f)%2;^\xf1\x07\xe7x\xa6%b\x9f\x9d\xec>/\xeb\x9d\xb0\xb2\xb3\xc3\xee\xf5\xa6\xdfr\xe8\x9e1\xa1V\xcefu\xab\xd1\xb0\x86\x04\xaf\xd9\x04D\xbc\xa1\x14\xdd\x15\xea\x87o\xdd\xab\xb1\xabB\xe2Z\x87\xc1\xeb\xd0\xe1\xe6i\xa0\xa5N\x99\x7f\x05f6j}b\xd6\xa2\xf6\xc9ir\xdd.\xfd4\x92zg\xde\x11r\x1f"

    derived_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b'ACV-API',
        backend=default_backend()
    ).derive(b'ACV-API')


    # Decrypt the ciphertext with AES-GCM
    aesgcm = AESGCM(derived_key)
    admin_public_key_base64 = aesgcm.decrypt(nonce, encrypted_admin_public_key_base64, None)
    user_private_key_base64 = aesgcm.decrypt(nonce, encrypted_user_private_key_base64, None)

    # Decode public key from base64
    public_key = serialization.load_pem_public_key(admin_public_key_base64, backend=default_backend())

    # Decode private key from base64
    private_key = serialization.load_pem_private_key(user_private_key_base64, password=None, backend=default_backend())

    public_key = cast(ec.EllipticCurvePublicKey, public_key)
    private_key = cast(ec.EllipticCurvePrivateKey, private_key)

    # Generate a shared secret using ECDH
    shared_key = private_key.exchange(ec.ECDH(), public_key)

    # Derive a symmetric key from the shared secret
    derived_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b'ECC example',
        backend=default_backend()
    ).derive(shared_key)

    # Decrypt the ciphertext with AES-GCM
    aesgcm = AESGCM(derived_key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    return plaintext
