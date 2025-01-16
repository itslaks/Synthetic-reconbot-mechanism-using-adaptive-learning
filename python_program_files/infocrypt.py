from flask import Blueprint, request, jsonify, render_template
import hashlib
import os
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.fernet import Fernet

# Create a blueprint
infocrypt = Blueprint('infocrypt', __name__, template_folder='templates')

@infocrypt.route('/')
def index():
    return render_template('infocrypt.html')

def hash_data(data, algorithm):
    # Previous hash_data function remains the same
    try:
        if algorithm == 'CRC32':
            return format(hashlib.new('crc32', data.encode()).digest(), 'x')
        elif algorithm == 'SHA-256':
            return hashlib.sha256(data.encode()).hexdigest()
        elif algorithm == 'SHA-1':
            return hashlib.sha1(data.encode()).hexdigest()
        elif algorithm == 'SHA3-256':
            return hashlib.sha3_256(data.encode()).hexdigest()
        elif algorithm == 'BLAKE2b':
            return hashlib.blake2b(data.encode()).hexdigest()
        elif algorithm == 'SHAKE-128':
            h = hashlib.shake_128(data.encode())
            return h.digest(32).hex()
        elif algorithm == 'SHA-512':
            return hashlib.sha512(data.encode()).hexdigest()
        elif algorithm == 'SHA-384':
            return hashlib.sha384(data.encode()).hexdigest()
        else:
            return None
    except Exception as e:
        return f"Hashing Error: {str(e)}"

def encrypt_data(data, algorithm, key=None):
    try:
        if algorithm in ['AES-128', 'AES-256']:
            if key:
                # Use provided key
                key = key.encode()
                if algorithm == 'AES-128':
                    key = key[:16].ljust(16, b'\0')  # Pad to 16 bytes
                else:
                    key = key[:32].ljust(32, b'\0')  # Pad to 32 bytes
            else:
                # Generate random key
                key_size = 16 if algorithm == 'AES-128' else 32
                key = os.urandom(key_size)
            
            iv = os.urandom(16)
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            padder = padding.PKCS7(algorithms.AES.block_size).padder()
            padded_data = padder.update(data.encode()) + padder.finalize()
            encrypted = encryptor.update(padded_data) + encryptor.finalize()
            return base64.b64encode(iv + key + encrypted).decode()

        elif algorithm == 'ChaCha20':
            if key:
                key = key.encode()[:32].ljust(32, b'\0')  # Pad to 32 bytes
            else:
                key = os.urandom(32)
            nonce = os.urandom(16)
            cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None, backend=default_backend())
            encryptor = cipher.encryptor()
            encrypted = encryptor.update(data.encode()) + encryptor.finalize()
            return base64.b64encode(nonce + key + encrypted).decode()

        elif algorithm == 'Fernet':
            if key:
                try:
                    # Try to use provided key
                    key_bytes = base64.b64decode(key)
                    if len(key_bytes) != 32:
                        raise ValueError("Invalid key length")
                    f = Fernet(key.encode())
                except:
                    # If key is invalid, generate new one
                    key = Fernet.generate_key()
                    f = Fernet(key)
            else:
                key = Fernet.generate_key()
                f = Fernet(key)
            
            encrypted = f.encrypt(data.encode())
            return base64.b64encode(key + encrypted).decode()

        elif algorithm == 'RSA':
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            public_key = private_key.public_key()
            encrypted = public_key.encrypt(
                data.encode(),
                asym_padding.OAEP(
                    mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            private_pem_b64 = base64.b64encode(private_pem).decode()
            encrypted_b64 = base64.b64encode(encrypted).decode()
            return private_pem_b64 + ":" + encrypted_b64
        else:
            return None
    except Exception as e:
        return f"Encryption Error: {str(e)}"

def decrypt_data(encrypted_data, algorithm, key=None):
    try:
        if algorithm in ['AES-128', 'AES-256']:
            decoded_data = base64.b64decode(encrypted_data)
            key_size = 16 if algorithm == 'AES-128' else 32
            iv = decoded_data[:16]
            if key:
                key = key.encode()[:key_size].ljust(key_size, b'\0')
            else:
                key = decoded_data[16:16 + key_size]
            ciphertext = decoded_data[16 + key_size:]
            
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            padded_data = decryptor.update(ciphertext) + decryptor.finalize()
            unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
            data = unpadder.update(padded_data) + unpadder.finalize()
            return data.decode()
        
        elif algorithm == 'ChaCha20':
            decoded_data = base64.b64decode(encrypted_data)
            nonce = decoded_data[:16]
            if key:
                key = key.encode()[:32].ljust(32, b'\0')
            else:
                key = decoded_data[16:48]
            ciphertext = decoded_data[48:]
            
            cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None, backend=default_backend())
            decryptor = cipher.decryptor()
            decrypted = decryptor.update(ciphertext) + decryptor.finalize()
            return decrypted.decode()

        elif algorithm == 'Fernet':
            decoded_data = base64.b64decode(encrypted_data)
            if key:
                key = key.encode()
            else:
                key = decoded_data[:44]
                decoded_data = decoded_data[44:]
            f = Fernet(key)
            decrypted = f.decrypt(decoded_data)
            return decrypted.decode()
        
        elif algorithm == 'RSA':
            private_pem_b64, encrypted_b64 = encrypted_data.split(":")
            private_pem = base64.b64decode(private_pem_b64)
            ciphertext = base64.b64decode(encrypted_b64)
            
            private_key = serialization.load_pem_private_key(
                private_pem,
                password=None,
                backend=default_backend()
            )
            plaintext = private_key.decrypt(
                ciphertext,
                asym_padding.OAEP(
                    mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return plaintext.decode()
        
        else:
            return None
    except Exception as e:
        return f"Decryption Error: {str(e)}"

@infocrypt.route('/process', methods=['POST'])
def process_request():
    data = request.json.get('text')
    algorithm = request.json.get('algorithm')
    action = request.json.get('action')
    key = request.json.get('key')  # New parameter for custom key

    if not data or not algorithm or not action:
        return jsonify({'error': 'Text, algorithm, and action must be provided'}), 400

    result = None
    if action == 'hash':
        result = hash_data(data, algorithm)
    elif action == 'encrypt':
        result = encrypt_data(data, algorithm, key)
    elif action == 'decrypt':
        result = decrypt_data(data, algorithm, key)
    else:
        return jsonify({'error': 'Invalid action'}), 400

    if result is None:
        return jsonify({'error': 'Invalid algorithm or action'}), 400

    return jsonify({'result': result})

if __name__ == '__main__':
    infocrypt.run(debug=True)