import hashlib
import blake3
import binascii
import zlib

def generate_file_hashes(file_path):
    # Initialize hash objects
    sha256_hash = hashlib.sha256()
    md5_hash = hashlib.md5()
    blake3_hasher = blake3.blake3()
    blake2_hash = hashlib.blake2b(digest_size=32)
    crc32_hash = 0  # Initialize CRC32

    # Process the file in chunks for efficiency
    chunk_size = 8192  # You can adjust this based on your needs

    with open(file_path, "rb") as file:
        hash_sha256 = hashlib.sha256()
        blksize = 1024 * 1024

        file.seek(0)
        header = file.read(8)
        n = int.from_bytes(header, "little")

        offset = n + 8
        file.seek(offset)
        for chunk in iter(lambda: file.read(blksize), b""):
            hash_sha256.update(chunk)
        
        addnet = hash_sha256.hexdigest()[0:8]

    with open(file_path, "rb") as file:
        m = hashlib.sha256()
        file.seek(0x100000)
        m.update(file.read(0x10000))
        model_hash =  m.hexdigest()[0:8]
    
    with open(file_path, 'rb') as file:
        while chunk := file.read(chunk_size):
            sha256_hash.update(chunk)
            md5_hash.update(chunk)
            blake3_hasher.update(chunk)
            blake2_hash.update(chunk)
            crc32_hash = zlib.crc32(chunk, crc32_hash)

    # Get the hexadecimal representations of the hashes
    sha256_hex = sha256_hash.hexdigest()
    md5_hex = md5_hash.hexdigest()
    blake3_hex = blake3_hasher.hexdigest()
    blake2_hex = blake2_hash.hexdigest()
    crc32_hex = format(crc32_hash & 0xFFFFFFFF, '08x')

    return {
        'SHA256': sha256_hex,
        'MD5': md5_hex,
        'BLAKE3': blake3_hex,
        'BLAKE2': blake2_hex,
        'CRC32': crc32_hex,
        'model_hash' : model_hash,
        'add_net' : addnet,
    }


model = "X:/dif/stable-diffusion-webui-docker/data/models/Lora/Concepts/add_detail.safetensors"

#7c6bad76eb54

print(str(generate_file_hashes(model)))

