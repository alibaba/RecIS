import base64
import os

from Crypto import Random
from Crypto.Cipher import AES, PKCS1_v1_5
from Crypto.Hash import SHA256, MD5
from Crypto.PublicKey import RSA

from column_io.dataset.log_util import logger, varlogger, LOG_DIR

PRIVATE_KEY = os.getenv("NEBULA_ENCODE_ACCESS_PRIVATE_KEY")
PUBLIC_KEY = "LS0tLS1CRUdJTiBQVUJMSUMgS0VZLS0tLS0KTUlHZk1BMEdDU3FHU0liM0RRRUJBUVVBQTRHTkFEQ0JpUUtCZ1FDM3FoV09WNGc2cS9hVmUyZFE3RzUrd1A2cgpMUlhqYTQ1STVqdkxFTmQxSkNhWmFDdldqbzZ6cVRkbmtreUVXWGFEMnRBNE4vZmhYNW9rTyt0SE5oZU0yVjdvCkJtMmxUaWVVTTVtV2VQTDMwd1N0Z25mQjJrRVgvTExudXdENFdlZGtwU0c0NjZKOFRPNmNUbWxCc1Y2anp2aGQKSzhTWEpVeE1STEtvVFI0aHVRSURBUUFCCi0tLS0tRU5EIFBVQkxJQyBLRVktLS0tLQ=="

def decode(data, private_key=PRIVATE_KEY, binary_code='utf-8'):
    if data is None or str(data) == '':
        logger.info("empty data passed into decode function in open_storage_utils, skip decode...")
        return None
    if private_key is None or str(private_key) == '':
        logger.error("empty private_key in open_storage_utils for ENCODED_ODPS_ACCESS_ID and KEY")
    data = base64.b64decode(data.encode(binary_code))
    private_key = base64.b64decode(private_key)
    cipher = PKCS1_v1_5.new(RSA.import_key(private_key))
    dsize = SHA256.digest_size
    sentinel = Random.new().read(15 + dsize)
    decrypted = cipher.decrypt(data, sentinel)
    return decrypted.decode(binary_code)

def encode(data, public_key=PUBLIC_KEY, binary_code='utf-8'):
    pk = os.getenv("PUBLIC_KEY", None)
    if pk is not None:
        logger.warning("env PUBLIC_KEY is not None, assign public_key to new value.")
        public_key = pk
    public_key = base64.b64decode(public_key)
    data = data.encode(binary_code)
    cipher = PKCS1_v1_5.new(RSA.import_key(public_key))
    encrypted = cipher.encrypt(data)
    return base64.b64encode(encrypted).decode(binary_code)

