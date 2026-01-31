import hashlib


def calc_md5_hash(input: str):
    return hashlib.md5(input.encode()).hexdigest()
