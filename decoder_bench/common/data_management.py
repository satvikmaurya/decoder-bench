from itertools import groupby
import numpy as np

def rle_encode(bool_array):
    """Encode using run-length encoding"""
    return [(len(list(group)), int(key)) for key, group in groupby(bool_array)]

def rle_decode(encoded):
    """Decode run-length encoding"""
    return np.array([bit for count, bit in encoded for _ in range(count)], dtype=bool)

def hash_bool_array(bool_array):
    """Convert boolean array to integer"""
    return int(''.join(map(str, bool_array.astype(int))), 2)

def unhash_bool_array(hash_val, length):
    """Convert integer back to boolean array"""
    binary = format(hash_val, f'0{length}b')
    return np.array([int(b) for b in binary], dtype=bool)

def pack_bits(bool_array):
    """Pack boolean array into uint8 array"""
    return np.packbits(bool_array)

def unpack_bits(packed_array, original_length):
    """Unpack uint8 array back to boolean array"""
    return np.unpackbits(packed_array)[:original_length].astype(np.bool_)