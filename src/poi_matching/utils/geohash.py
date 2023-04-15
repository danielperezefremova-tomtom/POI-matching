import pygeohash as pgh
import numpy as np
from typing import List, Union

__base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
geohash_parameters = {
    'precision': 16
}


def to_bytes_base32(integer):

    return f'{integer:05b}' # 5 bits 


def to_binary_base32(encoded_string: str) -> np.array:

    binary_representation = ''.join(to_bytes_base32(
        __base32.index(char)) for char in encoded_string)

    
    latitude = binary_representation[0::2]
    longitude = binary_representation[1::2]
    latitude_binary_representation = [int(bit) for bit in latitude]
    longitude_binary_representation = [int(bit) for bit in longitude]

    return np.array([latitude_binary_representation, longitude_binary_representation])


def encode_location(latitude: float, longitude: float) ->  np.array:

    location_encoded = pgh.encode(latitude=latitude,
                                  longitude=longitude,
                                  precision=geohash_parameters['precision'])
    location_binary = to_binary_base32(location_encoded)

    return location_binary
