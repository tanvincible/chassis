"""The embedding model for this task: embed(texts) -> a (len(texts), 256) float32 array.

A stand-in that needs no download: words of 4 letters or more, hashed into 256 buckets, so texts
that share words are near. Compare its vectors by cosine distance.
"""

import hashlib
import re

import numpy as np

DIMENSIONS = 256


def embed(texts):
    vectors = np.zeros((len(texts), DIMENSIONS), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in re.findall(r"[a-z]{4,}", text.lower()):
            bucket = hashlib.blake2b(word.encode(), digest_size=4).digest()
            vectors[row, int.from_bytes(bucket, "little") % DIMENSIONS] += 1.0
    return vectors
