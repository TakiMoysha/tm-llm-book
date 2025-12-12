from dataclasses import dataclass, field
from functools import cached_property

from opensearchpy import AsyncOpenSearch, helpers


@dataclass(frozen=True)
class OpenSearchConfig:
    host: str = field(default="localhost")
    port: int = field(default=9200)

    @cached_property
    def client(self):
        return AsyncOpenSearch(
            hosts=[{"host": self.host, "port": self.port}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_show_warn=False,
        )


import numpy as np


def quantize_to_binary(embedding: np.ndarray):
    assert len(embedding) == 768

    binary = (embedding > 0).astype(np.uint8)
    grouped = binary.reshape(96, 8)
    packed = np.packbits(grouped, axis=1).flatten()

    unpacked_grouped = np.unpackbits(packed.reshape(-1, 1), axis=1)
    recovered_binary = unpacked_grouped.flatten()

    binary_as_float = np.where(recovered_binary == 1, 1.0, -1.0).astype(np.float32)

    def cosine_similarity(a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0
        return np.dot(a, b) / (norm_a * norm_b)

    similarity = cosine_similarity(embedding, binary_as_float)

    l2_error = np.linalg.norm(embedding - binary_as_float)
    mae = np.mean(np.abs(embedding - binary_as_float))

    return {
        "cosine_similarity": similarity,
        "l2_error": l2_error,
        "mae": mae,
        "packed_bytes": packed,
        "original_shape": embedding.shape,
        "compressed_size_bytes": len(packed),
        "compression_ratio": len(embedding) / len(packed),
    }

