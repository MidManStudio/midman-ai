#!/usr/bin/env python3
"""Writes the reference fixtures that tests/reference_files.rs reads.

Each fixture is a file produced by the reference `safetensors` package, stored as a
hex dump in a `.hex` file because the repository does not allow weight files. Run this
to regenerate them:

    pip install safetensors numpy
    python3 generate_fixtures.py

The values must match the ones in tests/reference_files.rs. Standard library plus
numpy and safetensors; the committed output was made with safetensors 0.8.0 and
numpy 2.4.4.
"""
import os
import tempfile

import numpy as np
from safetensors.numpy import save_file

HERE = os.path.dirname(os.path.abspath(__file__))


def f32(values, shape=None):
    array = np.array(values, dtype=np.float32)
    return array.reshape(shape) if shape is not None else array


def write_hex(name, tensors, metadata=None):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, name + ".safetensors")
        save_file(tensors, path, metadata=metadata)
        data = open(path, "rb").read()
    text = data.hex()
    lines = [text[i : i + 64] for i in range(0, len(text), 64)]
    with open(os.path.join(HERE, name + ".hex"), "w") as out:
        out.write("\n".join(lines) + "\n")
    print(f"{name}.hex: {len(data)} bytes")


# Values every fixture shares: exactly representable or rounded identically in Rust.
write_hex(
    "basic",
    {
        "w": f32([1.5, -2.0, 0.0, 3.25, 1e-30, 1e30], (2, 3)),
        "b": f32([0.1, 0.2, 0.3]),
        "scalar": f32(7.0),
        "empty": np.zeros((0,), dtype=np.float32),
        "special": f32([np.nan, np.inf, -np.inf, -0.0, np.finfo(np.float32).tiny, 1e-45]),
    },
    metadata={"format": "pt"},
)

write_hex(
    "unicode_names",
    {
        "layer.0.\u00e9\u65e5\u672c": f32([1.0, 2.0]),
        'quote"back\\slash\ttab': f32([3.0]),
    },
    metadata={"note": "caf\u00e9 \"quoted\" \u65e5\u672c\nline"},
)

write_hex("empty", {})

write_hex("multi_metadata", {"x": f32([1.0])}, metadata={"alpha": "1", "beta": "2", "gamma": "3"})

# A file with another dtype next to an F32 tensor: our reader must reject it by name.
write_hex(
    "mixed_dtypes",
    {"a": f32([1.0, 2.0]), "h": np.array([1.0, 2.0], dtype=np.float16)},
)
