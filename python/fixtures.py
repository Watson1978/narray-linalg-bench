"""Reads the fixtures script/make_fixtures.rb writes.

  python/.venv/bin/python python/fixtures.py digest gaussian_process small
"""

import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("LINALG_BENCH_DATA") or os.path.join(ROOT, "data")


class Fixture:
    def __init__(self, problem, size, arrays, attrs, manifest):
        self.problem = problem
        self.size = size
        self.arrays = arrays
        self.attrs = attrs
        self.manifest = manifest

    def __getitem__(self, name):
        return self.arrays[name]

    def attr(self, name):
        return self.attrs[name]

    def device(self, name, dtype=None):
        from backend import device
        return device(self.arrays[name], dtype)


def read(problem, size, data_dir=DATA_DIR):
    base = os.path.join(data_dir, f"{problem}_{size}")
    with open(base + ".json") as f:
        manifest = json.load(f)
    if manifest["dtype"] != "<f8" or manifest["order"] != "C":
        raise ValueError(f"{base}.json: unexpected layout {manifest['dtype']} {manifest['order']}")
    arrays = {}
    for entry in manifest["arrays"]:
        count = int(np.prod(entry["shape"], dtype=np.int64))
        flat = np.fromfile(base + ".bin", dtype="<f8", count=count, offset=entry["offset"])
        if flat.size != count:
            raise ValueError(f"{base}.bin is truncated at {entry['name']}")
        arrays[entry["name"]] = flat.reshape(entry["shape"])
    return Fixture(problem, size, arrays, manifest["attrs"], manifest)


def digest(problem, size):
    fixture = read(problem, size)
    return {name: {"shape": list(array.shape),
                   "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}
            for name, array in fixture.arrays.items()}


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "digest":
        sys.exit("usage: fixtures.py digest <problem> <size>")
    json.dump(digest(sys.argv[2], sys.argv[3]), sys.stdout)
    print()
