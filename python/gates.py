"""Gate measures, the Python counterpart of lib/linalg_bench/gates.rb.

Built from elementwise operations and matmul only, never from the linalg
function under test.
"""

import math

from backend import dtype_name, scalar, to_host, xp

DEFAULT_TOL = {"fp64": 1e-10, "fp32": 1e-4}


class Gate:
    def __init__(self, name, value, tol, op):
        self.name = name
        self.value = value
        self.tol = tol
        self.op = op

    def passed(self):
        if self.op == "le":
            return self.value <= self.tol
        if self.op == "ge":
            return self.value >= self.tol
        if self.op == "eq":
            return self.value == self.tol
        raise ValueError(f"unknown gate op {self.op!r}")

    def to_dict(self):
        return {"value": self.value, "tol": self.tol, "op": self.op, "pass": self.passed()}


def tol(dtype=None):
    return DEFAULT_TOL[dtype or dtype_name()]


def le(name, value, limit):
    return Gate(name, float(value), float(limit), "le")


def ge(name, value, limit):
    return Gate(name, float(value), float(limit), "ge")


def eq(name, value, expected):
    return Gate(name, value, expected, "eq")


def fro(a):
    return math.sqrt(scalar((a * a).sum()))


def max_abs(a):
    return scalar(xp.abs(a).max())


def solve_residual(a, x, b):
    return fro(a @ x - b) / (fro(a) * fro(x) + fro(b))


def rel_diff(actual, expected):
    return fro(actual - expected) / fro(expected)


def orthogonality(q):
    return max_abs(q.T @ q - xp.eye(q.shape[1], dtype=q.dtype))


def descending(values):
    return sorted(to_host(values).tolist(), reverse=True)


def max_rel_diff_sorted(actual, expected):
    a = descending(actual)
    e = descending(expected)
    if len(a) != len(e):
        raise ValueError(f"length differs: {len(a)} vs {len(e)}")
    scale = max(abs(v) for v in e)
    return max(abs(x - y) for x, y in zip(a, e)) / scale


def report(gates):
    return {gate.name: gate.to_dict() for gate in gates}
