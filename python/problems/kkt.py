"""G. KKT system of equality-constrained least squares, one-to-one with lib/linalg_bench/problems/kkt.rb.

scipy.linalg.ldl defaults to lower=True where numo-linalg-alt defaults to
uplo: "U", so it gets lower=False. CuPy has no ldl, so the ldl gates exist
only where xsl has it (scipy on numpy) and are × on CuPy. solve has one
driver here, so the sym/gen gate is ×.
"""

import compat
import fixtures
import gates
from backend import to_host, xl, xp, xsl
from problems import register

VARIANTS = ("solve", "eigvalsh", "ldl")


def system(a, b, bc, c):
    n = a.shape[1]
    p = bc.shape[0]
    k = xp.zeros((n + p, n + p), dtype=a.dtype)
    k[:n, :n] = a.T @ a
    k[:n, n:] = bc.T
    k[n:, :n] = bc
    rhs = xp.concatenate([a.T @ b, c])
    return k, rhs


def inertia(values, scale):
    host = to_host(values).tolist()
    cut = scale * len(host) * float(xp.finfo(values.dtype).eps)
    return [sum(v > cut for v in host), sum(v < -cut for v in host), sum(abs(v) <= cut for v in host)]


def eliminate(a, b, bc, c):
    x0 = xl.lstsq(bc, c, rcond=None)[0]
    z = compat.null_space(bc)
    y = xl.lstsq(a @ z, b - a @ x0, rcond=None)[0]
    return x0 + z @ y


def ldl(k, scale):
    u, d, perm = xsl.ldl(k, lower=False)
    return {
        "rebuild": gates.rel_diff(u @ d @ u.T, k),
        "off_triangle": gates.max_abs(xp.tril(u[perm], -1)),
        "inertia": inertia(xl.eigvalsh(d), scale),
    }


class Kkt:
    name = "kkt"
    TOL = {"fp64": 2e-14}

    def fixture(self, size):
        return fixtures.read(self.name, size)

    def run(self, fixture):
        a = fixture.device("A")
        b = fixture.device("b")
        bc = fixture.device("B")
        c = fixture.device("c")
        n = a.shape[1]
        k, rhs = system(a, b, bc, c)
        sol = xl.solve(k, rhs)
        x = sol[:n]
        lam = sol[n:]
        scale = gates.max_abs(k)
        result = {
            "n": n, "p": bc.shape[0],
            "constraint": gates.fro(bc @ x - c) / gates.fro(c),
            "lagrange": gates.fro(a.T @ (a @ x - b) + bc.T @ lam) / gates.fro(a.T @ b),
            "elimination": gates.rel_diff(eliminate(a, b, bc, c), x),
            "inertia": inertia(xl.eigvalsh(k), scale),
            "inv_vs_solve": gates.rel_diff(xl.inv(k), xl.solve(k, xp.eye(k.shape[0], dtype=k.dtype))),
            "ldl": ldl(k, scale) if hasattr(xsl, "ldl") else None,
        }
        result["values"] = {"x_norm": gates.fro(x), "lambda_norm": gates.fro(lam),
                            "objective": gates.fro(a @ x - b)}
        return result

    def gate(self, result):
        tol = self.TOL.get(gates.dtype_name(), gates.tol())
        expected = [result["n"], result["p"], 0]
        found = [
            gates.le("constraint_residual", result["constraint"], tol),
            gates.le("lagrange_residual", result["lagrange"], tol),
            gates.le("kkt_vs_null_space_elimination", result["elimination"], tol),
            gates.eq("inertia_from_eigvalsh", result["inertia"], expected),
            gates.le("inv_vs_solve_identity", result["inv_vs_solve"], tol),
        ]
        if result["ldl"] is not None:
            found += [
                gates.le("ldl_rebuilds_k", result["ldl"]["rebuild"], tol),
                gates.le("ldl_factor_is_triangular_after_perm", result["ldl"]["off_triangle"], 0),
                gates.eq("inertia_from_ldl_d", result["ldl"]["inertia"], expected),
            ]
        return found

    def bench_step(self, fixture, variant=None):
        k, rhs = system(fixture.device("A"), fixture.device("b"), fixture.device("B"), fixture.device("c"))
        variant = variant or VARIANTS[0]
        if variant == "solve":
            return lambda: xl.solve(k, rhs)
        if variant == "eigvalsh":
            return lambda: xl.eigvalsh(k)
        if variant == "ldl":
            if not hasattr(xsl, "ldl"):
                raise NotImplementedError("CuPy has no ldl (docs/cupy-gaps.md)")
            return lambda: xsl.ldl(k, lower=False)
        raise ValueError(f"unknown variant {variant!r}; use {', '.join(VARIANTS)}")


register(Kkt.name, Kkt())
