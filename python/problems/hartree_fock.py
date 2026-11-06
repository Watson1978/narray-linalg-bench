"""A. Restricted Hartree-Fock, one-to-one with lib/linalg_bench/problems/hartree_fock.rb.

CuPy has no generalized eigh, so compat.eigh_general reduces it through the
Cholesky factor (cholesky + solve_triangular x 3 + eigh) where Cumo calls
sygvd once (docs/cupy-gaps.md). There is no generalized eigvalsh to compare
with either, so that gate is ×.
"""

import compat
import fixtures
import gates
from backend import scalar, xl, xp, xsl
from problems import register

ENERGY_TOL = 1e-6
ITERATIONS = 50
MOLECULES = ("h2", "heh")
VARIANTS = ("eigh_general", "cholesky_solve")


def two_electron(eri, p):
    n = p.shape[0]
    j = (eri.reshape(n * n, n * n) @ p.reshape(-1)).reshape(n, n)
    k = (xp.ascontiguousarray(eri.transpose(0, 2, 1, 3)).reshape(n * n, n * n) @ p.reshape(-1)).reshape(n, n)
    return j - 0.5 * k


def density(c, occupied):
    occ = c[:, :occupied]
    return 2.0 * occ @ occ.T


def electronic_energy(h, f, p):
    return 0.5 * scalar((p * (h + f)).sum())


def scf(h, s, eri, occupied):
    p = xp.zeros_like(h)
    energies = []
    for _ in range(ITERATIONS):
        f = h + two_electron(eri, p)
        _eps, c = compat.eigh_general(f, s)
        energies.append(electronic_energy(h, f, p))
        p = density(c, occupied)
    return {"f": h + two_electron(eri, p), "p": p, "energies": energies}


def molecule(fixture, name):
    h = fixture.device(f"{name}_H")
    s = fixture.device(f"{name}_S")
    eri = fixture.device(f"{name}_eri")
    occupied = fixture.attr(f"{name}_n_electrons") // 2
    out = scf(h, s, eri, occupied)
    f = out["f"]
    energy = electronic_energy(h, f, out["p"]) + fixture.attr(f"{name}_e_nuc")
    eps, c = compat.eigh_general(f, s)
    w, u = xl.eigh(s)
    x = (u / xp.sqrt(w)) @ u.T
    l = xl.cholesky(s)
    x_chol = xl.inv(l).T
    n = s.shape[0]
    energies = out["energies"]
    return {
        "energy": energy,
        "energy_error": abs(energy - fixture.attr(f"{name}_e_total")),
        "last_change": abs(energies[-1] - energies[-2]),
        "max_rise": max(b - a for a, b in zip(energies[1:], energies[2:])),
        "eps_symmetric": gates.max_rel_diff_sorted(xl.eigh(x @ f @ x)[0], eps),
        "eps_cholesky": gates.max_rel_diff_sorted(xl.eigh(x_chol.T @ f @ x_chol)[0], eps),
        "s_orthonormal": gates.max_abs(c.T @ s @ c - xp.eye(n, dtype=s.dtype)),
        "cho_solve": gates.rel_diff(compat.cho_solve(l, f), xl.solve(s, f)),
        "inv": gates.rel_diff(x_chol.T @ f, xl.solve(l, f)),
    }


class HartreeFock:
    name = "hartree_fock"
    TOL = {"fp64": 1e-14}

    def fixture(self, size):
        return fixtures.read(self.name, size)

    def run(self, fixture):
        molecules = {name: molecule(fixture, name) for name in MOLECULES}
        return {"molecules": molecules,
                "values": {f"{name}_energy": m["energy"] for name, m in molecules.items()}}

    def gate(self, result):
        tol = self.TOL.get(gates.dtype_name(), gates.tol())
        found = []
        for name, m in result["molecules"].items():
            found += [
                gates.le(f"{name}_energy_vs_pyscf", m["energy_error"], ENERGY_TOL),
                gates.le(f"{name}_scf_converged", m["last_change"], 1e-10),
                gates.le(f"{name}_scf_energy_never_rises", m["max_rise"], 1e-12),
                gates.le(f"{name}_eps_general_vs_symmetric", m["eps_symmetric"], tol),
                gates.le(f"{name}_eps_general_vs_cholesky", m["eps_cholesky"], tol),
                gates.le(f"{name}_c_s_orthonormal", m["s_orthonormal"], tol),
                gates.le(f"{name}_cho_solve_vs_solve", m["cho_solve"], tol),
                gates.le(f"{name}_inv_x_vs_solve", m["inv"], tol),
            ]
        found.append(gates.le("h2_energy_vs_szabo_ostlund", abs(result["molecules"]["h2"]["energy"] + 1.1167), 5e-5))
        return found

    def bench_step(self, fixture, variant=None):
        h = fixture.device("H")
        s = fixture.device("S")
        variant = variant or VARIANTS[0]
        if variant == "eigh_general":
            return lambda: compat.eigh_general(h, s)
        if variant == "cholesky_solve":
            def step():
                l = xl.cholesky(s)
                return xl.solve(l, xl.solve(l, h).T)
            return step
        raise ValueError(f"unknown variant {variant!r}; use {' or '.join(VARIANTS)}")


register(HartreeFock.name, HartreeFock())
