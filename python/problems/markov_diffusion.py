"""F. Markov chain and implicit diffusion, one-to-one with lib/linalg_bench/problems/markov_diffusion.rb.

null_space comes from svd and lu_inv from lu_solve on the identity, since
CuPy has neither (docs/cupy-gaps.md). lu, lu_factor, lu_solve and expm come
from scipy.linalg / cupyx.scipy.linalg; scipy's piv is 0-based.
"""

import compat
import fixtures
import gates
from backend import scalar, xl, xp, xsl
from problems import register

POWERS = (4, 8)
TIMES = (0.5, 1.0)
STEPS = 100
VARIANTS = ("lu_solve", "expm", "matrix_power", "eig")
BENCH_POWER = 64


def stationary_from_eig(pt):
    w, vr = xl.eig(pt)
    index = int(scalar(xp.abs(w.real - 1).argmin()))
    pi = vr[:, index].real
    return pi / pi.sum(), w


def stationary(p, pi, pi_null, null_columns):
    return {
        "min": scalar(pi.min()),
        "sum_error": abs(scalar(pi.sum()) - 1),
        "residual": gates.max_abs(pi @ p - pi),
        "null_columns": null_columns,
        "eig_vs_null": gates.rel_diff(pi_null, pi),
    }


def powers(p, pi):
    errors = [gates.max_abs(xl.matrix_power(p, k) - pi) for k in POWERS]
    return {
        "vs_products": gates.rel_diff(xl.matrix_power(p, 3), p @ p @ p),
        "rate": (errors[1] / errors[0]) ** (1.0 / (POWERS[1] - POWERS[0])),
    }


def generator(q, pi):
    e1, e2 = (xsl.expm(q * t) for t in TIMES)
    ones = xp.ones(q.shape[0], dtype=q.dtype)
    return {
        "row_sum_error": gates.max_abs(e2 @ ones - 1),
        "min": scalar(e2.min()),
        "semigroup": gates.rel_diff(e1 @ e1, e2),
        "squaring": gates.rel_diff(xl.matrix_power(xsl.expm(q * (TIMES[1] / 8)), 8), e2),
        "pi_residual": gates.max_abs(pi @ e2 - pi),
    }


def diffusion(m, u0):
    factor = xsl.lu_factor(m)
    total = scalar(u0.sum())
    u = u0
    max_rise = 0.0
    previous_max = scalar(u.max())
    for _ in range(STEPS):
        u = xsl.lu_solve(factor, u)
        current = scalar(u.max())
        max_rise = max(max_rise, current - previous_max)
        previous_max = current
    pm, l, up = xsl.lu(m)
    return {
        "total_drift": abs(scalar(u.sum()) - total) / total,
        "max_rise": max_rise,
        "final_max": previous_max,
        "plu": gates.rel_diff(pm @ l @ up, m),
        "lu_inv": gates.rel_diff(compat.lu_inv(factor), xl.inv(m)),
        "lu_solve": gates.rel_diff(xsl.lu_solve(factor, u0), xl.solve(m, u0)),
    }


class MarkovDiffusion:
    name = "markov_diffusion"
    TOL = {"fp64": 5e-14}

    def fixture(self, size):
        return fixtures.read(self.name, size)

    def run(self, fixture):
        p = fixture.device("P")
        q = fixture.device("Q")
        n = p.shape[0]
        eye = xp.eye(n, dtype=p.dtype)
        pi, w = stationary_from_eig(p.T)
        z = compat.null_space(p.T - eye)
        pi_null = z[:, 0] / z[:, 0].sum()
        moduli = gates.descending(xp.abs(xl.eigvals(p)))
        result = {
            "stationary": stationary(p, pi, pi_null, z.shape[1]),
            "spectrum": {"largest": moduli[0], "second": moduli[1],
                         "eig_vs_eigvals": gates.max_rel_diff_sorted(xp.abs(xl.eigvals(p)), xp.abs(w))},
            "powers": powers(p, pi),
            "generator": generator(q, pi),
            "diffusion": diffusion(fixture.device("M"), fixture.device("u0")),
        }
        result["values"] = {"pi_max": scalar(pi.max()), "lambda2": moduli[1],
                            "diffusion_max": result["diffusion"]["final_max"]}
        return result

    def gate(self, result):
        tol = self.TOL.get(gates.dtype_name(), gates.tol())
        st = result["stationary"]
        sp = result["spectrum"]
        pw = result["powers"]
        gen = result["generator"]
        df = result["diffusion"]
        return [
            gates.ge("pi_nonnegative", st["min"], -tol),
            gates.le("pi_sums_to_one", st["sum_error"], tol),
            gates.le("pi_is_stationary", st["residual"], tol),
            gates.eq("null_space_columns", st["null_columns"], 1),
            gates.le("pi_eig_vs_null_space", st["eig_vs_null"], tol),
            gates.le("largest_modulus_is_one", abs(sp["largest"] - 1), tol),
            gates.le("second_modulus_below_one", sp["second"], 1 - 1e-3),
            gates.le("eig_vs_eigvals", sp["eig_vs_eigvals"], tol),
            gates.le("matrix_power_vs_products", pw["vs_products"], tol),
            gates.le("power_convergence_rate_vs_lambda2", abs(pw["rate"] / sp["second"] - 1), 0.5),
            gates.le("expm_rows_sum_to_one", gen["row_sum_error"], tol),
            gates.ge("expm_nonnegative", gen["min"], -tol),
            gates.le("expm_semigroup", gen["semigroup"], tol),
            gates.le("expm_vs_matrix_power_of_expm", gen["squaring"], tol),
            gates.le("expm_keeps_pi", gen["pi_residual"], tol),
            gates.le("diffusion_conserves_total", df["total_drift"], tol),
            gates.le("diffusion_max_never_rises", df["max_rise"], tol),
            gates.le("p_l_u_rebuilds_m", df["plu"], tol),
            gates.le("lu_inv_vs_inv", df["lu_inv"], tol),
            gates.le("lu_solve_vs_solve", df["lu_solve"], tol),
        ]

    def bench_step(self, fixture, variant=None):
        variant = variant or VARIANTS[0]
        if variant == "lu_solve":
            factor = xsl.lu_factor(fixture.device("M"))
            u = fixture.device("u0")
            return lambda: xsl.lu_solve(factor, u)
        if variant == "expm":
            q = fixture.device("Q")
            return lambda: xsl.expm(q)
        if variant == "matrix_power":
            p = fixture.device("P")
            return lambda: xl.matrix_power(p, BENCH_POWER)
        if variant == "eig":
            pt = xp.ascontiguousarray(fixture.device("P").T)
            return lambda: xl.eig(pt)
        raise ValueError(f"unknown variant {variant!r}; use {', '.join(VARIANTS)}")


register(MarkovDiffusion.name, MarkovDiffusion())
