"""C. Kalman filter / discrete Riccati, one-to-one with lib/linalg_bench/problems/kalman_riccati.rb."""

import fixtures
import gates
from backend import scalar, to_host, xl, xp, xsl
from problems import register

ITERATIONS = 500
EXPM_TOL = 1e-10
SQUARINGS = 4
BURN_IN = 100
VARIANTS = ("riccati", "expm")


def riccati_step(a, c, q, r, p, route):
    apct = a @ p @ c.T
    s = c @ p @ c.T + r
    correction = apct @ xl.inv(s) @ apct.T if route == "inv" else apct @ xl.solve(s, apct.T)
    nxt = a @ p @ a.T - correction + q
    return (nxt + nxt.T) / 2


def riccati(a, c, q, r, route):
    p = q
    for _ in range(ITERATIONS):
        p = riccati_step(a, c, q, r, p, route)
    return p


def rel(actual, expected):
    return abs((scalar(actual) - expected) / expected)


def discretization(a_c, dt, a):
    small = xsl.expm(a_c * (dt / 2**SQUARINGS))
    return {"squaring": gates.rel_diff(xl.matrix_power(small, 2**SQUARINGS), a)}


def expm_wide(e):
    w, v = xl.eigh(e)
    return gates.rel_diff(xsl.expm(e), (v * xp.exp(w)) @ v.T)


def stability(a, closed):
    w, _ = xl.eig(a)
    return {
        "open_loop_radius": gates.descending(xp.abs(xl.eigvals(a)))[0],
        "eig_vs_eigvals": gates.max_rel_diff_sorted(xp.abs(w), xp.abs(xl.eigvals(a))),
        "closed_loop_radius": gates.descending(xp.abs(xl.eigvals(closed)))[0],
    }


def conditioning(s):
    cond = scalar(xl.cond(s))
    product = scalar(xl.norm(s, 2)) * scalar(xl.norm(xl.inv(s), 2))
    return {"cond": cond, "cond_vs_norms": abs((cond - product) / product)}


def run_filter(a, c, gain, xs, ys, p):
    steps = ys.shape[0]
    n = a.shape[0]
    x_hat = xp.zeros(n, dtype=xs.dtype)
    errors = xp.zeros((steps, n), dtype=xs.dtype)
    for k in range(steps):
        errors[k] = xs[k] - x_hat
        x_hat = a @ x_hat + gain @ (ys[k] - c @ x_hat)
    tail = errors[BURN_IN:]
    variances = to_host((tail * tail).mean(axis=0))
    expected = to_host(xp.diagonal(p))
    return {"variance_ratio_error": float(abs(variances / expected - 1).max())}


class KalmanRiccati:
    name = "kalman_riccati"
    TOL = {"fp64": 5e-14}

    def fixture(self, size):
        return fixtures.read(self.name, size)

    def run(self, fixture):
        a_c = fixture.device("A_c")
        c = fixture.device("C")
        q = fixture.device("Q")
        r = fixture.device("R")
        dt = fixture.attr("dt")
        a = xsl.expm(a_c * dt)
        p = riccati(a, c, q, r, "solve")
        p_inv = riccati(a, c, q, r, "inv")
        s = c @ p @ c.T + r
        gain = a @ p @ c.T @ xl.inv(s)
        result = {
            "discretization": {**discretization(a_c, dt, a), "wide": expm_wide(fixture.device("E_wide"))},
            "riccati": {"residual": gates.rel_diff(riccati_step(a, c, q, r, p, "solve"), p),
                        "solve_vs_inv": gates.rel_diff(p_inv, p),
                        "norm_fro": rel(xl.norm(p, "fro"), gates.fro(p))},
            "stability": stability(a, a - gain @ c),
            "conditioning": conditioning(s),
            "filter": run_filter(a, c, gain, fixture.device("xs"), fixture.device("ys"), p),
        }
        result["values"] = {"trace_p": scalar(xp.trace(p)),
                            "closed_loop_radius": result["stability"]["closed_loop_radius"],
                            "cond_s": result["conditioning"]["cond"]}
        return result

    def gate(self, result):
        tol = self.TOL.get(gates.dtype_name(), gates.tol())
        disc = result["discretization"]
        ric = result["riccati"]
        st = result["stability"]
        cn = result["conditioning"]
        fl = result["filter"]
        return [
            gates.le("expm_vs_matrix_power_of_expm", disc["squaring"], tol),
            gates.le("expm_vs_eigh_wide_symmetric", disc["wide"], EXPM_TOL),
            gates.le("open_loop_radius_below_one", st["open_loop_radius"], 1 - 1e-6),
            gates.le("eig_vs_eigvals", st["eig_vs_eigvals"], tol),
            gates.le("riccati_fixed_point", ric["residual"], tol),
            gates.le("riccati_solve_vs_inv", ric["solve_vs_inv"], tol),
            gates.le("norm_fro", ric["norm_fro"], tol),
            gates.le("closed_loop_radius_below_one", st["closed_loop_radius"], 1 - 1e-6),
            gates.le("cond_vs_norm_product", cn["cond_vs_norms"], tol),
            gates.le("filter_error_variance_vs_p", fl["variance_ratio_error"], 0.2),
        ]

    def bench_step(self, fixture, variant=None):
        variant = variant or VARIANTS[0]
        if variant == "riccati":
            a = xsl.expm(fixture.device("A_c") * fixture.attr("dt"))
            c = fixture.device("C")
            q = fixture.device("Q")
            r = fixture.device("R")
            state = {"p": q}

            def step():
                state["p"] = riccati_step(a, c, q, r, state["p"], "solve")
                return state["p"]
            return step
        if variant == "expm":
            a_c = fixture.device("A_c")
            return lambda: xsl.expm(a_c)
        raise ValueError(f"unknown variant {variant!r}; use {' or '.join(VARIANTS)}")


register(KalmanRiccati.name, KalmanRiccati())
