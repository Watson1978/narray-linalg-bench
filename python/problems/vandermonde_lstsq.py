"""E. Vandermonde least squares, one-to-one with lib/linalg_bench/problems/vandermonde_lstsq.rb.

numo-linalg-alt's qr modes map to numpy's as reduce -> complete,
economic -> reduced. lstsq has one driver here, so the driver gate is ×.
"""

import fixtures
import gates
from backend import scalar, xl, xp
from compat import cutoff as rcond
from compat import null_space, svdvals
from problems import register

VARIANTS = ("lstsq", "qr")


def raw_r(a):
    h, _tau = xl.qr(a, mode="raw")
    n = a.shape[1]
    return xp.triu(h.T[:n])


def solutions(v, y, coef):
    cut = rcond(v)
    x = xl.lstsq(v, y, rcond=cut)[0]
    q, r = xl.qr(v, mode="reduced")
    return {
        "qr": gates.rel_diff(xl.solve(r, q.T @ y), x),
        "pinv": gates.rel_diff(xl.pinv(v, rcond=cut) @ y, x),
        "normal": gates.rel_diff(xl.solve(v.T @ v, v.T @ y), x),
        "coef_error": gates.rel_diff(x, coef),
        "residual_norm": gates.fro(v @ x - y),
    }


def qr(v):
    q, r = xl.qr(v, mode="complete")
    qe, re = xl.qr(v, mode="reduced")
    n = v.shape[1]
    return {
        "q_orthogonality": gates.orthogonality(q),
        "qe_orthogonality": gates.orthogonality(qe),
        "rebuild": gates.rel_diff(q @ r, v),
        "rebuild_economic": gates.rel_diff(qe @ re, v),
        "mode_r": gates.rel_diff(xl.qr(v, mode="r"), r[:n]),
        "mode_raw": gates.rel_diff(raw_r(v), re),
    }


def spectrum(v):
    n = v.shape[1]
    u, s, vt = xl.svd(v, full_matrices=False)
    conds = [scalar(xl.cond(xp.ascontiguousarray(v[:, :k]))) for k in range(2, n + 1)]
    cond = scalar(xl.cond(v))
    s0 = scalar(s[0])
    fro = gates.fro(v)
    return {
        "cond": cond,
        "cond_growth_min": min(b / a for a, b in zip(conds, conds[1:])),
        "cond_vs_svd": abs((cond - s0 / scalar(s[-1])) / cond),
        "svdvals_vs_svd": gates.max_rel_diff_sorted(svdvals(v), s),
        "svd_rebuild": gates.rel_diff((u * s) @ vt, v),
        "norm_fro": abs((scalar(xl.norm(v, "fro")) - fro) / fro),
        "norm_2": abs((scalar(xl.norm(v, 2)) - s0) / s0),
    }


def deficient(vd, y):
    cut = rcond(vd)
    x, _resids, rank, _s = xl.lstsq(vd, y, rcond=cut)
    z = null_space(vd, cut)
    shifted = x + z[:, 0]
    residual = gates.fro(vd @ x - y)
    return {
        "matrix_rank": int(scalar(xl.matrix_rank(vd))),
        "lstsq_rank": int(scalar(rank)),
        "null_columns": z.shape[1],
        "null_residual": gates.max_abs(vd @ z) / gates.fro(vd),
        "null_orthogonality": gates.orthogonality(z),
        "pinv": gates.rel_diff(xl.pinv(vd, rcond=cut) @ y, x),
        "min_norm": gates.fro(x),
        "norm_ratio": gates.fro(x) / gates.fro(shifted),
        "shifted_residual": abs(gates.fro(vd @ shifted - y) - residual) / residual,
    }


class VandermondeLstsq:
    name = "vandermonde_lstsq"
    TOL = {"fp64": 2e-14}

    def fixture(self, size):
        return fixtures.read(self.name, size)

    def run(self, fixture):
        v = fixture.device("V")
        y = fixture.device("y")
        result = {"solutions": solutions(v, y, fixture.device("coef")), "qr": qr(v), "spectrum": spectrum(v),
                  "deficient": deficient(fixture.device("V_dup"), y), "rank_dup": fixture.attr("rank_dup"),
                  "n_dup": len(fixture.attr("duplicated"))}
        result["values"] = {
            "residual_norm": result["solutions"]["residual_norm"],
            "cond": result["spectrum"]["cond"],
            "coef_error": result["solutions"]["coef_error"],
            "min_norm": result["deficient"]["min_norm"],
        }
        return result

    def gate(self, result):
        tol = self.TOL.get(gates.dtype_name(), gates.tol())
        sol = result["solutions"]
        q = result["qr"]
        sp = result["spectrum"]
        dd = result["deficient"]
        cond = sp["cond"]
        return [
            gates.le("qr_vs_lstsq", sol["qr"] / cond, tol),
            gates.le("pinv_vs_lstsq", sol["pinv"] / cond, tol),
            gates.le("normal_equations_vs_lstsq", sol["normal"] / cond**2, tol),
            gates.le("coef_error", sol["coef_error"], 1e-2),
            gates.le("q_complete_orthogonality", q["q_orthogonality"], tol),
            gates.le("q_economic_orthogonality", q["qe_orthogonality"], tol),
            gates.le("q_r_rebuilds_v", q["rebuild"], tol),
            gates.le("q_r_economic_rebuilds_v", q["rebuild_economic"], tol),
            gates.le("mode_r_vs_reduce", q["mode_r"], tol),
            gates.le("mode_raw_vs_economic", q["mode_raw"], tol),
            gates.ge("cond_grows_with_degree", sp["cond_growth_min"], 1.0),
            gates.le("cond_vs_svd_ratio", sp["cond_vs_svd"], tol),
            gates.le("svdvals_vs_svd", sp["svdvals_vs_svd"], tol),
            gates.le("u_s_vt_rebuilds_v", sp["svd_rebuild"], tol),
            gates.le("norm_fro", sp["norm_fro"], tol),
            gates.le("norm_2_vs_largest_singular_value", sp["norm_2"], tol),
            gates.eq("matrix_rank_dup", dd["matrix_rank"], result["rank_dup"]),
            gates.eq("lstsq_rank_dup", dd["lstsq_rank"], result["rank_dup"]),
            gates.eq("null_space_columns", dd["null_columns"], result["n_dup"]),
            gates.le("null_space_annihilates", dd["null_residual"], tol),
            gates.le("null_space_orthogonality", dd["null_orthogonality"], tol),
            gates.le("min_norm_lstsq_vs_pinv", dd["pinv"], tol),
            gates.le("min_norm_beats_shifted", dd["norm_ratio"], 1.0 - 1e-6),
            gates.le("shifted_keeps_residual", dd["shifted_residual"], tol),
        ]

    def bench_step(self, fixture, variant=None):
        v = fixture.device("V")
        y = fixture.device("y")
        variant = variant or VARIANTS[0]
        if variant == "lstsq":
            cut = rcond(v)
            return lambda: xl.lstsq(v, y, rcond=cut)
        if variant == "qr":
            def step():
                q, r = xl.qr(v, mode="reduced")
                return xl.solve(r, q.T @ y)
            return step
        raise ValueError(f"unknown variant {variant!r}; use {' or '.join(VARIANTS)}")


register(VandermondeLstsq.name, VandermondeLstsq())
