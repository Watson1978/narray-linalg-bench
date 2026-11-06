"""D. Kabsch / Procrustes and PCA, one-to-one with lib/linalg_bench/problems/procrustes_pca.rb."""

import math

import fixtures
import gates
from backend import scalar, svd_gesvd, xl, xp
from compat import orth, svdvals
from problems import register

TOP_K = 3
VARIANTS = ("svd", "eigh")


def kabsch(p, q):
    p_mean = p.mean(axis=0)
    q_mean = q.mean(axis=0)
    h = (p - p_mean).T @ (q - q_mean)
    u, _s, vt = xl.svd(h)
    v = vt.T
    d = -1.0 if scalar(xl.det(v @ u.T)) < 0 else 1.0
    r = (v * xp.asarray([1.0, 1.0, d], dtype=p.dtype)) @ u.T
    return r, q_mean - r @ p_mean, d


def rmsd(p, q, r, t):
    diff = p @ r.T + t - q
    return math.sqrt(scalar((diff * diff).sum()) / p.shape[0])


def centered(x):
    return x - x.mean(axis=0)


def projector(basis):
    return basis @ basis.T


def rel(actual, expected):
    return abs((scalar(actual) - expected) / expected)


def align(p, q, r0):
    r, t, sign = kabsch(p, q)
    slog_sign, _ = xl.slogdet(r)
    return {
        "rmsd": rmsd(p, q, r, t),
        "rotation_error": gates.max_abs(r - r0),
        "det_error": abs(scalar(xl.det(r)) - 1),
        "slogdet_sign": scalar(slog_sign),
        "sign": sign,
    }


def pca(x):
    k = TOP_K
    xc = centered(x)
    n, d = xc.shape
    cov = xc.T @ xc / (n - 1)
    w, vecs = xl.eigh(cov)
    top_eigh = vecs[:, d - k:]
    _u, s, vt = xl.svd(xc, full_matrices=False)
    top_svd = vt[:k].T
    variances = s * s / (n - 1)
    sv_host = gates.descending(svdvals(xc))
    return {
        "subspace": gates.max_abs(projector(top_eigh) - projector(top_svd)),
        "variances": gates.max_rel_diff_sorted(w, variances),
        "eigvalsh": gates.max_rel_diff_sorted(xl.eigvalsh(cov), w),
        "orthogonality": gates.orthogonality(vecs),
        "norm_fro": rel(xl.norm(xc, "fro"), math.sqrt(sum(v * v for v in sv_host))),
        "norm_2": rel(xl.norm(xc, 2), sv_host[0]),
        "norm_nuc": rel(xl.norm(xc, "nuc"), sum(sv_host)),
        "rank": int(scalar(xl.matrix_rank(xc))),
        "top_variance": scalar(w[-1]),
        "nuclear_norm": sum(sv_host),
    }


def low_rank(x):
    rank = int(scalar(xl.matrix_rank(x)))
    cols = orth(x)
    d = x.shape[1]
    w, vecs = xl.eigh(x.T @ x)
    top = (x @ vecs[:, d - rank:]) / xp.sqrt(w[d - rank:])
    return {
        "rank": rank,
        "orth_columns": cols.shape[1],
        "column_space": gates.max_abs(projector(cols) - projector(top)),
        "orthogonality": gates.orthogonality(cols),
    }


class ProcrustesPca:
    name = "procrustes_pca"
    TOL = {"fp64": 5e-14}

    def fixture(self, size):
        return fixtures.read(self.name, size)

    def run(self, fixture):
        p = fixture.device("P")
        r0 = fixture.device("R0")
        result = {
            "clean": align(p, fixture.device("Q_clean"), r0),
            "noisy": align(p, fixture.device("Q_noisy"), r0),
            "reflected": align(p, fixture.device("Q_reflected"), r0),
            "pca": pca(fixture.device("X")),
            "low": low_rank(fixture.device("X_low")),
            "noise": fixture.attr("noise"), "rank": fixture.attr("rank"), "d": fixture["X"].shape[1],
        }
        result["values"] = {"rmsd_noisy": result["noisy"]["rmsd"], "rmsd_reflected": result["reflected"]["rmsd"],
                            "top_variance": result["pca"]["top_variance"],
                            "nuclear_norm": result["pca"]["nuclear_norm"]}
        return result

    def gate(self, result):
        tol = self.TOL.get(gates.dtype_name(), gates.tol())
        clean = result["clean"]
        noisy = result["noisy"]
        refl = result["reflected"]
        p = result["pca"]
        low = result["low"]
        expected_rmsd = result["noise"] * math.sqrt(3)
        return [
            gates.le("clean_rmsd", clean["rmsd"], tol),
            gates.le("clean_rotation_error", clean["rotation_error"], tol),
            gates.le("clean_det_is_one", clean["det_error"], tol),
            gates.eq("clean_sign", clean["sign"], 1.0),
            gates.le("noisy_rmsd_vs_noise", abs(noisy["rmsd"] / expected_rmsd - 1), 0.1),
            gates.le("noisy_rotation_error", noisy["rotation_error"], 1e-3),
            gates.eq("reflected_sign", refl["sign"], -1.0),
            gates.le("reflected_det_is_one", refl["det_error"], tol),
            gates.le("reflected_slogdet_sign", abs(refl["slogdet_sign"] - 1), 0),
            gates.ge("reflected_rmsd_is_large", refl["rmsd"] / expected_rmsd, 10.0),
            gates.le("pca_subspace_eigh_vs_svd", p["subspace"], tol),
            gates.le("pca_variances_eigh_vs_svd", p["variances"], tol),
            gates.le("pca_eigvalsh_vs_eigh", p["eigvalsh"], tol),
            gates.le("pca_eigenvectors_orthogonality", p["orthogonality"], tol),
            gates.le("norm_fro_vs_svdvals", p["norm_fro"], tol),
            gates.le("norm_2_vs_svdvals", p["norm_2"], tol),
            gates.le("norm_nuc_vs_svdvals", p["norm_nuc"], tol),
            gates.eq("matrix_rank_full", p["rank"], result["d"]),
            gates.eq("matrix_rank_low", low["rank"], result["rank"]),
            gates.eq("orth_columns_low", low["orth_columns"], result["rank"]),
            gates.le("orth_column_space_vs_eigh", low["column_space"], tol),
            gates.le("orth_orthogonality", low["orthogonality"], tol),
        ]

    def bench_step(self, fixture, variant=None):
        x = fixture.device("X")
        variant = variant or VARIANTS[0]
        if variant == "svd":
            return lambda: svd_gesvd(x)
        if variant == "eigh":
            return lambda: xl.eigh(x.T @ x)
        raise ValueError(f"unknown variant {variant!r}; use {' or '.join(VARIANTS)}")


register(ProcrustesPca.name, ProcrustesPca())
