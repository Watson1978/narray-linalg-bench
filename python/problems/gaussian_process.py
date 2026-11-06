"""B. Gaussian process regression, one-to-one with lib/linalg_bench/problems/gaussian_process.rb."""

import math

import fixtures
import gates
from backend import scalar, xl, xp
from compat import cho_inv, cho_solve
from problems import register

GRID_STEP = 1.5
GRID_HALF = 3
LOG_2PI = math.log(2 * math.pi)


def kernel(x1, x2, ell, sf):
    sq1 = (x1 * x1).sum(axis=1)
    sq2 = (x2 * x2).sum(axis=1)
    d2 = sq1[:, None] + sq2[None, :] - 2 * (x1 @ x2.T)
    return sf**2 * xp.exp(d2 * (-0.5 / ell**2))


def noisy_kernel(x, ell, sf, sn):
    return kernel(x, x, ell, sf) + sn**2 * xp.eye(x.shape[0], dtype=x.dtype)


def log_marginal_likelihood(x, y, ell, sf, sn):
    l = xl.cholesky(noisy_kernel(x, ell, sf, sn))
    alpha = cho_solve(l, y)
    return -0.5 * (y @ alpha) - xp.log(xp.diagonal(l)).sum() - 0.5 * y.shape[0] * LOG_2PI


def grid(center):
    return [center * GRID_STEP**k for k in range(-GRID_HALF, GRID_HALF + 1)]


def scan(x, y, ell, sf, sn):
    best = None
    for i, e in enumerate(grid(ell)):
        for j, f in enumerate(grid(sf)):
            for k, s in enumerate(grid(sn)):
                value = scalar(log_marginal_likelihood(x, y, e, f, s))
                if best is None or value > best[0]:
                    best = (value, (i, j, k))
    return {"peak": best[0], "peak_offset": [index - GRID_HALF for index in best[1]]}


def paths(x, y, ell, sf, sn):
    k = noisy_kernel(x, ell, sf, sn)
    l = xl.cholesky(k)
    alpha = cho_solve(l, y)
    logdet = 2 * scalar(xp.log(xp.diagonal(l)).sum())
    sign, slog = xl.slogdet(k)
    inv = xl.inv(k)
    return {
        "alpha_solve": gates.rel_diff(xl.solve(k, y), alpha),
        "alpha_inv": gates.rel_diff(inv @ y, alpha),
        "cho_inv": gates.rel_diff(xp.tril(cho_inv(l)), xp.tril(inv)),
        "logdet_slogdet": abs((scalar(slog) - logdet) / logdet),
        "logdet_det": abs((math.log(scalar(xl.det(k))) - logdet) / logdet),
        "slogdet_sign": scalar(sign),
    }


def predict(x, y, x_test, ell, sf, sn):
    l = xl.cholesky(noisy_kernel(x, ell, sf, sn))
    alpha = cho_solve(l, y)
    train_mean = kernel(x, x, ell, sf) @ alpha
    k_star = kernel(x, x_test, ell, sf)
    v = cho_solve(l, k_star)
    variance = sf**2 - (k_star * v).sum(axis=0)
    return {"train_rmse": gates.fro(train_mean - y) / math.sqrt(y.shape[0]),
            "variance_min": scalar(variance.min()),
            "variance_max_over_sf2": scalar(variance.max()) / sf**2}


class GaussianProcess:
    name = "gaussian_process"
    TOL = {"fp64": 5e-13}

    def fixture(self, size):
        return fixtures.read(self.name, size)

    def run(self, fixture):
        x = fixture.device("X")
        y = fixture.device("y")
        x_test = fixture.device("X_test")
        ell, sf, sn = (fixture.attr(key) for key in ("ell", "sf", "sn"))
        likelihood = scan(x, y, ell, sf, sn)
        prediction = predict(x, y, x_test, ell, sf, sn)
        return {"likelihood": likelihood, "paths": paths(x, y, ell, sf, sn), "prediction": prediction, "sn": sn,
                "values": {"log_likelihood_peak": likelihood["peak"], "train_rmse": prediction["train_rmse"],
                           "test_variance_min": prediction["variance_min"]}}

    def gate(self, result):
        tol = self.TOL.get(gates.dtype_name(), gates.tol())
        lik = result["likelihood"]
        p = result["paths"]
        pred = result["prediction"]
        return [
            gates.le("likelihood_peak_grid_steps", max(abs(o) for o in lik["peak_offset"]), 1),
            gates.le("alpha_solve_vs_cho_solve", p["alpha_solve"], tol),
            gates.le("alpha_inv_vs_cho_solve", p["alpha_inv"], tol),
            gates.le("cho_inv_vs_inv", p["cho_inv"], tol),
            gates.le("logdet_cholesky_vs_slogdet", p["logdet_slogdet"], tol),
            gates.le("logdet_cholesky_vs_det", p["logdet_det"], tol),
            gates.eq("slogdet_sign", p["slogdet_sign"], 1.0),
            gates.le("train_rmse_over_sn", pred["train_rmse"] / result["sn"], 1.0),
            gates.ge("test_variance_min", pred["variance_min"], 0.0),
            gates.le("test_variance_max_over_sf2", pred["variance_max_over_sf2"], 1.0),
        ]

    def bench_step(self, fixture, variant=None):
        x = fixture.device("X")
        y = fixture.device("y")
        ell, sf, sn = (fixture.attr(key) for key in ("ell", "sf", "sn"))
        return lambda: log_marginal_likelihood(x, y, ell, sf, sn)


register(GaussianProcess.name, GaussianProcess())
