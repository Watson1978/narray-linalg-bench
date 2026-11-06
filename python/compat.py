"""numo-linalg-alt functions that CuPy lacks, built from what it has (docs/cupy-gaps.md).

numpy goes through the same code, so both Python backends run one algorithm.
"""

from backend import scalar, xl, xp, xsl


def cutoff(a):
    return max(a.shape) * float(xp.finfo(a.dtype).eps)


def svdvals(a):
    return xl.svd(a, compute_uv=False)


def null_space(a, rcond=None):
    _, s, vh = xl.svd(a, full_matrices=True)
    rank = int(scalar((s > (rcond or cutoff(a)) * s.max()).sum()))
    return vh[rank:].T.conj()


def orth(a, rcond=None):
    u, s, _vt = xl.svd(a, full_matrices=False)
    rank = int(scalar((s > (rcond or cutoff(a)) * s.max()).sum()))
    return u[:, :rank]


def cho_solve(l, b):
    return xsl.solve_triangular(l, xsl.solve_triangular(l, b, lower=True), lower=True, trans="T")


def cho_inv(l):
    return cho_solve(l, xp.eye(l.shape[0], dtype=l.dtype))


def lu_inv(factor):
    lu, _piv = factor
    return xsl.lu_solve(factor, xp.eye(lu.shape[0], dtype=lu.dtype))


def eigh_general(a, b):
    """eigh(a, b) for symmetric a and SPD b. scipy has it (sygvd); CuPy does not,
    so it reduces to a standard problem through the Cholesky factor of b."""
    if hasattr(xsl, "eigh"):
        return xsl.eigh(a, b, driver="gvd")
    l = xl.cholesky(b)
    linv_a = xsl.solve_triangular(l, a, lower=True)
    m = xsl.solve_triangular(l, linv_a.T, lower=True)
    w, v = xl.eigh(m)
    return w, xsl.solve_triangular(l, v, lower=True, trans="T")
