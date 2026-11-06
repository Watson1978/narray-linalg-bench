"""xp / xl / xsl switch, the Python counterpart of lib/linalg_bench/backend.rb."""

import os

GPU = os.environ.get("GPU", "").lower() in ("1", "on", "true")

if GPU:
    os.environ.setdefault("CUPY_TF32", "0")
    import cupy as xp
    import cupy.linalg as xl
    import cupyx.scipy.linalg as xsl

    def synchronize():
        xp.cuda.Stream.null.synchronize()
else:
    import numpy as xp
    import numpy.linalg as xl
    import scipy.linalg as xsl

    def synchronize():
        pass

_DTYPES = {"": "float32", "fp32": "float32", "sfloat": "float32", "fp64": "float64", "dfloat": "float64"}
_dtype = os.environ.get("DTYPE", "").lower()
if _dtype not in _DTYPES:
    raise ValueError(f"unknown DTYPE {_dtype!r}; use fp32 or fp64")
XF = xp.dtype(_DTYPES[_dtype])


def svd_gesvd(a):
    """Thin SVD through gesvd on both backends.

    CuPy's 2-D svd is gesvd, but numpy's is always gesdd, so numpy goes
    through scipy to run the same algorithm as numo-linalg-alt's default.
    """
    if GPU:
        return xl.svd(a, full_matrices=False)
    return xsl.svd(a, full_matrices=False, lapack_driver="gesvd")


def dtype_name():
    return "fp64" if XF == xp.float64 else "fp32"


def backend_name():
    return "cupy" if GPU else "numpy"


def device(host, dtype=None):
    return xp.asarray(host, dtype=dtype or XF)


def to_host(array):
    return xp.asnumpy(array) if GPU else array


def scalar(value):
    return float(to_host(value))


def describe():
    """One line that says what actually ran, printed by every bench."""
    if not GPU:
        import numpy
        import scipy
        return f"numpy {numpy.__version__} / scipy {scipy.__version__}"
    from cupy_backends.cuda.libs import cublas
    handle = xp.cuda.device.get_cublas_handle()
    mode = cublas.getMathMode(handle)
    return (f"cupy {xp.__version__} / CUPY_TF32={os.environ.get('CUPY_TF32')} / "
            f"cublas math mode {mode}")
