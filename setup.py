import os

from setuptools import setup

# When MSTSA_PURE is set, build a pure-Python ``py3-none-any`` wheel: skip the
# CFFI/C extensions and drop the dependencies that cannot be satisfied on
# platforms without a C toolchain or LLVM (notably Pyodide / JupyterLite).
# Every C-backed function has a NumPy fallback, and ``numba`` is optional via
# ``mstsa._compat``.
_PURE = bool(os.environ.get("MSTSA_PURE"))

_install_requires = [
    "numpy>=1.21",
    "scipy>=1.7",
    "matplotlib>=3.4",
]
if not _PURE:
    _install_requires += [
        "numba>=0.54",
        "cffi>=1.15",
    ]

_cffi_modules = [] if _PURE else [
    "mstsa/_c_entropy_build.py:ffi",
    "mstsa/_se_build.py:ffi",
    "mstsa/_lz76_build.py:ffi",
]

setup(
    install_requires=_install_requires,
    cffi_modules=_cffi_modules,
)
