import os
import sys
from cffi import FFI

ffi = FFI()

ffi.cdef("""
    double sample_entropy_cont(double *x, size_t n, size_t m, size_t tau, double r);
    double sample_entropy_disc(long long *x, size_t n, size_t m, size_t tau);
    int    sample_entropy_cont_fast(double *se, double *x, size_t n, size_t m, double r);
    int    sample_entropy_disc_fast(double *se, long long *x, size_t n, size_t m);
""")

_src = os.path.join(os.path.dirname(__file__), "src", "se.c")

# MSVC (Windows) has no separate libm -- math functions are already in the
# C runtime -- and doesn't understand GCC/Clang's "-O3" flag syntax.
_is_msvc = sys.platform == "win32"

ffi.set_source(
    "mstsa._se",
    """
        double sample_entropy_cont(double *x, size_t n, size_t m, size_t tau, double r);
        double sample_entropy_disc(long long *x, size_t n, size_t m, size_t tau);
        int    sample_entropy_cont_fast(double *se, double *x, size_t n, size_t m, double r);
        int    sample_entropy_disc_fast(double *se, long long *x, size_t n, size_t m);
    """,
    sources=[_src],
    libraries=[] if _is_msvc else ["m"],
    extra_compile_args=["/O2"] if _is_msvc else ["-O3"],
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
