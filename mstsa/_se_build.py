import os
from cffi import FFI

ffi = FFI()

ffi.cdef("""
    double sample_entropy_cont(double *x, size_t n, size_t m, size_t tau, double r);
    double sample_entropy_disc(long *x, size_t n, size_t m, size_t tau);
    int    sample_entropy_cont_fast(double *se, double *x, size_t n, size_t m, double r);
    int    sample_entropy_disc_fast(double *se, long *x, size_t n, size_t m);
""")

_src = os.path.join(os.path.dirname(__file__), "src", "se.c")

ffi.set_source(
    "mstsa._se",
    """
        double sample_entropy_cont(double *x, size_t n, size_t m, size_t tau, double r);
        double sample_entropy_disc(long *x, size_t n, size_t m, size_t tau);
        int    sample_entropy_cont_fast(double *se, double *x, size_t n, size_t m, double r);
        int    sample_entropy_disc_fast(double *se, long *x, size_t n, size_t m);
    """,
    sources=[_src],
    libraries=["m"],
    extra_compile_args=["-O3"],
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
