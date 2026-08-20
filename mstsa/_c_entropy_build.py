import os
from cffi import FFI

ffi = FFI()

ffi.cdef("""
    double c_entropy(int *x, size_t n, size_t m, size_t k);
""")

_src = os.path.join(os.path.dirname(__file__), "src", "c_entropy.c")

ffi.set_source(
    "mstsa._c_entropy",
    """
        double c_entropy(int *x, size_t n, size_t m, size_t k);
    """,
    sources=[_src],
    libraries=["m"],
    extra_compile_args=["-O3"],
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
