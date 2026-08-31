import os
import sys
from cffi import FFI

ffi = FFI()

ffi.cdef("""
    double c_entropy(int *x, size_t n, size_t m, size_t k);
""")

_src = os.path.join(os.path.dirname(__file__), "src", "c_entropy.c")

# MSVC (Windows) has no separate libm -- math functions are already in the
# C runtime -- and doesn't understand GCC/Clang's "-O3" flag syntax.
_is_msvc = sys.platform == "win32"

ffi.set_source(
    "mstsa._c_entropy",
    """
        double c_entropy(int *x, size_t n, size_t m, size_t k);
    """,
    sources=[_src],
    libraries=[] if _is_msvc else ["m"],
    extra_compile_args=["/O2"] if _is_msvc else ["-O3"],
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
