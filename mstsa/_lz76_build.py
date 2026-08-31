import os
import sys
from cffi import FFI

ffi = FFI()

ffi.cdef("""
    int lz76(int *x, int n);
""")

_src = os.path.join(os.path.dirname(__file__), "src", "lz76.c")

# MSVC (Windows) doesn't understand GCC/Clang's "-O3" flag syntax.
_is_msvc = sys.platform == "win32"

ffi.set_source(
    "mstsa._lz76",
    """
        int lz76(int *x, int n);
    """,
    sources=[_src],
    extra_compile_args=["/O2"] if _is_msvc else ["-O3"],
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
