"""Compatibility shims for optional acceleration dependencies.

``mstsa`` uses ``numba`` only to JIT-compile a handful of pure-NumPy inner
loops.  Numba is unavailable on some platforms (notably Pyodide / JupyterLite,
which has no LLVM), so it is an *optional* dependency: when it is missing this
module provides a no-op ``jit`` decorator and the decorated functions run as
plain (slower) Python.

Install the ``speed`` extra (``pip install mstsa[speed]``) to get real JIT
compilation.
"""

try:  # pragma: no cover - exercised implicitly by the test suite
    from numba import jit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - Pyodide / numba-less installs
    HAVE_NUMBA = False

    def jit(*args, **kwargs):
        """No-op stand-in for :func:`numba.jit`.

        Supports both bare use (``@jit``) and call-with-options use
        (``@jit(nopython=True, cache=True)``); in either case the wrapped
        function is returned unchanged.
        """
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator
