from setuptools import setup

setup(
    cffi_modules=[
        "mstsa/_c_entropy_build.py:ffi",
        "mstsa/_se_build.py:ffi",
        "mstsa/_lz76_build.py:ffi",
    ],
)
