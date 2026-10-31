"""
Build the math_module C extension. Project metadata lives in pyproject.toml.

Wheels built in CI (cibuildwheel sets CIBUILDWHEEL=1) target a portable CPU
baseline, since each wheel is compiled once and installed on many machines.
Every other build (installs from the sdist, pip install ., local development)
is tuned to the CPU doing the build with -march=native.
"""

import os
import platform
import sys

from setuptools import Extension, setup

compile_args = ["-O3", "-funroll-loops", "-ffast-math", "-fPIC"]

if os.environ.get("CIBUILDWHEEL") == "1":
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        # AVX2/FMA baseline, equivalent to x86-64-v3 (the manylinux2014 GCC is too old for that name)
        compile_args += ["-march=haswell", "-mtune=generic"]
    elif sys.platform == "darwin" and machine == "arm64":
        compile_args += ["-mcpu=apple-m1"]
else:
    compile_args += ["-march=native"]

setup(
    ext_modules=[
        Extension(
            "CPET.utils.math_module",
            sources=["CPET/utils/math_module.c"],
            extra_compile_args=compile_args,
        )
    ]
)
