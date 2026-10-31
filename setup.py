"""
Build the math_module C extension. Project metadata lives in pyproject.toml.

Flags follow the "faster" build line at the top of math_module.c:
    gcc -fopenmp math_module.c -o math_module.so -shared -fPIC -O3 -march=native -funroll-loops -ffast-math

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
link_args = []

# Apple clang needs libomp and -Xpreprocessor for OpenMP; the macOS wheel build
# passes those in through CPPFLAGS/CFLAGS/LDFLAGS instead
if sys.platform != "darwin":
    compile_args.append("-fopenmp")
    link_args.append("-fopenmp")

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
            extra_link_args=link_args,
        )
    ]
)
