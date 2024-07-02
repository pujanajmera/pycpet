from setuptools import setup, find_packages

# to setup utils run: pip install -e .

setup(
    name="cpet-python",
    version="0.0.1",
    packages=find_packages(),
    scripts=[
        "./CPET/source/scripts/cpet.py",
        "./tests/benchmark_radius_convergence.py",
        "./tests/benchmark_sample_step.py",
    ],
)
