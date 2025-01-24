from setuptools import setup, find_packages

# to setup utils run: pip install -e .

setup(
    name="pycpet",
    version="0.0.1",
    packages=find_packages(),
    scripts=[
        "./CPET/source/scripts/cpet.py",
        "./CPET/source/scripts/benchmark_radius_convergence.py",
        "./CPET/source/scripts/benchmark_sample_step.py",
    ],
)
