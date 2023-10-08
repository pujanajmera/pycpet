from setuptools import setup, find_packages

# to setup utils run: pip install -e .

setup(
    name="cpet-python",
    version="0.0.1",
    packages=find_packages(),
    scripts=[
        "./CPET/compute_topology.py",
    ],
)
