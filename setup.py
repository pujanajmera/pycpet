from setuptools import setup, find_packages

# to setup utils run: pip install -e .

setup(
    name="cpet-python",
    version="0.0.1",
    packages=find_packages(),
    scripts=[
        "./CPET/source/compute_topology.py",
        "./CPET/source/compute_topo_dev.py",
        "./CPET/source/compute_field_volume.py",
        "./CPET/source/compute_topo_dev_multi_process.py"
    ],
)
