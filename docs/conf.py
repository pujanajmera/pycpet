# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import re
import sys

from sphinx.util import logging

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "pycpet"
copyright = "2025, Pujan Ajmera, Santiago Vargas, Matthew Hennefarth, Alexandrova Group"
author = "Pujan Ajmera, Santiago Vargas, Matthew Hennefarth, Alexandrova Group"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",  # To generate autodocs
    "sphinx.ext.mathjax",  # autodoc with maths
    "sphinx.ext.napoleon",  # For auto-doc configuration
]

napoleon_google_docstring = False  # Turn off googledoc strings
napoleon_numpy_docstring = True  # Turn on numpydoc strings
napoleon_use_ivar = True  # For maths symbology

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

autodoc_mock_imports = [
    # Needs the compiled math_module C extension, which the docs build doesn't compile
    "CPET.utils.calculator",
    # Heavy or compiled dependencies the docs build doesn't install
    "kneed",
    "matplotlib",
    "numba",
    "scipy",
    "sklearn",
    "sklearn_extra",
    "tensorly",
    "torch",
    "tqdm",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# Use readthedocs theme
#
html_theme = "furo"

# -- Reference order check ---------------------------------------------------

# Fail the build if the References list in citation.rst isn't alphabetical,
# so new references have to be added in their sorted place
def check_reference_order(app):
    with open(os.path.join(app.confdir, "citation.rst")) as f:
        keys = re.findall(r"^\.\. \[([^\]]+)\]", f.read(), flags=re.MULTILINE)
    for previous, current in zip(keys, keys[1:]):
        if current.lower() < previous.lower():
            logging.getLogger(__name__).error(
                "References in citation.rst are out of order: put [%s] before [%s]",
                current,
                previous,
            )
            app.statuscode = 1


def setup(app):
    app.connect("builder-inited", check_reference_order)
