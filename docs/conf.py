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
import sys

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
    "sphinx_book_theme",
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
    "CPET.utils.parallel",
    "CPET.utils.calculator",
    "CPET.utils.io",
    "CPET.utils.gpu",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# Use readthedocs theme
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
