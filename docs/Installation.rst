Installation
=================

PyCPET set-up
-----------------

PyCPET is a Python package that is coded in Python3, tested on Linux operating systems (Mac testing soon).

PyCPET can be downloaded directory from the `GitHub repository <https://github.com/pujanajmera/pycpet>` (recommended) or installed via pip.

Installing PyCPET via pip (recommended for general users)
---------------------------------------------------------
1. Install via pip:

.. code-block:: console

    pip install pycpet

Installing PyCPET from GitHub (recommended for developers)
----------------------------------------------------------

1. Clone the repository:

.. code-block:: console

    git clone https://github.com/pujanajmera/pycpet.git

2. Change to the directory:

.. code-block:: console
    cd pycpet

3. Install dependencies via conda:

.. code-block:: console

    conda env create -f environment.yml

4. Install the package:

.. code-block:: console

    pip install -e .
