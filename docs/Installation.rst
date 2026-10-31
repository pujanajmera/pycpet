Installation
=================

*PyCPET* supports Python 3.9–3.12. We recommend Python 3.11, the newest version for which every dependency installs prebuilt on Linux. Python 3.13 is not supported yet (see `#19 <https://github.com/pujanajmera/pycpet/issues/19>`_).

Start from a clean conda environment (recommended, not required):

.. code-block:: console

    conda create -n pycpet-env python=3.11 pip -c conda-forge -y
    conda activate pycpet-env

Then choose one of the following.

Latest release (recommended for most users)
-------------------------------------------

.. code-block:: console

    pip install pycpet

On Linux (x86_64) and on Apple Silicon Macs with macOS 14 or newer, this installs a prebuilt package whose C extension is already compiled with optimizations and OpenMP, so *PyCPET* itself needs no compiler. On other platforms, pip compiles *PyCPET* from source (see `Building from source`_).

One dependency, ``scikit-learn-extra``, only has prebuilt packages for Python 3.11 and older on x86_64. On Python 3.12, and on every Apple Silicon Mac, pip compiles it during installation, which needs a C compiler. On macOS, install the Xcode Command Line Tools first with ``xcode-select --install`` (see `#19 <https://github.com/pujanajmera/pycpet/issues/19>`_).

Latest development version
--------------------------

To get changes before they are released, or to modify the code:

.. code-block:: console

    git clone https://github.com/pujanajmera/pycpet.git
    cd pycpet
    pip install -e .

Update later with ``git pull``, and rerun ``pip install -e .`` if ``CPET/utils/math_module.c`` changed. This compiles the C extension on your machine (see below).

Building from source
--------------------

Source builds (the development version above, platforms without a prebuilt package, or ``pip install --no-binary pycpet pycpet`` to compile a release yourself) need gcc on Linux or the Xcode Command Line Tools on macOS. They compile the C extension for your own CPU (``-march=native``), while prebuilt packages target a portable baseline (AVX2 on x86_64) so they run on any modern machine. On Linux, source builds use OpenMP; on macOS they build without it, while the prebuilt macOS package includes it.

Before running
--------------

Set the number of OpenMP threads:

.. code-block:: console

    export OMP_NUM_THREADS=1

The C extension uses OpenMP, and topology calculations already run several worker processes. Without this setting, each worker starts one thread per CPU core, which can oversubscribe the node and slow topology calculations dramatically.

For GPU-accelerated code (available for topologies, fields in development), choose the GPU with:

.. code-block:: console

    export CUDA_VISIBLE_DEVICES=N

where N is the GPU number. This is only needed outside of HPC schedulers.
