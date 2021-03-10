Installing Pyresample
=====================

Pyresample depends on pyproj, numpy(>= 1.10), pyyaml, configobj,
and pykdtree (>= 1.1.1).

In order to use the pyresample plotting functionality Cartopy and
matplotlib (>= 1.0) must be installed. These packages are not a prerequisite
for using any other pyresample functionality.

Optionally, for dask and xarray support these libraries must also be installed.
Some utilities like converting from rasterio objects to pyresample objects
will require rasterio or other libraries to be installed. The older
multiprocessing interfaces (Proj_MP) use the ``scipy`` package's KDTree
implementation. These multiprocessing interfaces are used when the ``nprocs``
keyword argument in the various pyresample interfaces is greater than 1.
Newer xarray/dask interfaces are recommended when possible.

Package test
************

Testing pyresample requires all optional packages to be installed including
rasterio, dask, xarray, cartopy, pillow, and matplotlib. Without all of these
dependencies some tests may fail.
To run tests from a source tarball:

.. code-block:: bash

    tar -zxf pyresample-<version>.tar.gz
    cd pyresample-<version>
    pytest pyresample/test/

If all the tests passes the functionality of all pyresample functions on the system has been verified.

Package installation
********************

Pyresample is available from PyPI and can be installed with pip:

.. code-block:: bash

    pip install pyresample

Pyresample can also be installed with conda via the conda-forge channel:

.. code-block:: bash

    conda install -c conda-forge pyresample

Or directly from a source tarball:

.. code-block:: bash

    tar -zxvf pyresample-<version>.tar.gz
    cd pyresample-<version>
    pip install .

To install in a "development" mode where source file changes are immediately
reflected in your python environment run the following instead of the above
pip command:

    pip install -e .

pykdtree and numexpr
********************

Pyresample uses the ``pykdtree`` package which can be built with
multi-threaded support. If it is built with this support the environment
variable ``OMP_NUM_THREADS`` can be used to control the number of threads.
Please refer to the pykdtree_ repository for more information.

As of pyresample v1.0.0 numexpr_ will be used for minor bottleneck
optimization if available.

.. _pykdtree: https://github.com/storpipfugl/pykdtree
.. _numexpr: https://code.google.com/p/numexpr/
