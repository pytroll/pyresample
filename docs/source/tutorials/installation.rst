Installation
============

Pyresample can be installed from PyPI via pip or in a conda environment
using the conda-forge channel. The below sections will show the possible
ways to install Pyresample and what customizations can be made.

With pip
--------

Pyresample is available from PyPI and can be installed with pip:

.. code-block:: bash

   pip install pyresample

With conda
----------

Pyresample can also be installed with conda or mamba via the conda-forge
channel:

.. code-block:: bash

   conda install -c conda-forge pyresample

From source
-----------

You can install Pyresample from a source tarball (or other source directory):

.. code-block:: bash

   tar -zxvf pyresample-<version>.tar.gz
   cd pyresample-<version>
   pip install .

You could also install directly from github:

.. code-block:: bash

   pip install git+https://github.com/pytroll/pyresample.git@main

Where ``main`` is the primary git branch. This branch name and the user
account in the URL (``pytroll`` above) can be customized to install
in-development git branches.

For development
---------------

If you'd like to edit Pyresample and see the effects of the changes on your
current environment, you can install it in "editable" mode:

.. code-block:: bash

   pip install -e .

Note that Pyresample has some C extensions that must be recompiled if modified.
This compilation only happens during installation/build time so the above
command needs to be rerun to see the effects of changes to these extension
modules.

Run tests
---------

Testing pyresample requires all optional packages to be installed.
Without all of these dependencies some tests may fail.
To run tests from a local source directory:

.. code-block:: bash

    pytest pyresample/test/

Or you can run it on an installed version of the package:

.. code-block:: bash

    pytest --pyargs pyresample.test

If all the tests passes the functionality of all pyresample functions on the
system has been verified.

Optional dependencies
---------------------

Pyresample has a lot of functionality that may not be necessary for all
users. These features only import their dependencies when used so it may
not be obvious that you need them until after installation. These dependencies
are not installed by default and must be installed separately.

In order to use the Pyresample plotting functionality ``cartopy`` and
``matplotlib`` must be installed. These packages are not a prerequisite
for using any other pyresample functionality.

Additionally, for ``dask`` and ``xarray`` support these libraries must also be
installed. Some utility functions may have additional, hopefully obvious,
dependencies. For example, converting an object from the ``rasterio``
library requires ``rasterio`` to be installed.

Portions of Pyresample offer non-dask multiprocessing interfaces and may have
additional dependencies to accomplish this. For example, when ``nprocs`` is
available and specified with a value greater than 1, a special ``Proj_MP`` will
be used and requires the ``KDTree`` class from the ``scipy`` package.
Newer xarray/dask interfaces are recommended when possible.

Some of pyresamples functionality uses the ``KDTree`` object from the
``pykdtree`` package. This package benefits from being built with
multi-threaded support via the OpenMP library, but this support is not
always built-in by default. See the
`pykdtree README <https://github.com/storpipfugl/pykdtree/blob/master/README.rst>`_
for build tips and suggestions. You may want to control the number of threads
used by pykdtree with the ``OMP_NUM_THREADS`` environment variable.

Pyresample also uses the `numexpr <https://github.com/pydata/numexpr>`_
package for some minor bottleneck optimization if available.
