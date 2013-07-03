Installing Pyresample
=====================
Pyresample depends on pyproj, numpy(>= 1.3), scipy(>= 0.7), multiprocessing 
(builtin package for Python > 2.5) and configobj. Optionally pykdtree can be used instead of scipy from v0.8.0.

The correct version of the packages should be installed on your system 
(refer to numpy and scipy installation instructions) or use easy_install to handle dependencies automatically.

In order to use the pyresample plotting functionality Basemap and matplotlib (>= 0.98) must be installed. 
These packages are not a prerequisite for using any other pyresample functionality. 

Package test
************
Test the package (requires nose):

.. code-block:: bash

	$ tar -zxvf pyresample-<version>.tar.gz
	$ cd pyresample-<version>
	$ nosetests
	
If all the tests passes the functionality of all pyresample functions on the system has been verified.

Package installation
********************
A sandbox environment can be created for pyresample using `Virtualenv <http://pypi.python.org/pypi/virtualenv>`_

Pyresample is available from pypi.
  
Install Pyresample using pip:

.. code-block:: bash

	$ pip install pyresample

Alternatively install from tarball:

.. code-block:: bash

	$ tar -zxvf pyresample-<version>.tar.gz
	$ cd pyresample-<version>
	$ python setup.py install

Using pykdtree
**************

As of pyresample v0.8.0 pykdtree can be used as backend instead of scipy. 
This enables significant speedups for large datasets.

pykdtree is used as a drop-in replacement for scipy. If it's available it will be used otherwise scipy will be used.
To check which backend is active for your pyresample installation do:

 >>> import pyresample as pr
 >>> pr.kd_tree.which_kdtree()

which returns either 'pykdtree' or 'scipy.spatial'.

Please refere to pykdtree_ for installation description.

If pykdtree is built with OpenMP support the number of threads is controlled with the standard OpenMP environment variable OMP_NUM_THREADS.
The *nprocs* argument has no effect on pykdtree.

Using numexpr
*************

As of pyresample v1.0.0 numexpr_ will be used for minor bottleneck optimization if available

Show active plugins
*******************
The active drop-in plugins can be show using:

 >>> import pyresample as pr
 >>> pr.get_capabilities()

.. _pykdtree: https://github.com/storpipfugl/pykdtree
.. _numexpr: https://code.google.com/p/numexpr/
 
 