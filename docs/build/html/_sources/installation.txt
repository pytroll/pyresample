Installing Pyresample
=====================
Pyresample depends on pyproj, numpy(>= 1.3), scipy(>= 0.7), multiprocessing 
(builtin package for Python > 2.5) and configobj.

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
 
Install Pyresample using setuptools:

.. code-block:: bash

	$ easy_install pyresample-<version>.tar.gz
 