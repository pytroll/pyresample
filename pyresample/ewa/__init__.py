#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016

# Author(s):

#   David Hoese <david.hoese@ssec.wisc.edu>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Code for resampling using the Elliptical Weighted Averaging (EWA) algorithm.

The logic and original code for this algorithm were translated from the
software package "MODIS Swath 2 Grid Toolbox" or "ms2gt" created by the
NASA National Snow & Ice Data Center (NSIDC):

    https://nsidc.org/data/modis/ms2gt/index.html

Since the project has slowed down, Terry Haran has maintained the package
and made updates available:

    http://cires1.colorado.edu/~tharan/ms2gt/

The ms2gt C executables "ll2cr" and "fornav" were rewritten for the
Polar2Grid software package created by the Space Science Engineering Center
(SSEC)/Cooperative Institute for Meteorological Satellite Studies. They were
rewritten as a combination of C++ and Cython to make them more python friendly
by David Hoese and were then copied and modified here in pyresample. The
rewrite of "ll2cr" also included an important switch from using the "mapx"
library to using the more popular and capable pyproj (PROJ.4) library.

The EWA algorithm consists of two parts "ll2cr" and "fornav" and are described
below.

ll2cr
-----

The "ll2cr" process is the first step in the EWA algorithm. It stands for
"latitude/longitude to column/row". Its main purpose is to convert
input longitude and latitude coordinates to column and row coordinates
of the destination grid. These coordinates are then used in the next step
"fornav".

fornav
------

The "fornav" or "Forward Navigation" step of the EWA algorithm is where
the actual Elliptical Weighted Averaging algorithm is run. The algorithm
maps input swath pixels to output grid pixels by averaging multiple input
pixels based on an elliptical region and other coefficients, some of which
are determined at run time.

For more information on these steps see the documentation for the
corresponding modules.
"""

from .ewa import fornav, ll2cr  # noqa

try:
    from ._legacy_dask_ewa import LegacyDaskEWAResampler  # noqa
    from .dask_ewa import DaskEWAResampler  # noqa
except ImportError:
    # dask is required but not installed
    # fallback to old numpy-only implementation (ll2cr, fornav above)
    pass
