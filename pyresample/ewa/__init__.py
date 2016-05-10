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

import logging
import numpy as np
from pyresample.ewa import _ll2cr, _fornav

LOG = logging.getLogger(__name__)


def ll2cr(swath_def, area_def, fill=np.nan, **kwargs):
    lons, lats = swath_def.get_lonlats()
    # ll2cr requires 64-bit floats due to pyproj limitations
    # also need a copy of lons, lats since they are written to in-place
    lons = lons.astype(np.float64, copy=kwargs.get("copy", True))
    lats = lats.astype(np.float64, copy=kwargs.get("copy", True))

    # Break the input area up in to the expected parameters for ll2cr
    p = area_def.proj4_string
    cw = area_def.pixel_size_x
    # cell height must be negative for this to work as expected
    ch = -abs(area_def.pixel_size_y)
    w = area_def.x_size
    h = area_def.y_size
    ox = area_def.area_extent[0]
    oy = area_def.area_extent[3]
    swath_points_in_grid = _ll2cr.ll2cr_static(lons, lats, fill,
                                               p, cw, ch, w, h, ox, oy)
    return swath_points_in_grid, lons, lats


def fornav(cols, rows, area_def, *data_in, **kwargs):
    # we can only support one data type per call at this time
    assert(in_arr.dtype == data_in[0].dtype for in_arr in data_in[1:])

    # need a list for replacing these arrays later
    data_in = list(data_in)
    # determine a fill value
    if "fill" in kwargs:
        # they told us what they have as a fill value in the numpy arrays
        fill = kwargs["fill"]
    elif np.issubdtype(data_in[0].dtype, np.floating):
        fill = np.nan
    elif np.issubdtype(data_in[0].dtype, np.integer):
        fill = -999
    else:
        raise ValueError("Unsupported input data type for EWA Resampling: {}".format(data_in[0].dtype))

    convert_to_masked = False
    for idx, in_arr in enumerate(data_in):
        if isinstance(in_arr, np.ma.MaskedArray):
            convert_to_masked = True
            # convert masked arrays to single numpy arrays
            data_in[idx] = in_arr.filled(fill)
    data_in = tuple(data_in)

    if "outs" in kwargs:
        # the user may have provided memmapped arrays or other array-like objects
        outs = tuple(kwargs["outs"])
    else:
        # create a place for output data to be written
        outs = tuple(np.empty(area_def.shape, dtype=in_arr.dtype) for in_arr in data_in)

    # see if the user specified rows per scan
    # otherwise, use the entire swath as one "scanline"
    rows_per_scan = kwargs.get("rows_per_scan") or data_in[0].shape[0]

    results = _fornav.fornav_wrapper(cols, rows, data_in, outs,
                                     np.nan, np.nan, rows_per_scan)

    def _mask_helper(data, fill):
        if np.isnan(fill):
            return np.isnan(data)
        else:
            return data == fill

    if convert_to_masked:
        # they gave us masked arrays so give them masked arrays back
        outs = [np.ma.masked_where(_mask_helper(out_arr, fill), out_arr) for out_arr in outs]
    if len(outs) == 1:
        # they only gave us one data array as input, so give them one back
        outs = outs[0]

    return results, outs

