#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017

# Author(s):

#   Panu Lahtinen <panu.lahtinen@fmi.fi>

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

"""Code for resampling using bilinear algorithm for irregular grids.

The algorithm is taken from

http://www.ahinson.com/algorithms_general/Sections/InterpolationRegression/InterpolationIrregularBilinear.pdf

"""

import logging
import numpy as np
from pyproj import Proj

from pyresample import kd_tree

LOG = logging.getLogger(__name__)


def resample_bilinear(data, t__, s__, input_idxs, idx_arr, output_shape):
    """Resample array *data* using irregular bilinear algorithm.
    Arguments *t*, *s*, *input_idxs* and *idx_arr* are pre-calculated
    using function pyresample.bilinear.calc_params().  *output_shape* is
    the shape of the output array, ie. pixel counts as a tuple in (y, x)
    directions."""

    # Ravel the data
    new_data = data.ravel()[input_idxs]
    data_min = np.nanmin(new_data)
    data_max = np.nanmax(new_data)

    new_data = new_data[idx_arr]

    # Get neighbour data to separate variables
    p_1 = new_data[:, 0]
    p_2 = new_data[:, 1]
    p_3 = new_data[:, 2]
    p_4 = new_data[:, 3]

    result = (p_1 * (1 - s__) * (1 - t__) +
              p_2 * s__ * (1 - t__) +
              p_3 * (1 - s__) * t__ +
              p_4 * s__ * t__)

    mask = ((result > data_max) | (result < data_min) |
            np.isnan(result) | result.mask)

    result = np.ma.masked_where(mask, result.data)

    result = result.reshape(output_shape)

    return result


def calc_params(in_area, out_area, radius=50e3, neighbours=32, nprocs=1,
                masked=False):
    """Calculate parameters *s* and *t* for bilinear parametric lines.
    Returns also valid input indices and mapping array from valid
    input pixels to output pixels.  The input parameters are the input
    and output area definitions, radius of the search area in meters,
    number of neigbours to colelct and the number of CPUs to use for
    the neighbour search.
    """

    # Calculate neighbour information
    (input_idxs, output_idxs, idx_ref, dists) = \
        kd_tree.get_neighbour_info(in_area, out_area,
                                   radius, neighbours=neighbours,
                                   nprocs=nprocs)

    del output_idxs, dists

    # Reduce index reference
    input_size = input_idxs.sum()
    index_mask = (idx_ref == input_size)
    idx_ref = np.where(index_mask, 0, idx_ref)

    # Get output projection as pyproj object
    proj = Proj(out_area.proj4_string)

    # Get output x/y coordinates
    out_x, out_y = get_output_xy(out_area, proj)

    # Get input x/ycoordinates
    in_x, in_y = get_input_xy(in_area, proj, input_idxs, idx_ref)

    # Get the four closest corner points around each output location
    pt_1, pt_2, pt_3, pt_4, idx_ref = get_four_closest(in_x, in_y,
                                                       out_x, out_y,
                                                       neighbours, idx_ref)

    # Pairwise longitudal separations between reference points
    x_21 = pt_2[0] - pt_1[0]
    x_31 = pt_3[0] - pt_1[0]
    x_42 = pt_4[0] - pt_2[0]

    # Pairwise latitudal separations between reference points
    y_21 = pt_2[1] - pt_1[1]
    y_31 = pt_3[1] - pt_1[1]
    y_42 = pt_4[1] - pt_2[1]

    # Parameters for 2nd order polynomial
    a__ = x_31 * y_42 - y_31 * x_42
    b__ = out_y * (x_42 - x_31) - out_x * (y_42 - y_31) + \
        x_31 * pt_2[1] - y_31 * pt_2[0] + pt_1[0] * y_42 - pt_1[1] * x_42
    c__ = out_y * x_21 - out_x * y_21 + pt_1[0] * pt_2[1] - pt_2[0] * pt_1[1]

    # Get the valid roots from interval [0, 1]
    t__ = solve_quadratic(a__, b__, c__, min_val=0., max_val=1.)

    # Calculate parameter s
    s__ = ((out_y - pt_1[1] - y_31 * t__) /
           (pt_2[1] + y_42 * t__ - pt_1[1] - y_31 * t__))

    # Limit also values of s to interval [0, 1]
    idxs = (s__ < 0) | (s__ > 1)
    s__ = np.ma.masked_where(idxs, s__)

    # Remove mask and put np.nan at the masked locations instead
    if not masked:
        mask = t__.mask | s__.mask
        t__ = t__.data
        t__[mask] = np.nan
        s__ = s__.data
        s__[mask] = np.nan

    return t__, s__, input_idxs, idx_ref


def mask_coordinates(lons, lats):
    """Mask invalid coordinate values"""
    lons = lons.ravel()
    lats = lats.ravel()
    idxs = ((lons < -180.) | (lons > 180.) |
            (lats < -90.) | (lats > 90.))
    lons = np.ma.masked_where(idxs, lons)
    lats = np.ma.masked_where(idxs, lats)

    return lons, lats


def get_corner(stride, valid, in_x, in_y, idx_ref):
    """Get closest set of coordinates from the *valid* locations"""
    idxs = np.argmax(valid, axis=1)
    invalid = np.invert(np.max(valid, axis=1))
    x__ = np.ma.masked_where(invalid, in_x[stride, idxs])
    y__ = np.ma.masked_where(invalid, in_y[stride, idxs])
    idx = idx_ref[stride, idxs]

    return x__, y__, idx


def get_four_closest(in_x, in_y, out_x, out_y, neighbours, idx_ref):
    """"""

    # Find four closest pixels around the target location

    # Tile output coordinates to same shape as neighbour info
    out_x_tile = np.tile(out_x, (neighbours, 1)).T
    out_y_tile = np.tile(out_y, (neighbours, 1)).T

    # Get differences in both directions
    x_diff = out_x_tile - in_x
    y_diff = out_y_tile - in_y

    stride = np.arange(x_diff.shape[0])

    # Upper left source pixel
    valid = (x_diff > 0) & (y_diff < 0)
    x_1, y_1, idx_1 = get_corner(stride, valid, in_x, in_y, idx_ref)

    # Upper right source pixel
    valid = (x_diff < 0) & (y_diff < 0)
    x_2, y_2, idx_2 = get_corner(stride, valid, in_x, in_y, idx_ref)

    # Lower left source pixel
    valid = (x_diff > 0) & (y_diff > 0)
    x_3, y_3, idx_3 = get_corner(stride, valid, in_x, in_y, idx_ref)

    # Lower right source pixel
    valid = (x_diff < 0) & (y_diff > 0)
    x_4, y_4, idx_4 = get_corner(stride, valid, in_x, in_y, idx_ref)

    # Combine sorted indices to idx_ref
    idx_ref = np.vstack((idx_1, idx_2, idx_3, idx_4)).T

    return (x_1, y_1), (x_2, y_2), (x_3, y_3), (x_4, y_4), idx_ref


def solve_quadratic(a__, b__, c__, min_val=0.0, max_val=1.0):
    """Solve quadratic equation and return the valid roots from interval
    [*min_val*, *max_val*]"""

    # Mask out division by zero
    a__ = np.ma.masked_where(a__ == 0, a__)

    # Mask out invalid (complex) discriminants
    discriminant = b__ * b__ - 4 * a__ * c__
    idxs = discriminant < 0
    discriminant = np.ma.masked_where(idxs, discriminant)

    # Solve the quadratic polynomial
    t_1 = (-b__ + np.sqrt(discriminant)) / (2 * a__)
    t_2 = (-b__ - np.sqrt(discriminant)) / (2 * a__)

    # Find valid solutions, ie. 0 <= t <= 1
    t__ = t_1.copy()
    idxs = (t_1 < min_val) | (t_1 > max_val)
    t__[idxs] = t_2[idxs]

    idxs = (t__ < min_val) | (t__ > max_val)
    t__ = np.ma.masked_where(idxs, t__)

    return t__


def get_output_xy(out_area, proj):
    """Get x/y coordinates of the target grid."""
    # Read output coordinates
    out_lons, out_lats = out_area.get_lonlats()
    out_lons, out_lats = mask_coordinates(out_lons, out_lats)

    out_x, out_y = proj(out_lons, out_lats)

    return out_x, out_y


def get_input_xy(in_area, proj, input_idxs, idx_ref):
    """Get x/y coordinates for the input area and reduce the data."""
    in_lons, in_lats = in_area.get_lonlats()

    # Select valid locations
    in_lons = in_lons.ravel()[input_idxs]
    in_lats = in_lats.ravel()[input_idxs]

    # Mask invalid values
    in_lons, in_lats = mask_coordinates(in_lons, in_lats)

    # Expand input coordinates for each output location
    in_lons = in_lons[idx_ref]
    in_lats = in_lats[idx_ref]

    # Convert coordinates to output projection x/y space
    in_x, in_y = proj(in_lons, in_lats)

    return in_x, in_y
