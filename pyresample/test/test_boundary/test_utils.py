#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pyresample, Resampling of remote sensing image data in python
#
# Copyright (C) 2010-2022 Pyresample developers
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Test the boundary utility."""
import numpy as np
import pytest
from pyresample.boundary.utils import (
    find_boundary_mask,
    find_boundary_contour_indices, 
    get_ordered_contour,
)
    

@pytest.mark.parametrize("lonlat, expected", [
    # Case: All True values
    ((np.array([[1, 2, 3, 4], 
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4]]),
      np.array([[1, 2, 3, 4], 
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4]])),
     np.array([[True,  True,  True,  True],
               [True,  False, False, True],
               [True,  False, False, True],
               [True,  True,  True,  True]])),

    # Case: Multiple True values in the center
    ((np.array([[np.inf, np.inf, np.inf, np.inf], 
                [np.inf, 2, 3, np.inf],
                [np.inf, 2, 3, np.inf],
                [np.inf, np.inf, np.inf, np.inf]]),
      np.array([[np.inf, np.inf, np.inf, np.inf], 
                  [np.inf, 2, 3, np.inf],
                  [np.inf, 2, 3, np.inf],
                  [np.inf, np.inf, np.inf, np.inf]])),
     np.array([[False, False, False, False],
               [False, True,  True,  False],
               [False, True,  True,  False],
               [False, False, False, False]])),
])
def test_find_boundary_mask(lonlat, expected):
    """Test boundary mask for lon lat array with non finite values."""
    lons, lats = lonlat
    result = find_boundary_mask(lons, lats)
    np.testing.assert_array_equal(result, expected, err_msg=f"Expected {expected}, but got {result}")
    

@pytest.mark.parametrize("boundary_mask, expected", [
    # Case: All True values
    (np.array([[True,  True,  True,  True],
               [True,  False, False, True],
               [True,  False, False, True],
               [True,  True,  True,  True]]),
     np.array([[0, 0], [0, 1], [0, 2], [0, 3],
               [1, 3], [2, 3], [3, 3], 
               [3, 2], [3, 1], [3, 0],
               [2, 0], [1, 0]])
    ),
    # Case: Multiple True values in the center
    (np.array([[False, False, False, False],
               [False, True,  True,  False],
               [False, True,  True,  False],
               [False, False, False, False]]),
     np.array([[1, 1], [1, 2], [2, 2], [2, 1]])
    ),           
    # Case: Square on angle 
    (np.array([[True, True, True, False],
               [True, False, True, False],
               [True, True, True, False],
               [False, False, False, False]]),
     np.array([[0, 0], [0, 1], [0, 2], [1, 2], [2, 2], [2, 1], [2, 0], [1, 0]])
    ),
    # Case: Cross Pattern  
    (np.array([[False, True, True, False],
              [True, False, False, True],
              [True, False, False, True],
              [False, True, True, False]]),
    np.array([[0, 1], [0, 2], [1, 3], [2, 3], [3, 2],  [3, 1], [2, 0], [1, 0]])
    ),
    # Case: Possibile infinit loop if not checking visited 
    (np.array([[1, 1, 1, 1, 0],
               [1, 0, 1, 0, 0],
               [1, 0, 1, 0, 0],
               [1, 1, 0, 0, 0],
               [1, 0, 0, 0, 0],
               [1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]),
    np.array([[0, 0], [0, 1], [0, 2], [0, 3], [1, 2], [2, 2], [3, 1], [3, 0], [2, 0], [1, 0]])
    ),  
])
def test_get_ordered_contour(boundary_mask, expected):
    """Test order of the boundary contour indices (clockwise)."""
    result = get_ordered_contour(boundary_mask)
    np.testing.assert_array_equal(result, expected, err_msg=f"Expected {expected}, but got {result}")


@pytest.mark.parametrize("lonlat, expected", [
    # Case: All True values
    ((np.array([[1, 2, 3, 4], 
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4]]),
      np.array([[1, 2, 3, 4], 
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4]])),
     np.array([[0, 0], [0, 1], [0, 2], [0, 3],
               [1, 3], [2, 3], [3, 3], 
               [3, 2], [3, 1], [3, 0],
               [2, 0], [1, 0]])
    ),
    # Case: Multiple True values in the center
    ((np.array([[np.inf, np.inf, np.inf, np.inf], 
                [np.inf, 2, 3, np.inf],
                [np.inf, 2, 3, np.inf],
                [np.inf, np.inf, np.inf, np.inf]]),
      np.array([[np.inf, np.inf, np.inf, np.inf], 
                  [np.inf, 2, 3, np.inf],
                  [np.inf, 2, 3, np.inf],
                  [np.inf, np.inf, np.inf, np.inf]])),
     np.array([[1, 1], [1, 2], [2, 2], [2, 1]])
    ),           
])
def test_find_boundary_contour_indices(lonlat, expected):
    """Test order of the boundary contour indices (clockwise)."""
    lons, lats = lonlat
    result = find_boundary_contour_indices(lons, lats)
    np.testing.assert_array_equal(result, expected, err_msg=f"Expected {expected}, but got {result}")