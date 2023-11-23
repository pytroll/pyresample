#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 12:26:45 2023

@author: ghiggi
"""

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
"""Test AreaDefinition objects."""
import io
import sys
from glob import glob
from unittest.mock import MagicMock, patch
 
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pyproj import CRS, Proj

import pyresample
import pyresample.geometry
from pyresample import geo_filter, parse_area_file
from pyresample.future.geometry import AreaDefinition, SwathDefinition
from pyresample.future.geometry.area import (
    _get_geostationary_bounding_box_in_lonlats,
    get_full_geostationary_bounding_box_in_proj_coords,
    get_geostationary_angle_extent,
    get_geostationary_bounding_box_in_proj_coords,
    ignore_pyproj_proj_warnings,
)
from pyresample.future.geometry.base import get_array_hashable
from pyresample.geometry import AreaDefinition as LegacyAreaDefinition
from pyresample.test.utils import assert_future_geometry


def create_test_area(crs, shape, area_extent):
     """Create an AreaDefinition object for testing."""
     area = AreaDefinition(crs=crs, shape=shape, area_extent=area_extent)
     return area
 

@pytest.fixture
def geos_fd_area():
    """Create full disc geostationary area definition."""
    shape = (100, 100)
    area_extent = (-5500000., -5500000., 5500000., 5500000.)
    proj_dict = {'a': 6378169.00, 'b': 6356583.80, 'h': 35785831.0,
                 'lon_0': 0, 'proj': 'geos', 'units': 'm'}
    return create_test_area(
        crs=proj_dict,
        shape=shape,
        area_extent=area_extent,
    )


@pytest.fixture
def geos_out_disk_area():
    """Create out of Earth diskc geostationary area definition."""
    shape = (10, 10)
    area_extent = (-5500000., -5500000., -5300000., -5300000.)
    proj_dict = {'a': 6378169.00, 'b': 6356583.80, 'h': 35785831.0,
                 'lon_0': 0, 'proj': 'geos', 'units': 'm'}
    return create_test_area(
        crs=proj_dict,
        shape=shape,
        area_extent=area_extent,
    )

@pytest.fixture
def geos_half_out_disk_area():
    """Create geostationary area definition with portion of boundary out of earth_disk."""
    shape = (100, 100)
    area_extent = (-5500000., -10000., 0, 10000.)
    proj_dict = {'a': 6378169.00, 'b': 6356583.80, 'h': 35785831.0,
                 'lon_0': 0, 'proj': 'geos', 'units': 'm'}
    return create_test_area(
        crs=proj_dict,
        shape=shape,
        area_extent=area_extent,
    )


@pytest.fixture
def geos_conus_area():
    """Create CONUS geostationary area definition (portion is out-of-Earth disk)."""
    shape = (30, 50)  # (3000, 5000) for GOES-R CONUS/PACUS
    proj_dict = {'h': 35786023, 'sweep': 'x', 'x_0': 0, 'y_0': 0,
                 'ellps': 'GRS80', 'no_defs': None, 'type': 'crs',
                 'lon_0': -75, 'proj': 'geos', 'units': 'm'}
    area_extent = (-3627271.29128, 1583173.65752, 1382771.92872, 4589199.58952)
    return create_test_area(
        crs=proj_dict,
        shape=shape,
        area_extent=area_extent,
    )

    
class TestBoundary:
    """Test 'boundary' method for AreaDefinition classes."""
    
    def test_get_boundary_sides_call_geostationary_utility1(self, geos_fd_area):
        """Test that the geostationary boundary sides are retrieved correctly."""
        area_def = geos_fd_area
        
        with patch.object(area_def, '_get_geostationary_boundary_sides') as mock_get_geo:

            # Call the method that could trigger the geostationary _get_geostationary_boundary_sides
            _ = area_def._get_boundary_sides(coordinates="geographic", vertices_per_side=None)
            # Assert _get_geostationary_boundary_sides was not called
            mock_get_geo.assert_called_once()
    
    @pytest.mark.parametrize("area_def_name", ["geos_fd_area", "geos_conus_area", "geos_half_out_disk_area"])
    def test_get_boundary_sides_call_geostationary_utility2(self, request, area_def_name):
        """Test that the geostationary boundary sides are retrieved correctly."""
        area_def = request.getfixturevalue(area_def_name)
        
        with patch.object(area_def, '_get_geostationary_boundary_sides') as mock_get_geo:

            # Call the method that could trigger the geostationary _get_geostationary_boundary_sides
            _ = area_def._get_boundary_sides(coordinates="geographic", vertices_per_side=None)
            # Assert _get_geostationary_boundary_sides was not called
            mock_get_geo.assert_called_once() 
            
        
                
    @pytest.mark.parametrize('area_def_name,assert_is_called', [
        ("geos_fd_area", True),
        ("geos_out_disk_area", True),
        ("geos_half_out_disk_area", True),
        ("geos_conus_area", True),
    ])
    def test_get_boundary_sides_call_geostationary_utility(self, request, area_def_name, assert_is_called):
        area_def = request.getfixturevalue(area_def_name)
        
        with patch.object(area_def, '_get_geostationary_boundary_sides') as mock_get_geo:

            # Call the method that could trigger the geostationary _get_geostationary_boundary_sides
            _ = area_def._get_boundary_sides(coordinates="geographic", vertices_per_side=None)
            # Assert _get_geostationary_boundary_sides was not called
            if assert_is_called:
                mock_get_geo.assert_called_once() 
            else:
                mock_get_geo.assert_not_called()
    
 
       
    
   