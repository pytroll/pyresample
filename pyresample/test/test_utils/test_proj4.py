# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Pyresample developers
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
# You should have received a copy of the GNU Lesser General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Tests for pyproj/PROJ helpers."""

import dask.array as da
import numpy as np
import pytest
from pyproj import CRS

from pyresample.test.utils import assert_maximum_dask_computes


def test_proj4_radius_parameters_provided():
    """Test proj4_radius_parameters with a/b."""
    from pyresample import utils
    a, b = utils.proj4.proj4_radius_parameters(
        '+proj=stere +a=6378273 +b=6356889.44891',
    )
    np.testing.assert_almost_equal(a, 6378273)
    np.testing.assert_almost_equal(b, 6356889.44891)


def test_proj4_radius_parameters_ellps():
    """Test proj4_radius_parameters with ellps."""
    from pyresample import utils
    a, b = utils.proj4.proj4_radius_parameters(
        '+proj=stere +ellps=WGS84',
    )
    np.testing.assert_almost_equal(a, 6378137.)
    np.testing.assert_almost_equal(b, 6356752.314245, decimal=6)


def test_proj4_radius_parameters_default():
    """Test proj4_radius_parameters with default parameters."""
    from pyresample import utils
    a, b = utils.proj4.proj4_radius_parameters(
        '+proj=lcc +lat_0=10 +lat_1=10',
    )
    # WGS84
    np.testing.assert_almost_equal(a, 6378137.)
    np.testing.assert_almost_equal(b, 6356752.314245, decimal=6)


def test_proj4_radius_parameters_spherical():
    """Test proj4_radius_parameters in case of a spherical earth."""
    from pyresample import utils
    a, b = utils.proj4.proj4_radius_parameters(
        '+proj=stere +R=6378273',
    )
    np.testing.assert_almost_equal(a, 6378273.)
    np.testing.assert_almost_equal(b, 6378273.)


def test_proj4_str_dict_conversion():
    from pyresample import utils

    proj_str = "+proj=lcc +ellps=WGS84 +lon_0=-95 +lat_1=25.5 +no_defs"
    proj_dict = utils.proj4.proj4_str_to_dict(proj_str)
    proj_str2 = utils.proj4.proj4_dict_to_str(proj_dict)
    proj_dict2 = utils.proj4.proj4_str_to_dict(proj_str2)
    assert proj_dict == proj_dict2
    assert isinstance(proj_dict['lon_0'], (float, int))
    assert isinstance(proj_dict2['lon_0'], (float, int))
    assert isinstance(proj_dict['lat_1'], float)
    assert isinstance(proj_dict2['lat_1'], float)

    # EPSG
    proj_str = '+init=EPSG:4326'
    proj_dict = utils.proj4.proj4_str_to_dict(proj_str)
    proj_str2 = utils.proj4.proj4_dict_to_str(proj_dict)
    proj_dict2 = utils.proj4.proj4_str_to_dict(proj_str2)
    # pyproj usually expands EPSG definitions so we can only round trip
    assert proj_dict == proj_dict2

    proj_str = 'EPSG:4326'
    proj_dict_exp2 = {'proj': 'longlat', 'datum': 'WGS84', 'no_defs': None, 'type': 'crs'}
    proj_dict = utils.proj4.proj4_str_to_dict(proj_str)
    assert proj_dict == proj_dict_exp2
    # input != output for this style of EPSG code
    # EPSG to PROJ.4 can be lossy
    # self.assertEqual(utils._proj4.proj4_dict_to_str(proj_dict), proj_str)  # round-trip


def test_proj4_str_dict_conversion_with_valueless_parameter():
    from pyresample import utils

    # Value-less south parameter
    proj_str = "+ellps=WGS84 +no_defs +proj=utm +south +type=crs +units=m +zone=54"
    proj_dict = utils.proj4.proj4_str_to_dict(proj_str)
    proj_str2 = utils.proj4.proj4_dict_to_str(proj_dict)
    proj_dict2 = utils.proj4.proj4_str_to_dict(proj_str2)
    assert proj_dict == proj_dict2


def test_convert_proj_floats():
    from collections import OrderedDict

    import pyresample.utils as utils

    pairs = [('proj', 'lcc'), ('ellps', 'WGS84'), ('lon_0', '-95'), ('no_defs', True)]
    expected = OrderedDict([('proj', 'lcc'), ('ellps', 'WGS84'), ('lon_0', -95.0), ('no_defs', True)])
    assert utils.proj4.convert_proj_floats(pairs) == expected

    # EPSG
    pairs = [('init', 'EPSG:4326'), ('EPSG', 4326)]
    for pair in pairs:
        expected = OrderedDict([pair])
        assert utils.proj4.convert_proj_floats([pair]) == expected


@pytest.mark.parametrize("use_dask", [False, True])
@pytest.mark.parametrize("pass_z", [False, True, None])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dask_transformer(use_dask, pass_z, dtype):
    from pyresample.utils.proj4 import DaskFriendlyTransformer

    crs1 = CRS.from_epsg(4326)
    crs2 = CRS.from_epsg(4326)
    x = np.array([1, 2, 3], dtype=dtype)
    y = np.array([1, 2, 3], dtype=dtype)
    extra_args = ()
    if pass_z is None:
        extra_args += (None,)
    elif pass_z:
        z = np.zeros_like(x)
        if use_dask:
            z = da.from_array(z, chunks=1)
        extra_args += (z,)
    if use_dask:
        x = da.from_array(x, chunks=1)
        y = da.from_array(y, chunks=1)

    transformer = DaskFriendlyTransformer.from_crs(crs1, crs2, always_xy=True)
    results = transformer.transform(x, y, *extra_args)
    assert len(results) == (2 if pass_z in (False, None) else 3)

    if use_dask:
        for res in results:
            assert isinstance(res, da.Array)
            assert res.dtype == np.float64
        results = da.compute(*results)

    for res_arr in results:
        assert isinstance(res_arr, np.ndarray)
        assert res_arr.dtype == np.float64
