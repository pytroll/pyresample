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
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pyproj import CRS

from pyresample.test.utils import create_test_latitude, create_test_longitude


def _gen_swath_def_xarray_dask(create_test_swath):
    """Create a SwathDefinition with xarray[dask] data inside.

    Note that this function is not a pytest fixture so that each call returns a
    new instance of the swath definition with new instances of the data arrays.

    """
    lons, lats = _gen_swath_lons_lats()
    lons_dask = da.from_array(lons)
    lats_dask = da.from_array(lats)
    lons_xr = xr.DataArray(lons_dask, dims=('y', 'x'), attrs={'name': 'Longitude', 'resolution': 500, 'units': 'm'})
    lats_xr = xr.DataArray(lats_dask, dims=('y', 'x'), attrs={'name': 'Latitude', 'resolution': 500, 'units': 'm'})
    return create_test_swath(lons_xr, lats_xr)


def _gen_swath_def_xarray_numpy(create_test_swath):
    lons, lats = _gen_swath_lons_lats()
    lons_xr = xr.DataArray(lons, dims=('y', 'x'))
    lats_xr = xr.DataArray(lats, dims=('y', 'x'))
    return create_test_swath(lons_xr, lats_xr)


def _gen_swath_def_dask(create_test_swath):
    lons, lats = _gen_swath_lons_lats()
    lons_dask = da.from_array(lons)
    lats_dask = da.from_array(lats)
    return create_test_swath(lons_dask, lats_dask)


def _gen_swath_def_numpy(create_test_swath):
    lons, lats = _gen_swath_lons_lats()
    return create_test_swath(lons, lats)


def _gen_swath_def_numpy_small_noncontiguous(create_test_swath):
    swath_def = _gen_swath_def_numpy_small(create_test_swath)
    swath_def_subset = swath_def[:, slice(0, 2)]
    return swath_def_subset


def _gen_swath_def_numpy_small(create_test_swath):
    lons = np.array([[1.2, 1.3, 1.4, 1.5],
                     [1.2, 1.3, 1.4, 1.5]])
    lats = np.array([[65.9, 65.86, 65.82, 65.78],
                     [65.9, 65.86, 65.82, 65.78]])
    swath_def = create_test_swath(lons, lats)
    return swath_def


def _gen_swath_lons_lats():
    swath_shape = (50, 10)
    lon_start, lon_stop, lat_start, lat_stop = (3.0, 12.0, 75.0, 26.0)
    lons = create_test_longitude(lon_start, lon_stop, swath_shape)
    lats = create_test_latitude(lat_start, lat_stop, swath_shape)
    return lons, lats


class TestSwathHashability:
    """Test geometry objects being hashable and other related uses."""

    @pytest.mark.parametrize(
        "swath_def_func1",
        [
            _gen_swath_def_numpy,
            _gen_swath_def_dask,
            _gen_swath_def_xarray_numpy,
            _gen_swath_def_xarray_dask,
            _gen_swath_def_numpy_small_noncontiguous,
        ])
    def test_swath_as_dict_keys(self, swath_def_func1, create_test_swath):
        from ..utils import assert_maximum_dask_computes
        swath_def1 = swath_def_func1(create_test_swath)
        swath_def2 = swath_def_func1(create_test_swath)

        with assert_maximum_dask_computes(0):
            assert hash(swath_def1) == hash(swath_def2)
            assert isinstance(hash(swath_def1), int)

            test_dict = {}
            test_dict[swath_def1] = 5
            assert test_dict[swath_def1] == 5
            assert test_dict[swath_def2] == 5
            assert test_dict.get(swath_def2) == 5
            test_dict[swath_def2] = 6
            assert test_dict[swath_def1] == 6
            assert test_dict[swath_def2] == 6

    def test_non_contiguous_swath_hash(self, create_test_swath):
        """Test swath hash."""
        swath_def = _gen_swath_def_numpy_small(create_test_swath)
        swath_def_subset = _gen_swath_def_numpy_small_noncontiguous(create_test_swath)
        assert hash(swath_def) != hash(swath_def_subset)


class TestSwathDefinition:
    """Test the SwathDefinition."""

    def test_swath(self, create_test_swath):
        """Test swath."""
        lons1 = np.fromfunction(lambda y, x: 3 + (10.0 / 100) * x, (5000, 100))
        lats1 = np.fromfunction(lambda y, x: 75 - (50.0 / 5000) * y, (5000, 100))

        swath_def = create_test_swath(lons1, lats1)
        lons2, lats2 = swath_def.get_lonlats()
        assert not (id(lons1) != id(lons2) or id(lats1) != id(lats2)), 'Caching of swath coordinates failed'

    def test_slice(self, create_test_swath):
        """Test that SwathDefinitions can be sliced."""
        lons1 = np.fromfunction(lambda y, x: 3 + (10.0 / 100) * x, (5000, 100))
        lats1 = np.fromfunction(
            lambda y, x: 75 - (50.0 / 5000) * y, (5000, 100))

        swath_def = create_test_swath(lons1, lats1)
        new_swath_def = swath_def[1000:4000, 20:40]
        assert new_swath_def.lons.shape == (3000, 20)
        assert new_swath_def.lats.shape == (3000, 20)

    def test_concat_1d(self, create_test_swath):
        """Test concatenating in 1d."""
        lons1 = np.array([1, 2, 3])
        lats1 = np.array([1, 2, 3])
        lons2 = np.array([4, 5, 6])
        lats2 = np.array([4, 5, 6])
        swath_def1 = create_test_swath(lons1, lats1)
        swath_def2 = create_test_swath(lons2, lats2)
        swath_def_concat = swath_def1.concatenate(swath_def2)
        expected = np.array([1, 2, 3, 4, 5, 6])
        np.testing.assert_allclose(swath_def_concat.lons, expected)
        np.testing.assert_allclose(swath_def_concat.lats, expected)

    def test_concat_2d(self, create_test_swath):
        """Test concatenating in 2d."""
        lons1 = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
        lats1 = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
        lons2 = np.array([[4, 5, 6], [6, 7, 8]])
        lats2 = np.array([[4, 5, 6], [6, 7, 8]])
        swath_def1 = create_test_swath(lons1, lats1)
        swath_def2 = create_test_swath(lons2, lats2)
        swath_def_concat = swath_def1.concatenate(swath_def2)
        expected = np.array(
            [[1, 2, 3], [3, 4, 5], [5, 6, 7], [4, 5, 6], [6, 7, 8]])
        np.testing.assert_allclose(swath_def_concat.lons, expected)
        np.testing.assert_allclose(swath_def_concat.lats, expected)

    def test_append_1d(self, create_test_swath):
        """Test appending in 1d."""
        lons1 = np.array([1, 2, 3])
        lats1 = np.array([1, 2, 3])
        lons2 = np.array([4, 5, 6])
        lats2 = np.array([4, 5, 6])
        swath_def1 = create_test_swath(lons1, lats1)
        swath_def2 = create_test_swath(lons2, lats2)
        swath_def1.append(swath_def2)
        expected = np.array([1, 2, 3, 4, 5, 6])
        np.testing.assert_allclose(swath_def1.lons, expected)
        np.testing.assert_allclose(swath_def1.lats, expected)

    def test_append_2d(self, create_test_swath):
        """Test appending in 2d."""
        lons1 = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
        lats1 = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
        lons2 = np.array([[4, 5, 6], [6, 7, 8]])
        lats2 = np.array([[4, 5, 6], [6, 7, 8]])
        swath_def1 = create_test_swath(lons1, lats1)
        swath_def2 = create_test_swath(lons2, lats2)
        swath_def1.append(swath_def2)
        expected = np.array(
            [[1, 2, 3], [3, 4, 5], [5, 6, 7], [4, 5, 6], [6, 7, 8]])
        np.testing.assert_allclose(swath_def1.lons, expected)
        np.testing.assert_allclose(swath_def1.lats, expected)

    def test_swath_equal(self, create_test_swath):
        """Test swath equality."""
        lons = np.array([1.2, 1.3, 1.4, 1.5])
        lats = np.array([65.9, 65.86, 65.82, 65.78])
        swath_def = create_test_swath(lons, lats)
        swath_def2 = create_test_swath(lons, lats)
        # Identical lons and lats
        assert not (swath_def != swath_def2), 'swath_defs are not equal as expected'
        # Identical objects
        assert not (swath_def != swath_def), 'swath_defs are not equal as expected'

        lons = np.array([1.2, 1.3, 1.4, 1.5])
        lats = np.array([65.9, 65.86, 65.82, 65.78])
        lons2 = np.array([1.2, 1.3, 1.4, 1.5])
        lats2 = np.array([65.9, 65.86, 65.82, 65.78])
        swath_def = create_test_swath(lons, lats)
        swath_def2 = create_test_swath(lons2, lats2)
        # different arrays, same values
        assert not (swath_def != swath_def2), 'swath_defs are not equal as expected'

        lons = np.array([1.2, 1.3, 1.4, np.nan])
        lats = np.array([65.9, 65.86, 65.82, np.nan])
        lons2 = np.array([1.2, 1.3, 1.4, np.nan])
        lats2 = np.array([65.9, 65.86, 65.82, np.nan])
        swath_def = create_test_swath(lons, lats)
        swath_def2 = create_test_swath(lons2, lats2)
        # different arrays, same values, with nans
        assert not (swath_def != swath_def2), 'swath_defs are not equal as expected'

        lons = da.from_array(np.array([1.2, 1.3, 1.4, np.nan]), chunks=2)
        lats = da.from_array(np.array([65.9, 65.86, 65.82, np.nan]), chunks=2)
        lons2 = da.from_array(np.array([1.2, 1.3, 1.4, np.nan]), chunks=2)
        lats2 = da.from_array(np.array([65.9, 65.86, 65.82, np.nan]), chunks=2)
        swath_def = create_test_swath(lons, lats)
        swath_def2 = create_test_swath(lons2, lats2)
        # different arrays, same values, with nans
        assert not (swath_def != swath_def2), 'swath_defs are not equal as expected'

        lons = xr.DataArray(np.array([1.2, 1.3, 1.4, np.nan]))
        lats = xr.DataArray(np.array([65.9, 65.86, 65.82, np.nan]))
        lons2 = xr.DataArray(np.array([1.2, 1.3, 1.4, np.nan]))
        lats2 = xr.DataArray(np.array([65.9, 65.86, 65.82, np.nan]))
        swath_def = create_test_swath(lons, lats)
        swath_def2 = create_test_swath(lons2, lats2)
        # different arrays, same values, with nans
        assert not (swath_def != swath_def2), 'swath_defs are not equal as expected'

    def test_swath_not_equal(self, create_test_swath):
        """Test swath inequality."""
        lats1 = np.array([65.9, 65.86, 65.82, 65.78])
        lons = np.array([1.2, 1.3, 1.4, 1.5])
        lats2 = np.array([65.91, 65.85, 65.80, 65.75])
        swath_def = create_test_swath(lons, lats1)
        swath_def2 = create_test_swath(lons, lats2)
        assert not (swath_def == swath_def2), 'swath_defs are not expected to be equal'

    def test_compute_omerc_params(self, create_test_swath):
        """Test omerc parameters computation."""
        lats = np.array([[85.23900604248047, 62.256004333496094, 35.58000183105469],
                         [80.84000396728516, 60.74200439453125, 34.08500289916992],
                         [67.07600402832031, 54.147003173828125, 30.547000885009766]]).T

        lons = np.array([[-90.67900085449219, -21.565000534057617, -21.525001525878906],
                         [79.11000061035156, 7.284000396728516, -5.107000350952148],
                         [81.26400756835938, 29.672000885009766, 10.260000228881836]]).T

        area = create_test_swath(lons, lats)
        proj_dict = {'lonc': -11.391744043133668, 'ellps': 'WGS84',
                     'proj': 'omerc', 'alpha': 9.185764390923012,
                     'gamma': 0, 'lat_0': -0.2821013754097188}
        assert_np_dict_allclose(area._compute_omerc_parameters('WGS84'),
                                proj_dict)
        lats = xr.DataArray(np.array([[85.23900604248047, 62.256004333496094, 35.58000183105469, np.nan],
                                      [80.84000396728516, 60.74200439453125, 34.08500289916992, np.nan],
                                      [67.07600402832031, 54.147003173828125, 30.547000885009766, np.nan]]).T,
                            dims=['y', 'x'])

        lons = xr.DataArray(np.array([[-90.67900085449219, -21.565000534057617, -21.525001525878906, np.nan],
                                      [79.11000061035156, 7.284000396728516, -5.107000350952148, np.nan],
                                      [81.26400756835938, 29.672000885009766, 10.260000228881836, np.nan]]).T)

        area = create_test_swath(lons, lats)
        proj_dict = {'lonc': -11.391744043133668, 'ellps': 'WGS84',
                     'proj': 'omerc', 'alpha': 9.185764390923012,
                     'gamma': 0, 'lat_0': -0.2821013754097188}
        assert_np_dict_allclose(area._compute_omerc_parameters('WGS84'),
                                proj_dict)

    def test_get_edge_lonlats(self, create_test_swath):
        """Test the `get_edge_lonlats` functionality."""
        lats = np.array([[85.23900604248047, 62.256004333496094, 35.58000183105469],
                         [80.84000396728516, 60.74200439453125, 34.08500289916992],
                         [67.07600402832031, 54.147003173828125, 30.547000885009766]]).T
        lons = np.array([[-90.67900085449219, -21.565000534057617, -21.525001525878906],
                         [79.11000061035156, 7.284000396728516, -5.107000350952148],
                         [81.26400756835938, 29.672000885009766, 10.260000228881836]]).T
        area = create_test_swath(lons, lats)
        lons, lats = area.get_edge_lonlats()
        np.testing.assert_allclose(lons, [-90.67900085, 79.11000061, 81.26400757,
                                          81.26400757, 29.67200089, 10.26000023,
                                          10.26000023, -5.10700035, -21.52500153,
                                          -21.52500153, -21.56500053, -90.67900085])
        np.testing.assert_allclose(lats, [85.23900604, 80.84000397, 67.07600403,
                                          67.07600403, 54.14700317, 30.54700089,
                                          30.54700089, 34.0850029, 35.58000183,
                                          35.58000183, 62.25600433, 85.23900604])

        lats = np.array([[80., 80., 80.],
                         [80., 90., 80],
                         [80., 80., 80.]]).T
        lons = np.array([[-45., 0., 45.],
                         [-90, 0., 90.],
                         [-135., -180., 135.]]).T
        area = create_test_swath(lons, lats)
        lons, lats = area.get_edge_lonlats()
        np.testing.assert_allclose(lons, [-45., -90., -135., -135., -180., 135.,
                                          135., 90., 45., 45., 0., -45.])
        np.testing.assert_allclose(lats, [80., 80., 80., 80., 80., 80., 80.,
                                          80., 80., 80., 80., 80.])

    def test_compute_optimal_bb(self, create_test_swath):
        """Test computing the bb area."""
        nplats = np.array([[85.23900604248047, 62.256004333496094, 35.58000183105469],
                           [80.84000396728516, 60.74200439453125, 34.08500289916992],
                           [67.07600402832031, 54.147003173828125, 30.547000885009766]]).T
        lats = xr.DataArray(nplats)
        nplons = np.array([[-90.67900085449219, -21.565000534057617, -21.525001525878906],
                           [79.11000061035156, 7.284000396728516, -5.107000350952148],
                           [81.26400756835938, 29.672000885009766, 10.260000228881836]]).T
        lons = xr.DataArray(nplons)

        area = create_test_swath(lons, lats)

        res = area.compute_optimal_bb_area({'proj': 'omerc', 'ellps': 'WGS84'})

        np.testing.assert_allclose(res.area_extent, [-2348379.728104, 3228086.496211,
                                                     2432121.058435, 10775774.254169])
        proj_dict = {'gamma': 0.0, 'lonc': -11.391744043133668,
                     'ellps': 'WGS84', 'proj': 'omerc',
                     'alpha': 9.185764390923012, 'lat_0': -0.2821013754097188}
        # pyproj2 adds some extra defaults
        proj_dict.update({'x_0': 0, 'y_0': 0, 'units': 'm',
                          'k': 1, 'gamma': 0,
                          'no_defs': None, 'type': 'crs'})
        assert res.crs == CRS.from_dict(proj_dict)
        assert res.shape == (6, 3)

        area = create_test_swath(nplons, nplats)
        res = area.compute_optimal_bb_area({'proj': 'omerc', 'ellps': 'WGS84'})

        np.testing.assert_allclose(res.area_extent, [-2348379.728104, 3228086.496211,
                                                     2432121.058435, 10775774.254169])
        proj_dict = {'gamma': 0.0, 'lonc': -11.391744043133668,
                     'ellps': 'WGS84', 'proj': 'omerc',
                     'alpha': 9.185764390923012, 'lat_0': -0.2821013754097188}
        # pyproj2 adds some extra defaults
        proj_dict.update({'x_0': 0, 'y_0': 0, 'units': 'm',
                          'k': 1, 'gamma': 0,
                          'no_defs': None, 'type': 'crs'})
        assert res.crs == CRS.from_dict(proj_dict)
        assert res.shape == (6, 3)

    def test_compute_optimal_bb_with_resolution(self, create_test_swath):
        """Test computing the bb area while passing in the resolution."""
        nplats = np.array([[85.23900604248047, 62.256004333496094, 35.58000183105469],
                           [80.84000396728516, 60.74200439453125, 34.08500289916992],
                           [67.07600402832031, 54.147003173828125, 30.547000885009766]]).T
        lats = xr.DataArray(nplats)
        nplons = np.array([[-90.67900085449219, -21.565000534057617, -21.525001525878906],
                           [79.11000061035156, 7.284000396728516, -5.107000350952148],
                           [81.26400756835938, 29.672000885009766, 10.260000228881836]]).T
        lons = xr.DataArray(nplons)

        area = create_test_swath(lons, lats)
        res1000 = area.compute_optimal_bb_area({'proj': 'omerc', 'ellps': 'WGS84'}, resolution=1000)
        res10000 = area.compute_optimal_bb_area({'proj': 'omerc', 'ellps': 'WGS84'}, resolution=10000)
        assert res1000.shape[0] // 10 == res10000.shape[0]
        assert res1000.shape[1] // 10 == res10000.shape[1]

    def test_aggregation(self, create_test_swath):
        """Test aggregation on SwathDefinitions."""
        window_size = 2
        resolution = 3
        lats = np.array([[0, 0, 0, 0], [1, 1, 1, 1.0]])
        lons = np.array([[178.5, 179.5, -179.5, -178.5], [178.5, 179.5, -179.5, -178.5]])
        xlats = xr.DataArray(da.from_array(lats, chunks=2), dims=['y', 'x'],
                             attrs={'resolution': resolution})
        xlons = xr.DataArray(da.from_array(lons, chunks=2), dims=['y', 'x'],
                             attrs={'resolution': resolution})
        sd = create_test_swath(xlons, xlats)
        res = sd.aggregate(y=window_size, x=window_size)
        np.testing.assert_allclose(res.lons, [[179, -179]])
        np.testing.assert_allclose(res.lats, [[0.5, 0.5]], atol=2e-5)
        assert res.lons.resolution == pytest.approx(window_size * resolution)
        assert res.lats.resolution == pytest.approx(window_size * resolution)

    def test_striding(self, create_test_swath):
        """Test striding."""
        lats = np.array([[0, 0, 0, 0], [1, 1, 1, 1.0]])
        lons = np.array([[178.5, 179.5, -179.5, -178.5], [178.5, 179.5, -179.5, -178.5]])
        xlats = xr.DataArray(da.from_array(lats, chunks=2), dims=['y', 'x'])
        xlons = xr.DataArray(da.from_array(lons, chunks=2), dims=['y', 'x'])
        sd = create_test_swath(xlons, xlats)
        res = sd[::2, ::2]
        np.testing.assert_allclose(res.lons, [[178.5, -179.5]])
        np.testing.assert_allclose(res.lats, [[0, 0]], atol=2e-5)

    def test_swath_def_geocentric_resolution(self, create_test_swath):
        """Test the SwathDefinition.geocentric_resolution method."""
        lats = np.array([[0, 0, 0, 0], [1, 1, 1, 1.0]])
        lons = np.array([[178.5, 179.5, -179.5, -178.5], [178.5, 179.5, -179.5, -178.5]])
        xlats = xr.DataArray(da.from_array(lats, chunks=2), dims=['y', 'x'])
        xlons = xr.DataArray(da.from_array(lons, chunks=2), dims=['y', 'x'])
        sd = create_test_swath(xlons, xlats)
        geo_res = sd.geocentric_resolution()
        # google says 1 degrees of longitude is about ~111.321km
        # so this seems good
        np.testing.assert_allclose(111301.237078, geo_res)

        # with a resolution attribute that is None
        xlons.attrs['resolution'] = None
        xlats.attrs['resolution'] = None
        sd = create_test_swath(xlons, xlats)
        geo_res = sd.geocentric_resolution()
        np.testing.assert_allclose(111301.237078, geo_res)

        # with a resolution attribute that is a number
        xlons.attrs['resolution'] = 111301.237078 / 2
        xlats.attrs['resolution'] = 111301.237078 / 2
        sd = create_test_swath(xlons, xlats)
        geo_res = sd.geocentric_resolution()
        np.testing.assert_allclose(111301.237078, geo_res)

    def test_swath_def_geocentric_resolution_xarray_dask(self, create_test_swath):
        lats = np.array([[0, 0, 0, 0], [1, 1, 1, 1.0]])
        lons = np.array([[178.5, 179.5, -179.5, -178.5], [178.5, 179.5, -179.5, -178.5]])
        xlats = xr.DataArray(da.from_array(lats.ravel(), chunks=2), dims=['y'])
        xlons = xr.DataArray(da.from_array(lons.ravel(), chunks=2), dims=['y'])
        sd = create_test_swath(xlons, xlats)
        with pytest.raises(RuntimeError):
            sd.geocentric_resolution()

    def test_crs_is_stored(self, create_test_swath):
        """Check that the CRS attribute is stored when passed."""
        lats = np.array([[0, 0, 0, 0], [1, 1, 1, 1.0]])
        lons = np.array([[178.5, 179.5, -179.5, -178.5], [178.5, 179.5, -179.5, -178.5]])

        expected_crs = CRS(proj="longlat", ellps="bessel")
        sd = create_test_swath(lons, lats, crs=expected_crs)
        assert sd.crs == expected_crs

    def test_crs_is_created_by_default(self, create_test_swath):
        """Check that the CRS attribute is set to a default."""
        lats = np.array([[0, 0, 0, 0], [1, 1, 1, 1.0]])
        lons = np.array([[178.5, 179.5, -179.5, -178.5], [178.5, 179.5, -179.5, -178.5]])

        expected_crs = CRS(proj="longlat", ellps="WGS84")
        sd = create_test_swath(lons, lats)
        assert sd.crs == expected_crs


def assert_np_dict_allclose(dict1, dict2):
    """Check allclose on dicts."""
    assert set(dict1.keys()) == set(dict2.keys())
    for key, val in dict1.items():
        try:
            np.testing.assert_allclose(val, dict2[key])
        except TypeError:
            assert val == dict2[key]


def test_future_swath_has_attrs():
    """Test that future SwathDefinition has attrs."""
    from pyresample.future.geometry import SwathDefinition
    lons, lats = _gen_swath_lons_lats()
    attrs = dict(meta="data")
    swath = SwathDefinition(lons, lats, attrs=attrs)
    assert swath.attrs == attrs


def test_future_swath_slice_has_attrs():
    """Test that future sliced SwathDefinition has attrs."""
    from pyresample.future.geometry import SwathDefinition
    lons, lats = _gen_swath_lons_lats()
    attrs = dict(meta="data")
    swath = SwathDefinition(lons, lats, attrs=attrs)[0:1, 0:1]
    assert swath.attrs == attrs


def test_future_swath_concat_has_attrs():
    """Test that future concatenated SwathDefinition has attrs."""
    from pyresample.future.geometry import SwathDefinition
    lons, lats = _gen_swath_lons_lats()
    attrs1 = dict(meta1="data")
    swath1 = SwathDefinition(lons, lats, attrs=attrs1)
    attrs2 = dict(meta2="data")
    swath2 = SwathDefinition(lons, lats, attrs=attrs2)
    swath = swath1.concatenate(swath2)
    assert swath.attrs == dict(meta1="data", meta2="data")


def test_future_swath_concat_fails_on_different_crs():
    """Test that future concatenated SwathDefinition must have the same crs."""
    from pyresample.future.geometry import SwathDefinition
    lons, lats = _gen_swath_lons_lats()
    swath1 = SwathDefinition(lons, lats, crs="mycrs")
    swath2 = SwathDefinition(lons, lats, crs="myothercrs")
    with pytest.raises(ValueError, match="Incompatible CRSs."):
        _ = swath1.concatenate(swath2)
