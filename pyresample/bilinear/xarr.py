from xarray import DataArray
import dask.array as da
import dask

import numpy as np

from pyproj import Proj

from pykdtree.kdtree import KDTree
from pyresample import data_reduce, geometry, CHUNK_SIZE
from pyresample.kd_tree import XArrayResamplerNN


class XArrayResamplerBilinear(XArrayResamplerNN):

    def __init__(self,
                 source_geo_def,
                 target_geo_def,
                 radius_of_influence,
                 neighbours=32,
                 epsilon=0):
        """

        Parameters
        ----------
        source_geo_def : object
            Geometry definition of source
        target_geo_def : object
            Geometry definition of target
        radius_of_influence : float
            Cut off distance in meters
        neighbours : int, optional
            The number of neigbours to consider for each grid point.
            Default 1. Currently 1 is the only supported number.
        epsilon : float, optional
            Allowed uncertainty in meters. Increasing uncertainty
            reduces execution time
        """

        super(XArrayResamplerBilinear, self).__init__(source_geo_def,
                                                      target_geo_def,
                                                      radius_of_influence,
                                                      neighbours=neighbours,
                                                      epsilon=epsilon)
        self.bilinear_t = None
        self.bilinear_s = None

    def get_bil_info(self, mask=None):
        """Return neighbour info.

        Returns
        -------
        t__ : numpy array
            Vertical fractional distances from corner to the new points
        s__ : numpy array
            Horizontal fractional distances from corner to the new points
        input_idxs : numpy array
            Valid indices in the input data
        idx_arr : numpy array
            Mapping array from valid source points to target points
        """

        # Get neighbour info
        # valid_input_index, valid_output_index, index_array, distance_array
        vii, voi, iar, dar = self.get_neighbour_info(mask=mask)

        # Reduce index reference
        input_size = da.sum(vii)
        index_mask = iar == input_size
        iar = da.where(index_mask, 0, iar)
        shp = iar.shape
        iar = da.reshape(iar, (shp[0] * shp[1], shp[-1]))

        # Get output projection as pyproj object
        proj = Proj(self.target_geo_def.proj_str)

        # Get output x/y coordinates
        out_x, out_y = _get_output_xy_dask(self.target_geo_def, proj)

        # Get input x/y coordinates
        in_x, in_y = _get_input_xy_dask(self.source_geo_def, proj,
                                        self.valid_input_index, iar)

        # Get the four closest corner points around each output location
        pt_1, pt_2, pt_3, pt_4, iar = \
            _get_bounding_corners_dask(in_x, in_y, out_x, out_y,
                                       self.neighbours, iar)

        # Calculate vertical and horizontal fractional distances t and s
        t__, s__ = _get_ts_dask(pt_1, pt_2, pt_3, pt_4, out_x, out_y)

        shp = self.target_geo_def.shape
        self.bilinear_t = t__.reshape(shp)
        self.bilinear_s = s__.reshape(shp)

        self.valid_output_index = voi
        self.index_array = iar.reshape((shp[0], shp[1], 4))
        self.distance_array = dar

        return (self.bilinear_t, self.bilinear_s, self.valid_input_index,
                self.index_array)

    def get_sample_from_bil_info(self, data, fill_value=np.nan):
        """Get data using bilinear interpolation."""

        res = self.get_sample_from_neighbour_info(data, fill_value=fill_value)
        coords = res.coords

        try:
            p_1 = res[:, :, :, 0]
            p_2 = res[:, :, :, 1]
            p_3 = res[:, :, :, 2]
            p_4 = res[:, :, :, 3]
        except IndexError:
            p_1 = res[:, :, 0]
            p_2 = res[:, :, 1]
            p_3 = res[:, :, 2]
            p_4 = res[:, :, 3]

        s__, t__ = self.bilinear_s, self.bilinear_t

        res = (p_1 * (1 - s__) * (1 - t__) +
               p_2 * s__ * (1 - t__) +
               p_3 * (1 - s__) * t__ +
               p_4 * s__ * t__)

        epsilon = 1e-6
        data_min = da.nanmin(data) - epsilon
        data_max = da.nanmax(data) + epsilon

        idxs = (res > data_max) | (res < data_min)
        res = da.where(idxs, fill_value, res)
        res = DataArray(da.from_array(res, chunks=CHUNK_SIZE),
                        dims=data.dims, coords=coords)

        return res


def _get_output_xy_dask(target_geo_def, proj):
    """Get x/y coordinates of the target grid."""
    # Read output coordinates
    out_lons, out_lats = target_geo_def.get_lonlats_dask()

    # Mask invalid coordinates
    out_lons, out_lats = _mask_coordinates_dask(out_lons.ravel(),
                                                out_lats.ravel())

    # Convert coordinates to output projection x/y space
    out_x, out_y = proj(out_lons.compute(), out_lats.compute())

    return out_x, out_y


def _get_input_xy_dask(source_geo_def, proj, input_idxs, iar):
    """Get x/y coordinates for the input area and reduce the data."""
    in_lons, in_lats = source_geo_def.get_lonlats_dask()

    # Mask invalid values
    in_lons, in_lats = _mask_coordinates_dask(in_lons, in_lats)

    # Select valid locations
    # TODO: direct indexing w/o .compute() results in
    # "ValueError: object too deep for desired array
    # input_idxs = input_idxs.ravel()
    in_lons = in_lons.compute()
    in_lons = in_lons[input_idxs]
    in_lats = in_lats.compute()
    in_lats = in_lats[input_idxs]

    # Expand input coordinates for each output location
    iar = iar.compute()
    in_lons = in_lons[iar]
    in_lats = in_lats[iar]

    # Convert coordinates to output projection x/y space
    in_x, in_y = proj(in_lons, in_lats)

    return in_x, in_y


def _mask_coordinates_dask(lons, lats):
    """Mask invalid coordinate values"""
    idxs = ((lons < -180.) | (lons > 180.) |
            (lats < -90.) | (lats > 90.))
    lons = da.where(idxs, np.nan, lons)
    lats = da.where(idxs, np.nan, lats)

    return lons, lats


def _get_bounding_corners_dask(in_x, in_y, out_x, out_y, neighbours, iar):
    """Get four closest locations from (in_x, in_y) so that they form a
    bounding rectangle around the requested location given by (out_x,
    out_y).
    """

    # Find four closest pixels around the target location

    out_x_tile = np.repeat(out_x.reshape(out_x.size, 1), neighbours, 1)
    out_y_tile = np.repeat(out_y.reshape(out_y.size, 1), neighbours, 1)

    # Get differences in both directions
    x_diff = out_x_tile - in_x
    y_diff = out_y_tile - in_y

    stride = np.arange(x_diff.shape[0])

    # Upper left source pixel
    valid = (x_diff > 0) & (y_diff < 0)
    x_1, y_1, idx_1 = _get_corner_dask(stride, valid, in_x, in_y, iar)

    # Upper right source pixel
    valid = (x_diff < 0) & (y_diff < 0)
    x_2, y_2, idx_2 = _get_corner_dask(stride, valid, in_x, in_y, iar)

    # Lower left source pixel
    valid = (x_diff > 0) & (y_diff > 0)
    x_3, y_3, idx_3 = _get_corner_dask(stride, valid, in_x, in_y, iar)

    # Lower right source pixel
    valid = (x_diff < 0) & (y_diff > 0)
    x_4, y_4, idx_4 = _get_corner_dask(stride, valid, in_x, in_y, iar)

    # Combine sorted indices to iar
    iar = np.transpose(np.vstack((idx_1, idx_2, idx_3, idx_4)))

    return (np.transpose(np.vstack((x_1, y_1))),
            np.transpose(np.vstack((x_2, y_2))),
            np.transpose(np.vstack((x_3, y_3))),
            np.transpose(np.vstack((x_4, y_4))),
            iar)


def _get_corner_dask(stride, valid, in_x, in_y, iar):
    """Get closest set of coordinates from the *valid* locations"""
    # Find the closest valid pixels, if any
    idxs = np.argmax(valid, axis=1)
    # Check which of these were actually valid
    invalid = np.invert(np.max(valid, axis=1))

    # idxs = idxs.compute()
    iar = iar.compute()

    # Replace invalid points with np.nan
    x__ = in_x[stride, idxs]  # TODO: daskify
    x__ = np.where(invalid, np.nan, x__)
    y__ = in_y[stride, idxs]  # TODO: daskify
    y__ = np.where(invalid, np.nan, y__)

    idx = iar[stride, idxs]  # TODO: daskify

    return x__, y__, idx


def _get_ts_dask(pt_1, pt_2, pt_3, pt_4, out_x, out_y):
    """Calculate vertical and horizontal fractional distances t and s"""

    # General case, ie. where the the corners form an irregular rectangle
    t__, s__ = _get_ts_irregular_dask(pt_1, pt_2, pt_3, pt_4, out_y, out_x)

    # Cases where verticals are parallel
    idxs = da.isnan(t__) | da.isnan(s__)
    # Remove extra dimensions
    # idxs = da.ravel(idxs)

    if da.any(idxs):
        t_new, s_new = _get_ts_uprights_parallel_dask(pt_1, pt_2,
                                                      pt_3, pt_4,
                                                      out_y, out_x)

        t__ = da.where(idxs, t_new, t__)
        s__ = da.where(idxs, s_new, s__)

    # Cases where both verticals and horizontals are parallel
    idxs = da.isnan(t__) | da.isnan(s__)
    # Remove extra dimensions
    # idxs = da.ravel(idxs)
    if da.any(idxs):
        t_new, s_new = _get_ts_parallellogram_dask(pt_1, pt_2, pt_3,
                                                   out_y, out_x)
        t__ = da.where(idxs, t_new, t__)
        s__ = da.where(idxs, s_new, s__)

    idxs = (t__ < 0) | (t__ > 1) | (s__ < 0) | (s__ > 1)
    t__ = da.where(idxs, np.nan, t__)
    s__ = da.where(idxs, np.nan, s__)

    return t__, s__


def _get_ts_irregular_dask(pt_1, pt_2, pt_3, pt_4, out_y, out_x):
    """Get parameters for the case where none of the sides are parallel."""

    # Get parameters for the quadratic equation
    # TODO: check if needs daskifying
    a__, b__, c__ = _calc_abc_dask(pt_1, pt_2, pt_3, pt_4, out_y, out_x)

    # Get the valid roots from interval [0, 1]
    t__ = _solve_quadratic_dask(a__, b__, c__, min_val=0., max_val=1.)

    # Calculate parameter s
    s__ = _solve_another_fractional_distance_dask(t__, pt_1[:, 1], pt_3[:, 1],
                                                  pt_2[:, 1], pt_4[:, 1], out_y)

    return t__, s__


# Might not need daskifying
def _calc_abc_dask(pt_1, pt_2, pt_3, pt_4, out_y, out_x):
    """Calculate coefficients for quadratic equation for
    _get_ts_irregular() and _get_ts_uprights().  For _get_ts_uprights
    switch order of pt_2 and pt_3.
    """
    # Pairwise longitudal separations between reference points
    x_21 = pt_2[:, 0] - pt_1[:, 0]
    x_31 = pt_3[:, 0] - pt_1[:, 0]
    x_42 = pt_4[:, 0] - pt_2[:, 0]

    # Pairwise latitudal separations between reference points
    y_21 = pt_2[:, 1] - pt_1[:, 1]
    y_31 = pt_3[:, 1] - pt_1[:, 1]
    y_42 = pt_4[:, 1] - pt_2[:, 1]

    a__ = x_31 * y_42 - y_31 * x_42
    b__ = out_y * (x_42 - x_31) - out_x * (y_42 - y_31) + \
        x_31 * pt_2[:, 1] - y_31 * pt_2[:, 0] + \
        y_42 * pt_1[:, 0] - x_42 * pt_1[:, 1]
    c__ = out_y * x_21 - out_x * y_21 + pt_1[:, 0] * pt_2[:, 1] - \
        pt_2[:, 0] * pt_1[:, 1]

    return a__, b__, c__


def _solve_quadratic_dask(a__, b__, c__, min_val=0.0, max_val=1.0):
    """Solve quadratic equation and return the valid roots from interval
    [*min_val*, *max_val*]

    """

    discriminant = b__ * b__ - 4 * a__ * c__

    # Solve the quadratic polynomial
    x_1 = (-b__ + da.sqrt(discriminant)) / (2 * a__)
    x_2 = (-b__ - da.sqrt(discriminant)) / (2 * a__)

    # Find valid solutions, ie. 0 <= t <= 1
    idxs = (x_1 < min_val) | (x_1 > max_val)
    x__ = da.where(idxs, x_2, x_1)

    idxs = (x__ < min_val) | (x__ > max_val)
    x__ = da.where(idxs, np.nan, x__)

    return x__


def _solve_another_fractional_distance_dask(f__, y_1, y_2, y_3, y_4, out_y):
    """Solve parameter t__ from s__, or vice versa.  For solving s__,
    switch order of y_2 and y_3."""
    y_21 = y_2 - y_1
    y_43 = y_4 - y_3

    g__ = ((out_y - y_1 - y_21 * f__) /
           (y_3 + y_43 * f__ - y_1 - y_21 * f__))

    # Limit values to interval [0, 1]
    idxs = (g__ < 0) | (g__ > 1)
    g__ = da.where(idxs, np.nan, g__)

    return g__


def _get_ts_uprights_parallel_dask(pt_1, pt_2, pt_3, pt_4, out_y, out_x):
    """Get parameters for the case where uprights are parallel"""

    # Get parameters for the quadratic equation
    a__, b__, c__ = _calc_abc_dask(pt_1, pt_3, pt_2, pt_4, out_y, out_x)

    # Get the valid roots from interval [0, 1]
    s__ = _solve_quadratic_dask(a__, b__, c__, min_val=0., max_val=1.)

    # Calculate parameter t
    t__ = _solve_another_fractional_distance_dask(s__, pt_1[:, 1], pt_2[:, 1],
                                                  pt_3[:, 1], pt_4[:, 1], out_y)

    return t__, s__


def _get_ts_parallellogram_dask(pt_1, pt_2, pt_3, out_y, out_x):
    """Get parameters for the case where uprights are parallel"""

    # Pairwise longitudal separations between reference points
    x_21 = pt_2[:, 0] - pt_1[:, 0]
    x_31 = pt_3[:, 0] - pt_1[:, 0]

    # Pairwise latitudal separations between reference points
    y_21 = pt_2[:, 1] - pt_1[:, 1]
    y_31 = pt_3[:, 1] - pt_1[:, 1]

    t__ = (x_21 * (out_y - pt_1[:, 1]) - y_21 * (out_x - pt_1[:, 0])) / \
          (x_21 * y_31 - y_21 * x_31)
    idxs = (t__ < 0.) | (t__ > 1.)
    t__ = da.where(idxs, np.nan, t__)

    s__ = (out_x - pt_1[:, 0] + x_31 * t__) / x_21
    idxs = (s__ < 0.) | (s__ > 1.)
    s__ = da.where(idxs, np.nan, s__)

    return t__, s__
