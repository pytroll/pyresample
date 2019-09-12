"""XArray version of bilinear interpolation."""

import warnings

try:
    from xarray import DataArray
    import dask.array as da
except ImportError:
    DataArray = None
    da = None

import numpy as np

from pyresample._spatial_mp import Proj

from pykdtree.kdtree import KDTree
from pyresample import data_reduce, geometry, CHUNK_SIZE

CACHE_INDICES = ['bilinear_s',
                 'bilinear_t',
                 'slices_x',
                 'slices_y',
                 'mask_slices',
                 'out_coords_x',
                 'out_coords_y']


class XArrayResamplerBilinear(object):
    """Bilinear interpolation using XArray."""

    def __init__(self,
                 source_geo_def,
                 target_geo_def,
                 radius_of_influence,
                 neighbours=32,
                 epsilon=0,
                 reduce_data=True):
        """
        Initialize resampler.

        Parameters
        ----------
        source_geo_def : object
            Geometry definition of source
        target_geo_def : object
            Geometry definition of target
        radius_of_influence : float
            Cut off distance in meters
        neighbours : int, optional
            The number of neigbours to consider for each grid point
        epsilon : float, optional
            Allowed uncertainty in meters. Increasing uncertainty
            reduces execution time
        reduce_data : bool, optional
            Perform initial coarse reduction of source dataset in order
            to reduce execution time

        """
        if da is None:
            raise ImportError("Missing 'xarray' and 'dask' dependencies")

        self.valid_input_index = None
        self.valid_output_index = None
        self.index_array = None
        self.distance_array = None
        self.bilinear_t = None
        self.bilinear_s = None
        self.slices_x = None
        self.slices_y = None
        self.slices = {'x': self.slices_x, 'y': self.slices_y}
        self.mask_slices = None
        self.out_coords_x = None
        self.out_coords_y = None
        self.out_coords = {'x': self.out_coords_x, 'y': self.out_coords_y}
        self.neighbours = neighbours
        self.epsilon = epsilon
        self.reduce_data = reduce_data
        self.source_geo_def = source_geo_def
        self.target_geo_def = target_geo_def
        self.radius_of_influence = radius_of_influence

    def get_bil_info(self):
        """Return neighbour info.

        Returns
        -------
        t__ : numpy array
            Vertical fractional distances from corner to the new points
        s__ : numpy array
            Horizontal fractional distances from corner to the new points
        valid_input_index : numpy array
            Valid indices in the input data
        index_array : numpy array
            Mapping array from valid source points to target points

        """
        if self.source_geo_def.size < self.neighbours:
            warnings.warn('Searching for %s neighbours in %s data points' %
                          (self.neighbours, self.source_geo_def.size))

        # Create kd-tree
        valid_input_index, resample_kdtree = self._create_resample_kdtree()
        # This is a numpy array
        self.valid_input_index = valid_input_index

        if resample_kdtree.n == 0:
            # Handle if all input data is reduced away
            bilinear_t, bilinear_s, valid_input_index, index_array = \
                _create_empty_bil_info(self.source_geo_def,
                                       self.target_geo_def)
            self.bilinear_t = bilinear_t
            self.bilinear_s = bilinear_s
            self.valid_input_index = valid_input_index
            self.index_array = index_array

            return bilinear_t, bilinear_s, valid_input_index, index_array

        target_lons, target_lats = self.target_geo_def.get_lonlats()
        valid_output_idx = ((target_lons >= -180) & (target_lons <= 180) &
                            (target_lats <= 90) & (target_lats >= -90))

        index_array, distance_array = self._query_resample_kdtree(
            resample_kdtree, target_lons, target_lats, valid_output_idx)

        # Reduce index reference
        input_size = da.sum(self.valid_input_index)
        index_mask = index_array == input_size
        index_array = da.where(index_mask, 0, index_array)

        # Get output projection as pyproj object
        proj = Proj(self.target_geo_def.proj_str)

        # Get output x/y coordinates
        out_x, out_y = self.target_geo_def.get_proj_coords(chunks=CHUNK_SIZE)
        out_x = da.ravel(out_x)
        out_y = da.ravel(out_y)

        # Get input x/y coordinates
        in_x, in_y = _get_input_xy_dask(self.source_geo_def, proj,
                                        self.valid_input_index, index_array)

        # Get the four closest corner points around each output location
        pt_1, pt_2, pt_3, pt_4, index_array = \
            _get_bounding_corners_dask(in_x, in_y, out_x, out_y,
                                       self.neighbours, index_array)

        # Calculate vertical and horizontal fractional distances t and s
        t__, s__ = _get_ts_dask(pt_1, pt_2, pt_3, pt_4, out_x, out_y)
        self.bilinear_t, self.bilinear_s = t__, s__

        self.valid_output_index = valid_output_idx
        self.index_array = index_array
        self.distance_array = distance_array

        self._get_slices()

        return (self.bilinear_t, self.bilinear_s,
                self.slices, self.mask_slices,
                self.out_coords)

    def get_sample_from_bil_info(self, data, fill_value=None,
                                 output_shape=None):
        """Resample using pre-computed resampling LUTs."""
        del output_shape
        if fill_value is None:
            if np.issubdtype(data.dtype, np.integer):
                fill_value = 0
            else:
                fill_value = np.nan

        p_1, p_2, p_3, p_4 = self._slice_data(data, fill_value)
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
        res = da.where(np.isnan(res), fill_value, res)
        shp = self.target_geo_def.shape
        if data.ndim == 3:
            res = da.reshape(res, (res.shape[0], shp[0], shp[1]))
        else:
            res = da.reshape(res, (shp[0], shp[1]))

        # Add missing coordinates
        self._add_missing_coordinates(data)

        res = DataArray(res, dims=data.dims, coords=self.out_coords)

        return res

    def _compute_indices(self):
        for idx in CACHE_INDICES:
            var = getattr(self, idx)
            try:
                var = var.compute()
                setattr(self, idx, var)
            except AttributeError:
                continue

    def _add_missing_coordinates(self, data):
        if self.out_coords['x'] is None and self.out_coords_x is not None:
            self.out_coords['x'] = self.out_coords_x
            self.out_coords['y'] = self.out_coords_y
        for _, dim in enumerate(data.dims):
            if dim not in self.out_coords:
                try:
                    self.out_coords[dim] = data.coords[dim]
                except KeyError:
                    pass

    def _slice_data(self, data, fill_value):

        def _slicer(values, sl_x, sl_y, mask, fill_value):
            if values.ndim == 2:
                arr = values[(sl_y, sl_x)]
                arr[(mask, )] = fill_value
                p_1 = arr[:, 0]
                p_2 = arr[:, 1]
                p_3 = arr[:, 2]
                p_4 = arr[:, 3]
            elif values.ndim == 3:
                arr = values[(slice(None), sl_y, sl_x)]
                arr[(slice(None), mask)] = fill_value
                p_1 = arr[:, :, 0]
                p_2 = arr[:, :, 1]
                p_3 = arr[:, :, 2]
                p_4 = arr[:, :, 3]
            else:
                raise ValueError

            return p_1, p_2, p_3, p_4

        values = data.values
        sl_y = self.slices_y
        sl_x = self.slices_x
        mask = self.mask_slices

        return _slicer(values, sl_x, sl_y, mask, fill_value)

    def _get_slices(self):
        shp = self.source_geo_def.shape
        cols, lines = np.meshgrid(np.arange(shp[1]),
                                  np.arange(shp[0]))
        cols = np.ravel(cols)
        lines = np.ravel(lines)

        vii = self.valid_input_index
        ia_ = self.index_array

        # ia_ contains reduced (valid) indices of the source array, and has the
        # shape of the destination array
        rlines = lines[vii][ia_]
        rcols = cols[vii][ia_]

        try:
            coord_x, coord_y = self.target_geo_def.get_proj_vectors()
            self.out_coords['y'] = coord_y
            self.out_coords['x'] = coord_x
            self.out_coords_y = self.out_coords['y']
            self.out_coords_x = self.out_coords['x']
        except AttributeError:
            pass

        self.mask_slices = ia_ >= self.source_geo_def.size
        self.slices['y'] = rlines
        self.slices['x'] = rcols
        self.slices_y = self.slices['y']
        self.slices_x = self.slices['x']

    def _create_resample_kdtree(self):
        """Set up kd tree on input."""
        # Get input information
        valid_input_index, source_lons, source_lats = \
            _get_valid_input_index_dask(self.source_geo_def,
                                        self.target_geo_def,
                                        self.reduce_data,
                                        self.radius_of_influence)

        # FIXME: Is dask smart enough to only compute the pixels we end up
        #        using even with this complicated indexing
        input_coords = lonlat2xyz(source_lons, source_lats)
        valid_input_index = da.ravel(valid_input_index)
        input_coords = input_coords[valid_input_index, :]
        input_coords = input_coords.compute()
        # Build kd-tree on input
        input_coords = input_coords.astype(np.float)
        valid_input_index, input_coords = da.compute(valid_input_index,
                                                     input_coords)
        return valid_input_index, KDTree(input_coords)

    def _query_resample_kdtree(self,
                               resample_kdtree,
                               tlons,
                               tlats,
                               valid_oi,
                               reduce_data=True):
        """Query kd-tree on slice of target coordinates."""
        res = query_no_distance(tlons, tlats,
                                valid_oi, resample_kdtree,
                                self.neighbours, self.epsilon,
                                self.radius_of_influence)
        return res, None


def _get_input_xy_dask(source_geo_def, proj, valid_input_index, index_array):
    """Get x/y coordinates for the input area and reduce the data."""
    in_lons, in_lats = source_geo_def.get_lonlats(chunks=CHUNK_SIZE)

    # Mask invalid values
    in_lons, in_lats = _mask_coordinates_dask(in_lons, in_lats)

    # Select valid locations
    # TODO: direct indexing w/o .compute() results in
    # "ValueError: object too deep for desired array

    in_lons = da.ravel(in_lons)
    in_lons = in_lons.compute()
    in_lons = in_lons[valid_input_index]
    in_lats = da.ravel(in_lats)
    in_lats = in_lats.compute()
    in_lats = in_lats[valid_input_index]
    index_array = index_array.compute()

    # Expand input coordinates for each output location
    in_lons = in_lons[index_array]
    in_lats = in_lats[index_array]

    # Convert coordinates to output projection x/y space
    in_x, in_y = proj(in_lons, in_lats)

    return in_x, in_y


def _mask_coordinates_dask(lons, lats):
    """Mask invalid coordinate values."""
    idxs = ((lons < -180.) | (lons > 180.) |
            (lats < -90.) | (lats > 90.))
    lons = da.where(idxs, np.nan, lons)
    lats = da.where(idxs, np.nan, lats)

    return lons, lats


def _get_bounding_corners_dask(in_x, in_y, out_x, out_y, neighbours, index_array):
    """Get bounding corners.

    Get four closest locations from (in_x, in_y) so that they form a
    bounding rectangle around the requested location given by (out_x,
    out_y).

    """
    # Find four closest pixels around the target location

    # FIXME: how to daskify?
    # Tile output coordinates to same shape as neighbour info
    # Replacing with da.transpose and da.tile doesn't work
    out_x_tile = np.reshape(np.tile(out_x, neighbours),
                            (neighbours, out_x.size)).T
    out_y_tile = np.reshape(np.tile(out_y, neighbours),
                            (neighbours, out_y.size)).T

    # Get differences in both directions
    x_diff = out_x_tile - in_x
    y_diff = out_y_tile - in_y

    stride = np.arange(x_diff.shape[0])

    # Upper left source pixel
    valid = (x_diff > 0) & (y_diff < 0)
    x_1, y_1, idx_1 = _get_corner_dask(stride, valid, in_x, in_y, index_array)

    # Upper right source pixel
    valid = (x_diff < 0) & (y_diff < 0)
    x_2, y_2, idx_2 = _get_corner_dask(stride, valid, in_x, in_y, index_array)

    # Lower left source pixel
    valid = (x_diff > 0) & (y_diff > 0)
    x_3, y_3, idx_3 = _get_corner_dask(stride, valid, in_x, in_y, index_array)

    # Lower right source pixel
    valid = (x_diff < 0) & (y_diff > 0)
    x_4, y_4, idx_4 = _get_corner_dask(stride, valid, in_x, in_y, index_array)

    # Combine sorted indices to index_array
    index_array = np.transpose(np.vstack((idx_1, idx_2, idx_3, idx_4)))

    return (np.transpose(np.vstack((x_1, y_1))),
            np.transpose(np.vstack((x_2, y_2))),
            np.transpose(np.vstack((x_3, y_3))),
            np.transpose(np.vstack((x_4, y_4))),
            index_array)


def _get_corner_dask(stride, valid, in_x, in_y, index_array):
    """Get closest set of coordinates from the *valid* locations."""
    # Find the closest valid pixels, if any
    idxs = np.argmax(valid, axis=1)
    # Check which of these were actually valid
    invalid = np.invert(np.max(valid, axis=1))

    # idxs = idxs.compute()
    index_array = index_array.compute()

    # Replace invalid points with np.nan
    x__ = in_x[stride, idxs]  # TODO: daskify
    x__ = da.where(invalid, np.nan, x__)
    y__ = in_y[stride, idxs]  # TODO: daskify
    y__ = da.where(invalid, np.nan, y__)

    idx = index_array[stride, idxs]  # TODO: daskify

    return x__, y__, idx


def _get_ts_dask(pt_1, pt_2, pt_3, pt_4, out_x, out_y):
    """Calculate vertical and horizontal fractional distances t and s."""
    def invalid_to_nan(t__, s__):
        idxs = (t__ < 0) | (t__ > 1) | (s__ < 0) | (s__ > 1)
        t__ = da.where(idxs, np.nan, t__)
        s__ = da.where(idxs, np.nan, s__)
        return t__, s__

    # General case, ie. where the the corners form an irregular rectangle
    t__, s__ = _get_ts_irregular_dask(pt_1, pt_2, pt_3, pt_4, out_y, out_x)

    # Replace invalid values with NaNs
    t__, s__ = invalid_to_nan(t__, s__)

    # Cases where verticals are parallel
    idxs = da.isnan(t__) | da.isnan(s__)
    # Remove extra dimensions
    idxs = da.ravel(idxs)

    if da.any(idxs):
        t_new, s_new = _get_ts_uprights_parallel_dask(pt_1, pt_2,
                                                      pt_3, pt_4,
                                                      out_y, out_x)
        t__ = da.where(idxs, t_new, t__)
        s__ = da.where(idxs, s_new, s__)

    # Replace invalid values with NaNs
    t__, s__ = invalid_to_nan(t__, s__)

    # Cases where both verticals and horizontals are parallel
    idxs = da.isnan(t__) | da.isnan(s__)
    # Remove extra dimensions
    idxs = da.ravel(idxs)
    if da.any(idxs):
        t_new, s_new = _get_ts_parallellogram_dask(pt_1, pt_2, pt_3,
                                                   out_y, out_x)
        t__ = da.where(idxs, t_new, t__)
        s__ = da.where(idxs, s_new, s__)

    # Replace invalid values with NaNs
    t__, s__ = invalid_to_nan(t__, s__)

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
    """Calculate coefficients for quadratic equation.

    In this order of arguments used for _get_ts_irregular() and
    _get_ts_uprights().  For _get_ts_uprights switch order of pt_2 and
    pt_3.

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
    """Solve quadratic equation.

    Solve quadratic equation and return the valid roots from interval
    [*min_val*, *max_val*].

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
    """Solve parameter t__ from s__, or vice versa.

    For solving s__, switch order of y_2 and y_3.
    """
    y_21 = y_2 - y_1
    y_43 = y_4 - y_3

    g__ = ((out_y - y_1 - y_21 * f__) /
           (y_3 + y_43 * f__ - y_1 - y_21 * f__))

    # Limit values to interval [0, 1]
    idxs = (g__ < 0) | (g__ > 1)
    g__ = da.where(idxs, np.nan, g__)

    return g__


def _get_ts_uprights_parallel_dask(pt_1, pt_2, pt_3, pt_4, out_y, out_x):
    """Get parameters for the case where uprights are parallel."""
    # Get parameters for the quadratic equation
    a__, b__, c__ = _calc_abc_dask(pt_1, pt_3, pt_2, pt_4, out_y, out_x)

    # Get the valid roots from interval [0, 1]
    s__ = _solve_quadratic_dask(a__, b__, c__, min_val=0., max_val=1.)

    # Calculate parameter t
    t__ = _solve_another_fractional_distance_dask(s__, pt_1[:, 1], pt_2[:, 1],
                                                  pt_3[:, 1], pt_4[:, 1], out_y)

    return t__, s__


def _get_ts_parallellogram_dask(pt_1, pt_2, pt_3, out_y, out_x):
    """Get parameters for the case where uprights are parallel."""
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


def query_no_distance(target_lons, target_lats,
                      valid_output_index, kdtree, neighbours, epsilon, radius):
    """Query the kdtree. No distances are returned."""
    voi = valid_output_index
    voir = da.ravel(voi)
    target_lons_valid = da.ravel(target_lons)[voir]
    target_lats_valid = da.ravel(target_lats)[voir]

    coords = lonlat2xyz(target_lons_valid, target_lats_valid)
    distance_array, index_array = kdtree.query(
        coords.compute(),
        k=neighbours,
        eps=epsilon,
        distance_upper_bound=radius)

    return index_array


def _get_valid_input_index_dask(source_geo_def,
                                target_geo_def,
                                reduce_data,
                                radius_of_influence):
    """Find indices of reduced input data."""
    source_lons, source_lats = source_geo_def.get_lonlats(chunks=CHUNK_SIZE)
    source_lons = da.ravel(source_lons)
    source_lats = da.ravel(source_lats)

    if source_lons.size == 0 or source_lats.size == 0:
        raise ValueError('Cannot resample empty data set')
    elif source_lons.size != source_lats.size or \
            source_lons.shape != source_lats.shape:
        raise ValueError('Mismatch between lons and lats')

    # Remove illegal values
    valid_input_index = ((source_lons >= -180) & (source_lons <= 180) &
                         (source_lats <= 90) & (source_lats >= -90))

    if reduce_data:
        # Reduce dataset
        if (isinstance(source_geo_def, geometry.CoordinateDefinition) and
            isinstance(target_geo_def, (geometry.GridDefinition,
                                        geometry.AreaDefinition))) or \
           (isinstance(source_geo_def, (geometry.GridDefinition,
                                        geometry.AreaDefinition)) and
            isinstance(target_geo_def, (geometry.GridDefinition,
                                        geometry.AreaDefinition))):
            # Resampling from swath to grid or from grid to grid
            lonlat_boundary = target_geo_def.get_boundary_lonlats()

            # Combine reduced and legal values
            valid_input_index &= \
                data_reduce.get_valid_index_from_lonlat_boundaries(
                    lonlat_boundary[0],
                    lonlat_boundary[1],
                    source_lons, source_lats,
                    radius_of_influence)

    if (isinstance(valid_input_index, np.ma.core.MaskedArray)):
        # Make sure valid_input_index is not a masked array
        valid_input_index = valid_input_index.filled(False)

    return valid_input_index, source_lons, source_lats


def lonlat2xyz(lons, lats):
    """Convert geographic coordinates to cartesian 3D coordinates."""
    R = 6370997.0
    x_coords = R * da.cos(da.deg2rad(lats)) * da.cos(da.deg2rad(lons))
    y_coords = R * da.cos(da.deg2rad(lats)) * da.sin(da.deg2rad(lons))
    z_coords = R * da.sin(da.deg2rad(lats))

    return da.stack(
        (x_coords.ravel(), y_coords.ravel(), z_coords.ravel()), axis=-1)


def _create_empty_bil_info(source_geo_def, target_geo_def):
    """Create dummy info for empty result set."""
    valid_input_index = np.ones(source_geo_def.size, dtype=np.bool)
    index_array = np.ones((target_geo_def.size, 4), dtype=np.int32)
    bilinear_s = np.nan * np.zeros(target_geo_def.size)
    bilinear_t = np.nan * np.zeros(target_geo_def.size)

    return bilinear_t, bilinear_s, valid_input_index, index_array
