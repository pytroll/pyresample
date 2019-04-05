
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


class XArrayResamplerBilinear(object):

    def __init__(self,
                 source_geo_def,
                 target_geo_def,
                 radius_of_influence,
                 neighbours=32,
                 epsilon=0,
                 reduce_data=True,
                 nprocs=1,
                 segments=None):
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
            The number of neigbours to consider for each grid point
        epsilon : float, optional
            Allowed uncertainty in meters. Increasing uncertainty
            reduces execution time
        reduce_data : bool, optional
            Perform initial coarse reduction of source dataset in order
            to reduce execution time
        nprocs : int, optional
            Number of processor cores to be used
        segments : int or None
            Number of segments to use when resampling.
            If set to None an estimate will be calculated
        """
        if da is None:
            raise ImportError("Missing 'xarray' and 'dask' dependencies")

        self.valid_input_index = None
        self.valid_output_index = None
        self.index_array = None
        self.distance_array = None
        self.bilinear_t = None
        self.bilinear_s = None
        self.neighbours = neighbours
        self.epsilon = epsilon
        self.reduce_data = reduce_data
        self.nprocs = nprocs
        self.segments = segments
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
        input_idxs : numpy array
            Valid indices in the input data
        idx_arr : numpy array
            Mapping array from valid source points to target points

        """
        if self.source_geo_def.size < self.neighbours:
            warnings.warn('Searching for %s neighbours in %s data points' %
                          (self.neighbours, self.source_geo_def.size))

        # Create kd-tree
        valid_input_idx, resample_kdtree = self._create_resample_kdtree()
        # This is a numpy array
        self.valid_input_index = valid_input_idx

        if resample_kdtree.n == 0:
            # Handle if all input data is reduced away
            bilinear_t, bilinear_s, valid_input_index, index_array = \
                _create_empty_bil_info(self.source_geo_def,
                                       self.target_geo_def)
            self.bilinear_t = bilinear_t
            self.bilinear_s = bilinear_s
            self.valid_input_index = valid_input_idx
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
        out_x, out_y = _get_output_xy_dask(self.target_geo_def, proj)

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

        return (self.bilinear_t, self.bilinear_s, self.valid_input_index,
                self.index_array)

    def get_sample_from_bil_info(self, data, fill_value=np.nan,
                                 output_shape=None):
        if fill_value is None:
            fill_value = np.nan
        # FIXME: can be this made into a dask construct ?
        cols, lines = np.meshgrid(np.arange(data['x'].size),
                                  np.arange(data['y'].size))
        cols = da.ravel(cols)
        lines = da.ravel(lines)
        try:
            self.valid_input_index = self.valid_input_index.compute()
        except AttributeError:
            pass
        vii = self.valid_input_index.squeeze()
        try:
            self.index_array = self.index_array.compute()
        except AttributeError:
            pass

        # ia contains reduced (valid) indices of the source array, and has the
        # shape of the destination array
        ia = self.index_array
        rlines = lines[vii][ia]
        rcols = cols[vii][ia]

        slices = []
        mask_slices = []
        mask_2d_added = False
        coords = {}
        try:
            # FIXME: Use same chunk size as input data
            coord_x, coord_y = self.target_geo_def.get_proj_vectors_dask()
        except AttributeError:
            coord_x, coord_y = None, None

        for _, dim in enumerate(data.dims):
            if dim == 'y':
                slices.append(rlines)
                if not mask_2d_added:
                    mask_slices.append(ia >= self.target_geo_def.size)
                    mask_2d_added = True
                if coord_y is not None:
                    coords[dim] = coord_y
            elif dim == 'x':
                slices.append(rcols)
                if not mask_2d_added:
                    mask_slices.append(ia >= self.target_geo_def.size)
                    mask_2d_added = True
                if coord_x is not None:
                    coords[dim] = coord_x
            else:
                slices.append(slice(None))
                mask_slices.append(slice(None))
                try:
                    coords[dim] = data.coords[dim]
                except KeyError:
                    pass

        res = data.values[slices]
        res[mask_slices] = fill_value

        try:
            p_1 = res[:, :, 0]
            p_2 = res[:, :, 1]
            p_3 = res[:, :, 2]
            p_4 = res[:, :, 3]
        except IndexError:
            p_1 = res[:, 0]
            p_2 = res[:, 1]
            p_3 = res[:, 2]
            p_4 = res[:, 3]

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
        shp = self.target_geo_def.shape
        if data.ndim == 3:
            res = da.reshape(res, (res.shape[0], shp[0], shp[1]))
        else:
            res = da.reshape(res, (shp[0], shp[1]))
        res = DataArray(da.from_array(res, chunks=CHUNK_SIZE),
                        dims=data.dims, coords=coords)

        return res

    def _create_resample_kdtree(self):
        """Set up kd tree on input"""
        # Get input information
        valid_input_index, source_lons, source_lats = \
            _get_valid_input_index_dask(self.source_geo_def,
                                        self.target_geo_def,
                                        self.reduce_data,
                                        self.radius_of_influence,
                                        nprocs=self.nprocs)

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

#        res = da.map_blocks(query_no_distance, tlons, tlats,
#                            valid_oi, dtype=np.int, kdtree=resample_kdtree,
#                            neighbours=self.neighbours, epsilon=self.epsilon,
#                            radius=self.radius_of_influence)
        res = query_no_distance(tlons, tlats,
                                valid_oi, resample_kdtree,
                                self.neighbours, self.epsilon,
                                self.radius_of_influence)
        return res, None


def _get_fill_mask_value(data_dtype):
    """Returns the maximum value of dtype"""
    if issubclass(data_dtype.type, np.floating):
        fill_value = np.finfo(data_dtype.type).max
    elif issubclass(data_dtype.type, np.integer):
        fill_value = np.iinfo(data_dtype.type).max
    else:
        raise TypeError('Type %s is unsupported for masked fill values' %
                        data_dtype.type)
    return fill_value


def _get_output_xy_dask(target_geo_def, proj):
    """Get x/y coordinates of the target grid."""
    # Read output coordinates
    out_lons, out_lats = target_geo_def.get_lonlats_dask()

    # Mask invalid coordinates
    out_lons, out_lats = _mask_coordinates_dask(out_lons, out_lats)

    # Convert coordinates to output projection x/y space
    res = da.dstack(proj(out_lons.compute(), out_lats.compute()))
    # _run_proj(proj, out_lons, out_lats)
    #,
    #                    chunks=(out_lons.chunks[0], out_lons.chunks[1], 2),
    #                    new_axis=[2])
    out_x = da.ravel(res[:, :, 0])
    out_y = da.ravel(res[:, :, 1])

    return out_x, out_y


def _get_input_xy_dask(source_geo_def, proj, input_idxs, idx_ref):
    """Get x/y coordinates for the input area and reduce the data."""
    in_lons, in_lats = source_geo_def.get_lonlats_dask()

    # Mask invalid values
    in_lons, in_lats = _mask_coordinates_dask(in_lons, in_lats)

    # Select valid locations
    # TODO: direct indexing w/o .compute() results in
    # "ValueError: object too deep for desired array

    in_lons = da.ravel(in_lons)
    in_lons = in_lons.compute()
    in_lons = in_lons[input_idxs]
    in_lats = da.ravel(in_lats)
    in_lats = in_lats.compute()
    in_lats = in_lats[input_idxs]

    # Expand input coordinates for each output location
    # in_lons = in_lons.compute()
    in_lons = in_lons[idx_ref]
    # in_lats = in_lats.compute()
    in_lats = in_lats[idx_ref]

    # Convert coordinates to output projection x/y space
    in_x, in_y = proj(in_lons, in_lats)

    return in_x, in_y


def _run_proj(proj, lons, lats):
    return da.dstack(proj(lons, lats))


def _mask_coordinates_dask(lons, lats):
    """Mask invalid coordinate values"""
    # lons = da.ravel(lons)
    # lats = da.ravel(lats)
    idxs = ((lons < -180.) | (lons > 180.) |
            (lats < -90.) | (lats > 90.))
    lons = da.where(idxs, np.nan, lons)
    lats = da.where(idxs, np.nan, lats)

    return lons, lats


def _get_bounding_corners_dask(in_x, in_y, out_x, out_y, neighbours, idx_ref):
    """Get four closest locations from (in_x, in_y) so that they form a
    bounding rectangle around the requested location given by (out_x,
    out_y).
    """

    # Find four closest pixels around the target location

    # FIXME: how to daskify?
    # Tile output coordinates to same shape as neighbour info
    # Replacing with da.transpose and da.tile doesn't work
    out_x_tile = np.transpose(np.tile(out_x, (neighbours, 1)))
    out_y_tile = np.transpose(np.tile(out_y, (neighbours, 1)))

    # Get differences in both directions
    x_diff = out_x_tile - in_x
    y_diff = out_y_tile - in_y

    stride = np.arange(x_diff.shape[0])

    # Upper left source pixel
    valid = (x_diff > 0) & (y_diff < 0)
    x_1, y_1, idx_1 = _get_corner_dask(stride, valid, in_x, in_y, idx_ref)

    # Upper right source pixel
    valid = (x_diff < 0) & (y_diff < 0)
    x_2, y_2, idx_2 = _get_corner_dask(stride, valid, in_x, in_y, idx_ref)

    # Lower left source pixel
    valid = (x_diff > 0) & (y_diff > 0)
    x_3, y_3, idx_3 = _get_corner_dask(stride, valid, in_x, in_y, idx_ref)

    # Lower right source pixel
    valid = (x_diff < 0) & (y_diff > 0)
    x_4, y_4, idx_4 = _get_corner_dask(stride, valid, in_x, in_y, idx_ref)

    # Combine sorted indices to idx_ref
    idx_ref = np.transpose(np.vstack((idx_1, idx_2, idx_3, idx_4)))

    return (np.transpose(np.vstack((x_1, y_1))),
            np.transpose(np.vstack((x_2, y_2))),
            np.transpose(np.vstack((x_3, y_3))),
            np.transpose(np.vstack((x_4, y_4))),
            idx_ref)


def _get_corner_dask(stride, valid, in_x, in_y, idx_ref):
    """Get closest set of coordinates from the *valid* locations"""
    # Find the closest valid pixels, if any
    idxs = np.argmax(valid, axis=1)
    # Check which of these were actually valid
    invalid = np.invert(np.max(valid, axis=1))

    # idxs = idxs.compute()
    idx_ref = idx_ref.compute()

    # Replace invalid points with np.nan
    x__ = in_x[stride, idxs]  # TODO: daskify
    x__ = np.where(invalid, np.nan, x__)
    y__ = in_y[stride, idxs]  # TODO: daskify
    y__ = np.where(invalid, np.nan, y__)

    idx = idx_ref[stride, idxs]  # TODO: daskify

    return x__, y__, idx


def _get_ts_dask(pt_1, pt_2, pt_3, pt_4, out_x, out_y):
    """Calculate vertical and horizontal fractional distances t and s"""

    # General case, ie. where the the corners form an irregular rectangle
    t__, s__ = _get_ts_irregular_dask(pt_1, pt_2, pt_3, pt_4, out_y, out_x)

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

    # Cases where both verticals and horizontals are parallel
    idxs = da.isnan(t__) | da.isnan(s__)
    # Remove extra dimensions
    idxs = da.ravel(idxs)
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


def _check_data_shape_dask(data, input_idxs):
    """Check data shape and adjust if necessary."""
    # Handle multiple datasets
    if data.ndim > 2 and data.shape[0] * data.shape[1] == input_idxs.shape[0]:
        data = da.reshape(data, data.shape[0] * data.shape[1], data.shape[2])
    # Also ravel single dataset
    elif data.shape[0] != input_idxs.size:
        data = da.ravel(data)

    # Ensure two dimensions
    if data.ndim == 1:
        data = da.reshape(data, (data.size, 1))

    return data


def query_no_distance(target_lons, target_lats,
                      valid_output_index, kdtree, neighbours, epsilon, radius):
    """Query the kdtree. No distances are returned."""
    voi = valid_output_index
    shape = voi.shape
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
                                radius_of_influence,
                                nprocs=1):
    """Find indices of reduced inputput data"""

    source_lons, source_lats = source_geo_def.get_lonlats_dask()
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

    R = 6370997.0
    x_coords = R * da.cos(da.deg2rad(lats)) * da.cos(da.deg2rad(lons))
    y_coords = R * da.cos(da.deg2rad(lats)) * da.sin(da.deg2rad(lons))
    z_coords = R * da.sin(da.deg2rad(lats))

    return da.stack(
        (x_coords.ravel(), y_coords.ravel(), z_coords.ravel()), axis=-1)


def _create_empty_bil_info(source_geo_def, target_geo_def):
    """Creates dummy info for empty result set"""

    valid_input_index = np.ones(source_geo_def.size, dtype=np.bool)
    index_array = np.ones((target_geo_def.size, 4), dtype=np.int32)
    bilinear_s = np.nan * np.zeros(target_geo_def.size)
    bilinear_t = np.nan * np.zeros(target_geo_def.size)

    return bilinear_t, bilinear_s, valid_input_index, index_array
