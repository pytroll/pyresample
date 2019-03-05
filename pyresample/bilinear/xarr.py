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
                                                      neightbours=neighbours,
                                                      epsilon=epsilon)
        self.bilinear_t = None
        self.bilinear_s = None

    # def query_resample_kdtree(self)
    # can be used

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
                                       self.neighbours, iar)

        # Calculate vertical and horizontal fractional distances t and s
        t__, s__ = _get_ts_dask(pt_1, pt_2, pt_3, pt_4, out_x, out_y)
        self.bilinear_t, self.bilinear_s = t__, s__

        self.valid_output_index = voi
        self.index_array = iar
        self.distance_array = dar

        return (self.bilinear_t, self.bilinear_s, self.valid_input_index,
                self.index_array)

    def get_sample_from_bil_info(self, data, fill_value=np.nan):
        """Get data using bilinear interpolation."""

        res = self.get_sample_from_neighbour_info(data, fill_value=fill_value)

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
