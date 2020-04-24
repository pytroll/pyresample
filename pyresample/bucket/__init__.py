# pyresample, Resampling of remote sensing image data in python
#
# Copyright (C) 2019  Pyresample developers
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Code for resampling using bucket resampling."""

import dask.array as da
import xarray as xr
import numpy as np
import logging
from pyresample._spatial_mp import Proj

LOG = logging.getLogger(__name__)


class BucketResampler(object):

    """Class for bucket resampling.

    Bucket resampling is useful for calculating averages and hit-counts
    when aggregating data to coarser scale grids.

    Below are examples how to use the resampler.

    Read data using Satpy.  The resampling can also be done (apart from
    fractions) directly from Satpy, but this demonstrates the direct
    low-level usage.

    >>> from pyresample.bucket import BucketResampler
    >>> from satpy import Scene
    >>> from satpy.resample import get_area_def
    >>> fname = "hrpt_noaa19_20170519_1214_42635.l1b"
    >>> glbl = Scene(filenames=[fname])
    >>> glbl.load(['4'])
    >>> data = glbl['4']
    >>> lons, lats = data.area.get_lonlats()
    >>> target_area = get_area_def('euro4')

    Initialize the resampler

    >>> resampler = BucketResampler(adef, lons, lats)

    Calculate the sum of all the data in each grid location:

    >>> sums = resampler.get_sum(data)

    Calculate how many values were collected at each grid location:

    >>> counts = resampler.get_count()

    The average can be calculated from the above two results, or directly
    using the helper method:

    >>> average = resampler.get_average(data)

    Calculate fractions of occurrences of different values in each grid
    location.  The data needs to be categorical (in integers), so
    we'll create some categorical data from the brightness temperature
    data that were read earlier.  The data are returned in a
    dictionary with the categories as keys.

    >>> data = da.where(data > 250, 1, 0)
    >>> fractions = resampler.get_fractions(data, categories=[0, 1])
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(fractions[0]); plt.show()
    """

    def __init__(self, target_area, source_lons, source_lats):

        self.target_area = target_area
        self.source_lons = source_lons
        self.source_lats = source_lats
        self.prj = Proj(self.target_area.proj_dict)
        self.x_idxs = None
        self.y_idxs = None
        self.idxs = None
        self._get_indices()
        self.counts = None

    def _get_proj_coordinates(self, lons, lats):
        """Calculate projection coordinates.

        Parameters
        ----------
        lons : Numpy or Dask array
            Longitude coordinates
        lats : Numpy or Dask array
            Latitude coordinates
        """
        proj_x, proj_y = self.prj(lons, lats)
        return np.stack((proj_x, proj_y))

    def _get_indices(self):
        """Calculate projection indices.

        Returns
        -------
        x_idxs : Dask array
            X indices of the target grid where the data are put
        y_idxs : Dask array
            Y indices of the target grid where the data are put
        """
        LOG.info("Determine bucket resampling indices")

        # Transform source lons/lats to target projection coordinates x/y
        lons = self.source_lons.ravel()
        lats = self.source_lats.ravel()
        result = da.map_blocks(self._get_proj_coordinates, lons, lats,
                               new_axis=0, chunks=(2,) + lons.chunks)
        proj_x = result[0, :]
        proj_y = result[1, :]

        # Calculate array indices. Orient so that 0-meridian is pointing down.
        adef = self.target_area
        x_res, y_res = adef.resolution
        x_idxs = da.floor((proj_x - adef.area_extent[0]) / x_res).astype(np.int)
        y_idxs = da.floor((adef.area_extent[3] - proj_y) / y_res).astype(np.int)

        # Get valid index locations
        mask = ((x_idxs >= 0) & (x_idxs < adef.width) &
                (y_idxs >= 0) & (y_idxs < adef.height))
        self.y_idxs = da.where(mask, y_idxs, -1)
        self.x_idxs = da.where(mask, x_idxs, -1)

        # Convert X- and Y-indices to raveled indexing
        target_shape = self.target_area.shape
        self.idxs = self.y_idxs * target_shape[1] + self.x_idxs

    def get_sum(self, data, mask_all_nan=False):
        """Calculate sums for each bin with drop-in-a-bucket resampling.

        Parameters
        ----------
        data : Numpy or Dask array
        mask_all_nan : boolean (optional)
            Mask bins that have only NaN results, default: False

        Returns
        -------
        data : Numpy or Dask array
            Bin-wise sums in the target grid
        """
        LOG.info("Get sum of values in each location")
        if isinstance(data, xr.DataArray):
            data = data.data
        data = data.ravel()
        # Remove NaN values from the data when used as weights
        weights = da.where(np.isnan(data), 0, data)

        # Rechunk indices to match the data chunking
        if weights.chunks != self.idxs.chunks:
            self.idxs = da.rechunk(self.idxs, weights.chunks)

        # Calculate the sum of the data falling to each bin
        out_size = self.target_area.size
        sums, _ = da.histogram(self.idxs, bins=out_size, range=(0, out_size),
                               weights=weights, density=False)

        if mask_all_nan:
            nans = np.isnan(data)
            nan_sums, _ = da.histogram(self.idxs[nans], bins=out_size,
                                       range=(0, out_size))
            counts = self.get_count().ravel()
            sums = da.where(nan_sums == counts, np.nan, sums)

        return sums.reshape(self.target_area.shape)

    def get_count(self):
        """Count the number of occurrences for each bin using drop-in-a-bucket
        resampling.

        Returns
        -------
        data : Dask array
            Bin-wise count of hits for each target grid location
        """
        LOG.info("Get number of values in each location")

        out_size = self.target_area.size

        # Calculate the sum of the data falling to each bin
        if self.counts is None:
            counts, _ = da.histogram(self.idxs, bins=out_size,
                                     range=(0, out_size))
            self.counts = counts.reshape(self.target_area.shape)

        return self.counts

    def get_average(self, data, fill_value=np.nan, mask_all_nan=False):
        """Calculate bin-averages using bucket resampling.

        Parameters
        ----------
        data : Numpy or Dask array
            Data to be binned and averaged
        fill_value : float
            Fill value to replace missing values.  Default: np.nan

        Returns
        -------
        average : Dask array
            Binned and averaged data.
        """
        LOG.info("Get average value for each location")

        sums = self.get_sum(data, mask_all_nan=mask_all_nan)
        counts = self.get_sum(np.logical_not(np.isnan(data)).astype(int),
                              mask_all_nan=False)

        average = sums / da.where(counts == 0, np.nan, counts)
        average = da.where(np.isnan(average), fill_value, average)

        return average

    def get_fractions(self, data, categories=None, fill_value=np.nan):
        """Get fraction of occurrences for each given categorical value.

        Parameters
        ----------
        data : Numpy or Dask array
            Categorical data to be processed
        categories : iterable or None
            One dimensional list of categories in the data, or None.  If None,
            categories are determined from the data by fully processing the
            data and finding the unique category values.
        fill_value : float
            Fill value to replace missing values.  Default: np.nan
        """
        if categories is None:
            LOG.warning("No categories given, need to compute the data.")
            # compute any dask arrays by converting to numpy
            categories = np.asarray(np.unique(data))
        try:
            num = categories.size
        except AttributeError:
            num = len(categories)
        LOG.info("Get fractions for %d categories", num)
        results = {}
        counts = self.get_count()
        counts = counts.astype(float)
        # Disable logging for calls to get_sum()
        LOG.disabled = True
        for cat in categories:
            cat_data = da.where(data == cat, 1.0, 0.0)

            sums = self.get_sum(cat_data)
            result = sums.astype(float) / counts
            result = da.where(counts == 0.0, fill_value, result)
            results[cat] = result
        # Re-enable logging
        LOG.disabled = False

        return results


def round_to_resolution(arr, resolution):
    """Round the values in *arr* to closest resolution element.

    Parameters
    ----------
    arr : list, tuple, Numpy or Dask array
        Array to be rounded
    resolution : float
        Resolution unit to which data are rounded

    Returns
    -------
    data : Numpy or Dask array
        Source data rounded to the closest resolution unit
    """
    if isinstance(arr, (list, tuple)):
        arr = np.array(arr)
    return resolution * np.round(arr / resolution)
