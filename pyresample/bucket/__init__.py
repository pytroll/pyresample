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

import logging
import math

import dask
import dask.array as da
import numpy as np
import xarray as xr
from pyproj import Proj

LOG = logging.getLogger(__name__)


def _sort_weights(statistic_method, weights):
    """Sort idxs and weights based on weights.

    By default the method for sorting is `'min'`.
    """
    order = np.argsort(weights)
    if statistic_method == 'max':
        return order[::-1]
    return order


def _get_bin_statistic(bins, idxs_sorted, weights_sorted):
    """Get the statistic of each bin."""
    (unique_bin, unique_idx) = _find_unique_bins_and_indices(bins, idxs_sorted)

    return _expand_bin_statistics(bins, unique_bin, unique_idx, weights_sorted)


def _find_unique_bins_and_indices(bins, idxs_sorted):
    """Get unique bins and corresponding indices."""
    # get where the `idxs_sorted` located in `bins`
    binned_values = np.digitize(idxs_sorted, bins, right=True)

    # mask value outside of bin
    binned_values_masked = np.ma.masked_array(binned_values,
                                              (idxs_sorted < bins.min()) | (idxs_sorted > bins.max()))

    # get the first index and value in each bin
    return np.unique(binned_values_masked, return_index=True)


def _expand_bin_statistics(bins, unique_bin, unique_idx, weights_sorted):
    """Expand bin statistics to cover all bins."""
    # create the full index array
    weight_idx = np.full(len(bins), -1, dtype=np.int32)

    # assign the valid index to array
    weight_idx[unique_bin[~unique_bin.mask].data] = unique_idx[~unique_bin.mask]

    return weights_sorted[weight_idx]  # last value of weights_sorted always nan


@dask.delayed(pure=True)
def _get_statistics(statistic_method, data, idxs, out_shape):
    """Help method to get bin max/min in a dask delayed manner."""
    (idxs_sorted, data_sorted) = _get_sorted_indices_and_data(statistic_method, data, idxs)
    out_size = math.prod(out_shape)

    bins = np.linspace(0, out_size - 1, out_size, dtype=np.int64)

    return _get_bin_statistic(bins, idxs_sorted, data_sorted).reshape(out_shape)


def _get_sorted_indices_and_data(statistic_method, data, idxs):
    # sort idxs and data
    order = _sort_weights(statistic_method, data)
    idxs_sorted = idxs[order]
    data_sorted = np.append(data[order], np.nan)
    return (idxs_sorted, data_sorted)


class BucketResampler(object):
    """Bucket resampler.

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

    >>> resampler = BucketResampler(target_area, lons, lats)

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
                               meta=np.array((), dtype=lons.dtype),
                               dtype=lons.dtype,
                               new_axis=0, chunks=(2,) + lons.chunks)
        proj_x = result[0, :]
        proj_y = result[1, :]

        # Calculate array indices. Orient so that 0-meridian is pointing down.
        adef = self.target_area
        x_res, y_res = adef.resolution
        x_idxs = da.floor((proj_x - adef.area_extent[0]) / x_res).astype(np.int64)
        y_idxs = da.floor((adef.area_extent[3] - proj_y) / y_res).astype(np.int64)

        # Get valid index locations
        mask = (x_idxs >= 0) & (x_idxs < adef.width) & (y_idxs >= 0) & (y_idxs < adef.height)
        self.y_idxs = da.where(mask, y_idxs, -1)
        self.x_idxs = da.where(mask, x_idxs, -1)

        # Convert X- and Y-indices to raveled indexing
        target_shape = self.target_area.shape
        self.idxs = self.y_idxs * target_shape[1] + self.x_idxs

    def get_sum(self, data, fill_value=np.nan, skipna=True, empty_bucket_value=0):
        """Calculate sums for each bin with drop-in-a-bucket resampling.

        Parameters
        ----------
        data : Numpy or Dask array
            Data to be binned and summed.
        fill_value : float
            Fill value of the input data marking missing/invalid values.
            Default: np.nan
        skipna : boolean (optional)
            If True, skips missing values (as marked by NaN or `fill_value`) for the sum calculation
            (similarly to Numpy's `nansum`). Buckets containing only missing values are set to `empty_bucket_value`.
            If False, sets the bucket to fill_value if one or more missing values are present in the bucket
            (similarly to Numpy's `sum`).
            In both cases, empty buckets are set to `empty_bucket_value`.
            Default: True
        empty_bucket_value : float
            Set empty buckets to the given value. Empty buckets are considered as the buckets with value 0.
            Note that a bucket could become 0 as the result of a sum
            of positive and negative values. If the user needs to identify these zero-buckets reliably,
            `get_count()` can be used for this purpose.
            Default: 0

        Returns
        -------
        data : Numpy or Dask array
            Bin-wise sums in the target grid
        """
        LOG.info("Get sum of values in each location")

        if isinstance(data, xr.DataArray):
            data = data.data
        data = data.ravel()

        # Remove fill_values values from the data when used as weights
        invalid_mask = _get_invalid_mask(data, fill_value)
        weights = da.where(invalid_mask, 0, data)

        # Rechunk indices to match the data chunking
        if weights.chunks != self.idxs.chunks:
            self.idxs = da.rechunk(self.idxs, weights.chunks)

        # Calculate the sum of the data falling to each bin
        out_size = self.target_area.size
        sums, _ = da.histogram(self.idxs, bins=out_size, range=(0, out_size),
                               weights=weights, density=False)

        # TODO remove following line in favour of weights = data when dask histogram bug (issue #6935) is fixed
        sums = self._mask_bins_with_nan_if_not_skipna(skipna, data, out_size, sums, fill_value)

        if empty_bucket_value != 0:
            sums = da.where(sums == 0, empty_bucket_value, sums)

        return sums.reshape(self.target_area.shape)

    def _mask_bins_with_nan_if_not_skipna(self, skipna, data, out_size, statistic, fill_value):
        if not skipna:
            missing_val = _get_invalid_mask(data, fill_value)
            missing_val_bins, _ = da.histogram(self.idxs[missing_val], bins=out_size,
                                               range=(0, out_size))
            statistic = da.where(missing_val_bins > 0, fill_value, statistic)
        return statistic

    def _call_bin_statistic(self, statistic_method, data, fill_value=None, skipna=None):
        """Calculate statistics (min/max) for each bin with drop-in-a-bucket resampling."""
        if isinstance(data, xr.DataArray):
            data = data.data
        data = data.ravel()

        # Rechunk indices to match the data chunking
        if data.chunks != self.idxs.chunks:
            self.idxs = da.rechunk(self.idxs, data.chunks)

        out_shape = self.target_area.shape

        statistics = da.from_delayed(
            _get_statistics(statistic_method, data, self.idxs, out_shape),
            shape=out_shape,
            dtype=np.float64)

        return statistics

    def get_min(self, data, fill_value=np.nan, skipna=True):
        """Calculate minimums for each bin with drop-in-a-bucket resampling.

        Parameters
        ----------
        data : Numpy or Dask array
            Data to be binned.
        skipna : boolean (optional)
                If True, skips NaN values for the minimum calculation
                (similarly to Numpy's `nanmin`). Buckets containing only NaN are set to zero.
                If False, sets the bucket to NaN if one or more NaN values are present in the bucket
                (similarly to Numpy's `min`).
                In both cases, empty buckets are set to 0.
                Default: True

        Returns
        -------
        data : Numpy or Dask array
            Bin-wise minimums in the target grid
        """
        LOG.info("Get min of values in each location")
        return self._call_bin_statistic('min', data, fill_value, skipna)

    def get_max(self, data, fill_value=np.nan, skipna=True):
        """Calculate maximums for each bin with drop-in-a-bucket resampling.

        Parameters
        ----------
        data : Numpy or Dask array
            Data to be binned.
        skipna : boolean (optional)
                If True, skips NaN values for the maximum calculation
                (similarly to Numpy's `nanmax`). Buckets containing only NaN are set to zero.
                If False, sets the bucket to NaN if one or more NaN values are present in the bucket
                (similarly to Numpy's `max`).
                In both cases, empty buckets are set to 0.
                Default: True

        Returns
        -------
        data : Numpy or Dask array
            Bin-wise maximums in the target grid
        """
        LOG.info("Get max of values in each location")
        return self._call_bin_statistic('max', data, fill_value, skipna)

    def get_abs_max(self, data, fill_value=np.nan, skipna=True):
        """Calculate absolute maximums for each bin with drop-in-a-bucket resampling.

        Returns for each bin the original signed value which has the largest
        absolute value.

        .. warning::

            The slow :meth:`pandas.DataFrame.groupby` method is temporarily used here,
            as the `dask_groupby <https://github.com/dcherian/dask_groupby>`_ is still under development.

        Parameters
        ----------
        data : Numpy or Dask array
            Data to be binned.
        fill_value : number (optional)
            Value to use for empty buckets or all-NaN buckets.
        skipna : boolean (optional)
            If True, skips NaN values for the maximum calculation
            (similarly to Numpy's `nanmax`). Buckets containing only NaN are
            set to fill value.
            If False, sets the bucket to NaN if one or more NaN values are present in the bucket
            (similarly to Numpy's `max`).
            In both cases, empty buckets are set to fill value.
            Default: True

        Returns
        -------
        data : Numpy or Dask array
            Bin-wise maximums in the target grid
        """
        max_ = self.get_max(data, fill_value=fill_value, skipna=skipna)
        min_ = self.get_min(data, fill_value=fill_value, skipna=skipna)
        return self._get_abs_max_from_min_max(min_, max_)

    @staticmethod
    def _get_abs_max_from_min_max(min_, max_):
        """From array of min and array of max, get array of abs max."""
        return da.where(-min_ > max_, min_, max_)

    def get_count(self):
        """Count the number of occurrences for each bin using drop-in-a-bucket resampling.

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

    def get_average(self, data, fill_value=np.nan, skipna=True):
        """Calculate bin-averages using bucket resampling.

        Parameters
        ----------
        data : Numpy or Dask array
            Data to be binned and averaged.
        fill_value : float
            Fill value to mark missing/invalid values in the input data,
            as well as in the binned and averaged output data.
            Default: np.nan
        skipna : bool
            If True, skips missing values (as marked by NaN or `fill_value`) for the average calculation
            (similarly to Numpy's `nanmean`). Buckets containing only missing values are set to fill_value.
            If False, sets the bucket to fill_value if one or more missing values are present in the bucket
            (similarly to Numpy's `mean`).
            In both cases, empty buckets are set to NaN.
            Default: True

        Returns
        -------
        average : Dask array
            Binned and averaged data.
        """
        LOG.info("Get average value for each location")

        if not np.isnan(fill_value):
            data = da.where(data == fill_value, np.nan, data)

        sums = self.get_sum(data, skipna=skipna)
        counts = self.get_sum(np.logical_not(np.isnan(data)).astype(np.int64))

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


def _get_invalid_mask(data, fill_value):
    """Get a boolean array where values equal to fill_value in data are True."""
    if np.isnan(fill_value):
        return np.isnan(data)
    else:
        return data == fill_value


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
