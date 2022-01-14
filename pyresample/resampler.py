#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019-2021 Pyresample developers
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
"""Base resampler class made for subclassing."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from numbers import Number
from typing import Union

import numpy as np

from .slicer import Slicer, _enumerate_chunk_slices

try:
    import dask.array as da
    from dask.highlevelgraph import HighLevelGraph
except ImportError:
    da = None

try:
    import xarray as xr
except ImportError:
    xr = None

from pyresample.geometry import (
    AreaDefinition,
    CoordinateDefinition,
    IncompatibleAreas,
    SwathDefinition,
)

from .future.resamplers.resampler import hash_dict

logger = logging.getLogger(__name__)


class BaseResampler:
    """Base abstract resampler class."""

    def __init__(self,
                 source_geo_def: Union[SwathDefinition, AreaDefinition],
                 target_geo_def: Union[CoordinateDefinition, AreaDefinition],
                 ):
        """Initialize resampler with geolocation information.

        Args:
            source_geo_def:
                Geolocation definition for the data to be resampled
            target_geo_def:
                Geolocation definition for the area to resample data to.

        """
        self.source_geo_def = source_geo_def
        self.target_geo_def = target_geo_def

    def get_hash(self, source_geo_def=None, target_geo_def=None, **kwargs):
        """Get hash for the current resample with the given *kwargs*."""
        if source_geo_def is None:
            source_geo_def = self.source_geo_def
        if target_geo_def is None:
            target_geo_def = self.target_geo_def
        the_hash = source_geo_def.update_hash()
        target_geo_def.update_hash(the_hash)
        hash_dict(kwargs, the_hash)
        return the_hash.hexdigest()

    def precompute(self, **kwargs):
        """Do the precomputation.

        This is an optional step if the subclass wants to implement more
        complex features like caching or can share some calculations
        between multiple datasets to be processed.

        """
        return None

    def compute(self, data, **kwargs):
        """Do the actual resampling.

        This must be implemented by subclasses.

        """
        raise NotImplementedError

    def resample(self, data, cache_dir=None, mask_area=None, **kwargs):
        """Resample `data` by calling `precompute` and `compute` methods.

        Only certain resampling classes may use `cache_dir` and the `mask`
        provided when `mask_area` is True. The return value of calling the
        `precompute` method is passed as the `cache_id` keyword argument
        of the `compute` method, but may not be used directly for caching. It
        is up to the individual resampler subclasses to determine how this
        is used.

        Args:
            data (xarray.DataArray): Data to be resampled
            cache_dir (str): directory to cache precomputed results
                             (default False, optional)
            mask_area (bool): Mask geolocation data where data values are
                              invalid. This should be used when data values
                              may affect what neighbors are considered valid.

        Returns (xarray.DataArray): Data resampled to the target area

        """
        # default is to mask areas for SwathDefinitions
        if mask_area is None and isinstance(
                self.source_geo_def, SwathDefinition):
            mask_area = True

        if mask_area:
            if isinstance(self.source_geo_def, SwathDefinition):
                geo_dims = self.source_geo_def.lons.dims
            else:
                geo_dims = ('y', 'x')
            flat_dims = [dim for dim in data.dims if dim not in geo_dims]
            if np.issubdtype(data.dtype, np.integer):
                kwargs['mask'] = data == data.attrs.get('_FillValue', np.iinfo(data.dtype.type).max)
            else:
                kwargs['mask'] = data.isnull()
            kwargs['mask'] = kwargs['mask'].all(dim=flat_dims)

        cache_id = self.precompute(cache_dir=cache_dir, **kwargs)
        return self.compute(data, cache_id=cache_id, **kwargs)

    def _create_cache_filename(self, cache_dir=None, prefix='',
                               fmt='.zarr', **kwargs):
        """Create filename for the cached resampling parameters."""
        cache_dir = cache_dir or '.'
        hash_str = self.get_hash(**kwargs)

        return os.path.join(cache_dir, prefix + hash_str + fmt)


def resample_blocks(src_area, dst_area, funk, *args, dst_arrays=(), chunks=None, dtype=None, name=None, **kwargs):
    """Resample blockwise.

    Args:
        src_area: a source geo definition
        dst_area:
        funk: a function to use. If func has a block_info keyword argument, the chunk info is passed, as in map_blocks
        args: data to use
        dst_arrays: arrays to use that are already in dst_area space. If the array has more than 2 dimensions,
            the last two are expected to be y, x.
        chunks: Has to be provided
        dtype: Has to be provided
        kwargs:

    """
    if args:
        data = args[0]
    else:
        data = None
    if dst_area == src_area:
        return data
    from dask.base import tokenize
    from dask.utils import apply, funcname, has_keyword

    name = f"{name or funcname(funk)}-{tokenize(funk, src_area, dst_area, dst_arrays, dtype, chunks, *args, **kwargs)}"
    dask_graph = dict()
    dependencies = []

    for target_block_info, target_geo_def in _enumerate_dst_area_chunks(dst_area, chunks):
        position = target_block_info["chunk-location"]
        output_shape = target_block_info["shape"]
        try:
            smaller_data, source_geo_def, source_block_info = crop_data_around_area(src_area, data, target_geo_def)
        except IncompatibleAreas:  # no relevant data matching
            if np.issubdtype(dtype, np.integer):
                fill_value = np.iinfo(dtype).min
            else:
                fill_value = np.nan
            task = [np.full, target_block_info["chunk-shape"], fill_value]
        else:
            args = [source_geo_def, target_geo_def]
            if data is not None:
                args.append((smaller_data.name, 0, 0))
                dependencies.append(smaller_data)

            for dst_array in dst_arrays:
                dst_position = [0] * (dst_array.ndim - 2) + list(position)
                args.append((dst_array.name, *dst_position))
                dependencies.append(dst_array)
            funk_kwargs = kwargs.copy()
            if has_keyword(funk, "block_info"):
                funk_kwargs["block_info"] = {0: source_block_info,
                                             None: target_block_info}
            task = [apply, funk, args, funk_kwargs]

        dask_graph[(name, *position)] = tuple(task)

    dask_graph = HighLevelGraph.from_collections(name, dask_graph, dependencies=dependencies)
    return da.Array(dask_graph, name, chunks=chunks, dtype=dtype, shape=output_shape)


@lru_cache(None)
def crop_source_area(source_geo_def, target_geo_def):
    """Crop a source area around a provided a target area."""
    slicer = Slicer(source_geo_def, target_geo_def)
    x_slice, y_slice = slicer.get_slices()
    small_source_geo_def = source_geo_def[y_slice, x_slice]
    if isinstance(small_source_geo_def, SwathDefinition):
        small_source_geo_def.lons.data = small_source_geo_def.lons.data.rechunk((-1, -1))
        small_source_geo_def.lats.data = small_source_geo_def.lats.data.rechunk((-1, -1))
    return small_source_geo_def, x_slice, y_slice


def crop_data_around_area(source_geo_def, data, target_geo_def):
    """Crop the data around the provided area."""
    small_source_geo_def, x_slice, y_slice = crop_source_area(source_geo_def, target_geo_def)
    if data is not None:
        smaller_data = data[..., y_slice, x_slice].rechunk(data.chunks[:-2] + (-1, -1))
    else:
        smaller_data = None
    block_info = {"shape": source_geo_def.shape, "array-location": (y_slice, x_slice)}
    return smaller_data, small_source_geo_def, block_info


def _enumerate_dst_area_chunks(dst_area, chunks):
    """Enumerate the chunks in function of the dst_area."""
    rest_shape = []
    if not isinstance(chunks, Number) and len(chunks) > len(dst_area.shape):
        rest_chunks = chunks[:-len(dst_area.shape)]
        for elt in rest_chunks:
            try:
                rest_shape.append(sum(elt))
            except TypeError:
                rest_shape.append(elt)
    output_shape = tuple(rest_shape) + dst_area.shape
    dst_chunks = da.core.normalize_chunks(chunks, output_shape)

    for position, slices in _enumerate_chunk_slices(dst_chunks):
        chunk_shape = tuple(chunk[pos] for pos, chunk in zip(position, dst_chunks))
        target_geo_def = dst_area[slices[-2:]]
        block_info = {"shape": output_shape,
                      "num-chunks": [len(chunks) for chunks in dst_chunks],
                      "chunk-location": position,
                      "array-location": slices,
                      "chunk-shape": chunk_shape,
                      }
        yield block_info, target_geo_def
