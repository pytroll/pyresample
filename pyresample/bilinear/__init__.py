#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017-2020 Pyresample developers.
#
# This file is part of Pyresample
#
# Author(s):
#
#   Panu Lahtinen <panu.lahtinen@fmi.fi>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Code for resampling using bilinear algorithm for irregular grids.

The algorithm is taken from

http://www.ahinson.com/algorithms_general/Sections/InterpolationRegression/InterpolationIrregularBilinear.pdf
"""
# file deepcode ignore W0611: sub-package imports

import warnings

from ._numpy_resampler import (  # noqa: F401
    NumpyBilinearResampler,
    NumpyResamplerBilinear,
    get_bil_info,
    get_sample_from_bil_info,
    resample_bilinear,
)

try:
    from .xarr import CACHE_INDICES, XArrayBilinearResampler, XArrayResamplerBilinear  # noqa: F401
except ImportError:
    warnings.warn("XArray, dask, and/or zarr not found, XArrayBilinearResampler won't be available.", stacklevel=2)
    XArrayBilinearResampler = None  # type: ignore
    CACHE_INDICES = None  # type: ignore
    XArrayResamplerBilinear = None  # type: ignore
