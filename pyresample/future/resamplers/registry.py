#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Pyresample developers
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
"""Registry of resampler classes."""

from __future__ import annotations

from typing import Optional, Type

from pyresample.future.cache import ResampleCache
from pyresample.future.resamplers.resampler import Resampler
from pyresample.kd_tree import XArrayResamplerNN

# TODO: Allow the resampler classes to register themselves
RESAMPLER_REGISTRY = {
    "nearest": XArrayResamplerNN,
}


def register_resampler(resampler_name: str, resampler_cls: Type[Resampler]) -> None:
    """Register :class:`~pyresample.future.resampler.Resampler` subclass for future use."""
    RESAMPLER_REGISTRY[resampler_name] = resampler_cls


def unregister_resampler(resampler_name: str) -> None:
    """Remove previously registered Resampler so it can't be used anymore."""
    del RESAMPLER_REGISTRY[resampler_name]


def list_resamplers() -> list[str, ...]:
    """Get sorted list of registered resamplers."""
    resampler_names = sorted(RESAMPLER_REGISTRY.keys())
    return resampler_names


def create_resampler(
        src_geom,
        dst_geom,
        resampler: str = None,
        cache: Optional[ResampleCache] = None,
        **kwargs
) -> Resampler:
    """Create instance of a :class:`~pyresample.future.resampler.Resampler` with the provided arguments.

    Args:
        src_geom:
            Geometry object defining the source geographic region that input
            data lies on and will be resampled from.
        dst_geom:
            Geometry object defining the destination geographic region that
            input data will be resampled to.
        resampler:
            The name of a resampler class that has been previously
            registered with :func:`~pyresample.future.resampler_registry` and
            will be instantiated. If not provided then a registered Resampler
            class will be chosen based on the geometry types provided. This
            is currently always the 'nearest' (nearest neighbor) resampler.
        cache:
            :class:`pyresample.future.cache.ResampleCache` instance used by
            the resampler to cache intermediate results for improved resampling
            performance on multiple executions or future use of the resampler.
        kwargs:
            Additional keyword arguments to pass to the Resampler. Note that
            most resamplers do not have additional keyword arguments on
            creation, but instead have extra arguments passed when their
            ``resample`` methods are called.

    """
    if resampler is None:
        resampler = "nearest"
    rcls = RESAMPLER_REGISTRY[resampler]
    return rcls(src_geom, dst_geom, cache=cache, **kwargs)
