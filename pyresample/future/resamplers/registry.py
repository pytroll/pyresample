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

import sys
import warnings
from typing import Type

from .resampler import Resampler

RESAMPLER_REGISTRY: dict[str, Type[Resampler]] = {}
ENTRY_POINTS_LOADED: bool = False


def register_resampler(resampler_name: str, resampler_cls: Type[Resampler]) -> None:
    """Register :class:`~pyresample.future.resampler.Resampler` subclass for future use.

    Args:
        resampler_name:
            Name of the resampler in the registry. This name can then be used
            in functions like
            :func:`~pyresample.future.resamplers.registry.create_resampler`.
        resampler_cls:
            Subclass of
            :class:`~pyresample.future.resamplers.resampler.Resampler` that
            will be added to the registry.

    Examples:
        Register a custom class::

            register_resampler("my_resampler", MyResamplerClass)

        Register as a plugin from third-party package (in your setup.py)::

            entry_points = {
                "pyresample.resamplers": [
                    "my_resampler = mypkg.mymodule:MyResamplerClass",
                ],
            }

    """
    if resampler_name in RESAMPLER_REGISTRY:
        warnings.warn(f"Resampler with name '{resampler_name} is already "
                      f"registered. Replacing with new resampler class.", RuntimeWarning)

    RESAMPLER_REGISTRY[resampler_name] = resampler_cls


def unregister_resampler(resampler_name: str) -> None:
    """Remove previously registered Resampler so it can't be used anymore."""
    del RESAMPLER_REGISTRY[resampler_name]


def list_resamplers() -> list[str, ...]:
    """Get sorted list of registered resamplers."""
    _load_and_validate_registry()
    resampler_names = sorted(RESAMPLER_REGISTRY.keys())
    return resampler_names


def create_resampler(
        src_geom,
        dst_geom,
        resampler: str = None,
        cache=None,
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
            ResampleCache instance used by
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
    _load_and_validate_registry()
    resampler_cls = RESAMPLER_REGISTRY[resampler]
    return resampler_cls(src_geom, dst_geom, cache=cache, **kwargs)


def _load_and_validate_registry():
    _load_entry_point_resamplers()
    if not RESAMPLER_REGISTRY:
        warnings.warn("No builtin resamplers found. This probably means you "
                      "pyresample installed in editable mode. Try reinstalling "
                      "pyresample to ensure builtin resamplers are included.")


def _load_entry_point_resamplers():
    """Load setuptools plugins via entry_points.

    Based on https://packaging.python.org/guides/creating-and-discovering-plugins/#using-package-metadata.

    """
    global ENTRY_POINTS_LOADED
    if ENTRY_POINTS_LOADED:
        return
    if sys.version_info < (3, 10):
        from importlib_metadata import entry_points
    else:
        from importlib.metadata import entry_points

    discovered_plugins = entry_points(group="pyresample.resamplers")
    for entry_point in discovered_plugins:
        try:
            loaded_resampler = entry_point.load()
        except ImportError:
            warnings.warn(f"Unable to load resampler from plugin: {entry_point.name}")
        else:
            register_resampler(entry_point.name, loaded_resampler)
    ENTRY_POINTS_LOADED = True
