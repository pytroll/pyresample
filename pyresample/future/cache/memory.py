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
"""Parent cache classes and utilities for Resampler classes."""

from typing import Any, Hashable

from .base import ResampleCache


class InMemoryCache(ResampleCache):
    """Basic cache that stores everything in-memory."""

    def __init__(self):
        self._store = {}

    def store(self, key, value):
        """Write ``value`` data identified by the unique ``key`` to the cache."""
        self._store[key] = value

    def load(self, key):
        """Retrieve data from the cache using the unique ``key``."""
        return self._store[key]

    def clear(self) -> None:
        """Remove all contents managed by this cache."""
        self._store.clear()

    def pop(self, key: Hashable) -> Any:
        """Retrieve data and remove its entry from the cache."""
        return self._store.pop(key)

    def __len__(self) -> int:
        """Get number of items in the cache."""
        return len(self._store)

    def __contains__(self, key):
        """Check if this cache contains the specified key."""
        return key in self._store
