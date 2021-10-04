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

from abc import ABC, abstractmethod
from typing import Any, Hashable


class ResampleCache(ABC):
    """Base class for all Resampler cache classes."""

    @abstractmethod
    def store(self, key: Hashable, value: Any) -> None:
        """Write ``value`` data identified by the unique ``key`` to the cache."""

    @abstractmethod
    def load(self, key: Hashable) -> Any:
        """Retrieve data from the cache using the unique ``key``."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all contents managed by this cache."""

    def remove(self, key: Hashable) -> None:
        """Remove an item from the cache's internal storage."""
        self.pop(key)

    @abstractmethod
    def pop(self, key: Hashable) -> Any:
        """Retrieve data and remove its entry from the cache."""

    @abstractmethod
    def __len__(self) -> int:
        """Get number of items in the cache."""

    def __contains__(self, key):
        """Check if this cache contains the specified key."""
        try:
            self.load(key)
        except KeyError:
            return False
        else:
            return True

    def __repr__(self):
        """Summarize the current state of the cache as a string.

        Information that would be good to include is:

        * Number of items in the cache
        * Total size (in-memory or on-disk)
        * Important settings for the cache (ex. expiration time, on-disk path)

        """
        mod = self.__class__.__module__
        qualname = self.__class__.__qualname__
        return f"<{mod}.{qualname} cache with {len(self)} item(s) at {hex(id(self))}>"
