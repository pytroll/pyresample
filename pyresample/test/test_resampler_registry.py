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

from unittest import mock

import pytest


class TestResamplerRegistryManipulation:
    """Test basic behavior of the resampler registry when it is modified."""

    def setup_method(self):
        """Mock the registry container so we don't effect the "real" registry."""
        self.mock_reg = mock.patch("pyresample.future.resampler_registry", "RESAMPLER_REGISTRY", {})
        self.mock_reg.start()

    def teardown_method(self):
        """Undo mock of registry."""
        self.mock_reg.stop()

    def test_manual_resampler_registration(self):
        from pyresample.future import register_resampler, unregister_resampler, list_resamplers
        from pyresample.future import Resampler
        rname = "my_resampler"
        assert rname not in list_resamplers()
        register_resampler(rname, Resampler)
        assert rname in list_resamplers()
        unregister_resampler(rname)
        assert rname not in list_resamplers()


class TestBuiltinResamplerRegistry:
    """Test the registry based on known and builtin functionality."""

    @pytest.mark.parametrize("resampler", ["nearest"])
    def test_minimal_resamplers_exist(self, resampler):
        from pyresample.future import list_resamplers
        avail_resampler = list_resamplers()
        assert resampler in avail_resampler
