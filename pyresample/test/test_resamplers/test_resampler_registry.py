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
"""Test resampler registry operations."""

from __future__ import annotations

import contextlib
import warnings
from unittest import mock

import pytest

from pyresample.future import Resampler, list_resamplers, register_resampler, unregister_resampler
from pyresample.test.utils import assert_warnings_contain


def _custom_resampler_class():
    class _MyResampler(Resampler):
        """Fake resampler class."""

        def precompute(self):
            """Pretend to be a precompute method."""
            return "PRECOMPUTE"

        def compute(self):
            """Pretend to be a compute method."""
            return None
    return _MyResampler


class TestResamplerRegistryManipulation:
    """Test basic behavior of the resampler registry when it is modified."""

    def setup_method(self):
        """Mock the registry container so we don't effect the "real" registry."""
        _ = list_resamplers()  # force registry to be filled so we can overwrite it...
        self.mock_reg = mock.patch("pyresample.future.resamplers.registry.RESAMPLER_REGISTRY", {})
        self.mock_reg.start()

    def teardown_method(self):
        """Undo mock of registry."""
        self.mock_reg.stop()

    def test_no_builtins_warning(self):
        """Test that if no builtins are found that a warning is issued."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            avail_resamplers = list_resamplers()
            # the mocking should have made this empty
            assert not avail_resamplers
        assert_warnings_contain(w, "reinstall")

    def test_manual_resampler_registration(self):
        rname = "my_resampler"
        _register_resampler_class(rname, Resampler)
        unregister_resampler(rname)
        with _ignore_no_builtin_resamplers():
            assert rname not in list_resamplers()

    @pytest.mark.parametrize('new_resampler', [Resampler, _custom_resampler_class()])
    def test_multiple_registration_warning_same_class(self, new_resampler):
        rname = "my_resampler"
        _register_resampler_class(rname, Resampler)

        with pytest.raises(ValueError):
            _register_resampler_class(rname, new_resampler, no_exist=False)

        unregister_resampler(rname)
        _register_resampler_class(rname, Resampler)


def _register_resampler_class(rname, rcls, no_exist=True):
    if no_exist:
        with _ignore_no_builtin_resamplers():
            assert rname not in list_resamplers()
    register_resampler(rname, rcls)
    assert rname in list_resamplers()


@contextlib.contextmanager
def _ignore_no_builtin_resamplers():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No builtin resamplers", category=UserWarning)
        yield


class TestBuiltinResamplerRegistry:
    """Test the registry based on known and builtin functionality."""

    @pytest.mark.parametrize("resampler", ["nearest"])
    def test_minimal_resamplers_exist(self, resampler):
        avail_resampler = list_resamplers()
        assert resampler in avail_resampler
