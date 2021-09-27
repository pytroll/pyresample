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
        self.mock_reg = mock.patch("pyresample.future.resamplers.registry.RESAMPLER_REGISTRY", {})
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

    def test_multiple_registration_warning(self):
        import warnings
        from pyresample.future import register_resampler, list_resamplers
        from pyresample.future import Resampler
        rname = "my_resampler"
        assert rname not in list_resamplers()
        register_resampler(rname, Resampler)
        assert rname in list_resamplers()

        with warnings.catch_warnings(record=True) as w:
            # same class
            register_resampler(rname, Resampler)
        assert len(w) >= 1
        msgs = [msg.message.args[0].lower() for msg in w]
        assert any("already registered" in msg for msg in msgs)

        with warnings.catch_warnings(record=True) as w:
            # different class
            register_resampler(rname, _custom_resampler_class())
        assert len(w) >= 1
        msgs = [msg.message.args[0].lower() for msg in w]
        assert any("replacing" in msg for msg in msgs)

    @pytest.mark.parametrize(
        "names",
        [
            ["my_decorated_resampler"],
            ["my_decorated_resampler", "my_decorated_resampler2"]
        ]
    )
    def test_decorator_registration(self, names):
        from pyresample.future import register_resampler, list_resamplers, create_resampler
        for rname in names:
            assert rname not in list_resamplers()

        my_cls = _custom_resampler_class()
        reg_cls = my_cls
        for rname in names:
            reg_cls = register_resampler(rname)(reg_cls)
            assert reg_cls is my_cls

        for rname in names:
            assert rname in list_resamplers()
            inst = create_resampler(None, None, resampler=rname)
            assert isinstance(inst, my_cls)


def _custom_resampler_class():
    from pyresample.future import Resampler

    class _MyResampler(Resampler):
        """Fake resampler class."""

        def precompute(self):
            """Pretend to be a precompute method."""
            return "PRECOMPUTE"

        def compute(self):
            """Pretend to be a compute method."""
            return None
    return _MyResampler


class TestBuiltinResamplerRegistry:
    """Test the registry based on known and builtin functionality."""

    @pytest.mark.parametrize("resampler", ["nearest"])
    def test_minimal_resamplers_exist(self, resampler):
        from pyresample.future import list_resamplers
        avail_resampler = list_resamplers()
        assert resampler in avail_resampler
