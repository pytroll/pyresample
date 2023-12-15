# Copyright (c) 2023 Pyresample developers
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
"""Tests for feature flags changed using configuration."""
# import importlib
# import sys
#
# import pyresample
# from pyresample.geometry import AreaDefinition as LegacyAreaDefinition
# from pyresample.future.geometry import AreaDefinition as FutureAreaDefinition
#
#
# def test_future_geometries():
#     # with pyresample.config.set({"features.future_geometries": True}):
#     #     _reload_all_pyresample()
#     #     from pyresample import AreaDefinition
#     #     _equal_full_name(AreaDefinition, FutureAreaDefinition)
#     with pyresample.config.set({"features.future_geometries": False}):
#         _reload_all_pyresample()
#         from pyresample import AreaDefinition
#         _equal_full_name(AreaDefinition, LegacyAreaDefinition)
#     # reset environment back to default configuration
#     _reload_all_pyresample()
#
#
# def _equal_full_name(cls1, cls2):
#     """Compare full module class name.
#
#     This is needed when reloading modules makes past instances not the same as new instances.
#
#     """
#     assert cls1.__qualname__ == cls2.__qualname__
#     assert cls1.__module__ == cls2.__module__
#
#
# def _reload_all_pyresample():
#     pyresample_mods = [(mod_name, mod) for mod_name, mod in sys.modules.items() if "pyresample" in mod_name and
#                        ("config" not in mod_name)]
#     # pyresample_mods = [(mod_name, mod) for mod_name, mod in sys.modules.items() if "pyresample" in mod_name and
#     #                    ("config" not in mod_name and "pyresample.test" not in mod_name)]
#     for mod_name, module in pyresample_mods:
#         # print(f"Reloading {mod_name}")
#         del sys.modules[mod_name]
#         # importlib.reload(module)
#     for mod_name, module in pyresample_mods:
#         print(f"Reloading {mod_name}")
#         # del sys.modules[mod_name]
#         importlib.reload(module)
#     importlib.reload(sys.modules[__name__])
#     # for module in pyresample_mods:
#     #     importlib.reload(module)
