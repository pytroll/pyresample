# pyresample, Resampling of remote sensing image data in python
#
# Copyright (C) 2010, 2015  Esben S. Nielsen
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import

import ctypes
import multiprocessing as mp

import numpy as np


class Scheduler(object):

    def __init__(self, ndata, nprocs, chunk=None, schedule='guided'):
        if not schedule in ['guided', 'dynamic', 'static']:
            raise ValueError('unknown scheduling strategy')
        self._ndata = mp.RawValue(ctypes.c_int, ndata)
        self._start = mp.RawValue(ctypes.c_int, 0)
        self._lock = mp.Lock()
        self._schedule = schedule
        self._nprocs = nprocs
        if schedule == 'guided' or schedule == 'dynamic':
            min_chunk = ndata // (10 * nprocs)
            if chunk:
                min_chunk = chunk
            min_chunk = max(min_chunk, 1)
            self._chunk = min_chunk
        elif schedule == 'static':
            min_chunk = ndata // nprocs
            if chunk:
                min_chunk = max(chunk, min_chunk)
            min_chunk = max(min_chunk, 1)
            self._chunk = min_chunk

    def __iter__(self):
        while True:
            self._lock.acquire()
            ndata = self._ndata.value
            nprocs = self._nprocs
            start = self._start.value
            if self._schedule == 'guided':
                _chunk = ndata // nprocs
                chunk = max(self._chunk, _chunk)
            else:
                chunk = self._chunk
            if ndata:
                if chunk > ndata:
                    s0 = start
                    s1 = start + ndata
                    self._ndata.value = 0
                else:
                    s0 = start
                    s1 = start + chunk
                    self._ndata.value = ndata - chunk
                    self._start.value = start + chunk
                self._lock.release()
                yield slice(s0, s1)
            else:
                self._lock.release()
                return


def shmem_as_ndarray(raw_array):
    _ctypes_to_numpy = {
        ctypes.c_char: np.int8,
        ctypes.c_wchar: np.int16,
        ctypes.c_byte: np.int8,
        ctypes.c_ubyte: np.uint8,
        ctypes.c_short: np.int16,
        ctypes.c_ushort: np.uint16,
        ctypes.c_int: np.int32,
        ctypes.c_uint: np.int32,
        ctypes.c_long: np.int32,
        ctypes.c_ulong: np.int32,
        ctypes.c_float: np.float32,
        ctypes.c_double: np.float64
    }
    dtype = _ctypes_to_numpy[raw_array._type_]

    # The following works too, but occasionally raises
    # RuntimeWarning: Item size computed from the PEP 3118 buffer format string does not match the actual item size.
    # and appears to be slower.
    # return np.ctypeslib.as_array(raw_array)

    return np.frombuffer(raw_array, dtype=dtype)
