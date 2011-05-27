#pyresample, Resampling of remote sensing image data in python
# 
#Copyright (C) 2010  Esben S. Nielsen
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import ctypes

import numpy as np
import pyproj
import scipy.spatial as sp
import multiprocessing as mp

from _multi_proc import shmem_as_ndarray, Scheduler

#Earth radius
R = 6370997.0


class cKDTree_MP(object):
    ''' Multiprocessing cKDTree subclass, shared memory '''

    def __init__(self, data, leafsize=10, nprocs=2, chunk=None,\
                 schedule='guided'):
        '''
        Same as cKDTree.__init__ except that an internal copy
        of data to shared memory is made.
        Extra keyword arguments:
        chunk : Minimum chunk size for the load balancer.
        schedule: Strategy for balancing work load
        ('static', 'dynamic' or 'guided').
        '''

        self.n, self.m = data.shape
        # Allocate shared memory for data
        self.shmem_data = mp.RawArray(ctypes.c_double, self.n*self.m)
        
        # View shared memory as ndarray, and copy over the data.
        # The RawArray objects have information about the dtype and
        # buffer size.
        _data = shmem_as_ndarray(self.shmem_data).reshape((self.n, self.m))
        _data[:,:] = data
        
        # Initialize parent, we must do this last because
        # cKDTree stores a reference to the data array. We pass in
        # the copy in shared memory rather than the origial data.
        self.leafsize = leafsize
        self._nprocs = nprocs
        self._chunk = chunk
        self._schedule = schedule        
        
    def query(self, x, k=1, eps=0, p=2, distance_upper_bound=np.inf):
        '''
        Same as cKDTree.query except parallelized with multiple
        processes and shared memory.        
        '''
        
        # allocate shared memory for x and result
        nx = x.shape[0]
        shmem_x = mp.RawArray(ctypes.c_double, nx*self.m)
        shmem_d = mp.RawArray(ctypes.c_double, nx*k)
        shmem_i = mp.RawArray(ctypes.c_int, nx*k)
        
        # view shared memory as ndarrays
        _x = shmem_as_ndarray(shmem_x).reshape((nx, self.m))
        if k == 1:
            _d = shmem_as_ndarray(shmem_d)
            _i = shmem_as_ndarray(shmem_i)
        else:
            _d = shmem_as_ndarray(shmem_d).reshape((nx, k))
            _i = shmem_as_ndarray(shmem_i).reshape((nx, k))
        
        # copy x to shared memory
        _x[:] = x
        
        # set up a scheduler to load balance the query        
        scheduler = Scheduler(nx, self._nprocs, chunk=self._chunk,\
                              schedule=self._schedule)

        # query with multiple processes
        query_args = [scheduler, self.shmem_data, self.n, self.m,\
                      self.leafsize, shmem_x, nx, shmem_d, shmem_i,\
                      k, eps, p, distance_upper_bound]
                
        _run_jobs(_parallel_query, query_args, self._nprocs)
        # return results (private memory)
        return _d.copy(), _i.copy()
    

class Proj(pyproj.Proj):

    def __call__(self, data1, data2, inverse=False, radians=False,\
                 errcheck=False, nprocs=1):
        return super(Proj, self).__call__(data1, data2, inverse=inverse,\
                                          radians=radians, errcheck=errcheck)


class Proj_MP(pyproj.Proj):
    
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        
    def __call__(self, data1, data2, inverse=False, radians=False,\
                 errcheck=False, nprocs=2, chunk=None, schedule='guided'):
        grid_shape = data1.shape
        n = data1.size
        
        #Create shared memory
        shmem_data1 = mp.RawArray(ctypes.c_double, n)
        shmem_data2 = mp.RawArray(ctypes.c_double, n)
        shmem_res1 = mp.RawArray(ctypes.c_double, n)
        shmem_res2 = mp.RawArray(ctypes.c_double, n)
        
        # view shared memory as ndarrays
        _data1 = shmem_as_ndarray(shmem_data1)
        _data2 = shmem_as_ndarray(shmem_data2)
        _res1 = shmem_as_ndarray(shmem_res1)
        _res2 = shmem_as_ndarray(shmem_res2)
        
        # copy input data to shared memory
        _data1[:] = data1.ravel()
        _data2[:] = data2.ravel()
        
        # set up a scheduler to load balance the query        
        scheduler = Scheduler(n, nprocs, chunk=chunk, schedule=schedule)
                
        # Projection with multiple processes
        proj_call_args = [scheduler, shmem_data1, shmem_data2, shmem_res1,\
                          shmem_res2, self._args, self._kwargs, inverse,\
                          radians, errcheck]
        
        _run_jobs(_parallel_proj, proj_call_args, nprocs)
        return _res1.copy().reshape(grid_shape), _res2.copy().reshape(grid_shape)


class Cartesian(object):
    
    def transform_lonlats(self, lons, lats):
    
        coords = np.zeros((lons.size, 3))
        lons_rad = np.radians(lons)
        lats_rad = np.radians(lats)
        coords[:, 0] = R*np.cos(lats_rad)*np.cos(lons_rad)
        coords[:, 1] = R*np.cos(lats_rad)*np.sin(lons_rad)
        coords[:, 2] = R*np.sin(lats_rad)
        
        return coords
    
    
class Cartesian_MP(object):
    
    def __init__(self, nprocs=2, chunk=None, schedule='guided'):
        self._nprocs = nprocs
        self._chunk = chunk
        self._schedule = schedule
    
    def transform_lonlats(self, lons, lats):
        n = lons.size
        
        #Create shared memory
        shmem_lons = mp.RawArray(ctypes.c_double, n)
        shmem_lats = mp.RawArray(ctypes.c_double, n)        
        shmem_coords = mp.RawArray(ctypes.c_double, 3*n)
        
        # view shared memory as ndarrays
        _lons = shmem_as_ndarray(shmem_lons)
        _lats = shmem_as_ndarray(shmem_lats)
        #_coords_x = shmem_as_ndarray(shmem_coords_x)
        #_coords_y = shmem_as_ndarray(shmem_coords_y)
        #_coords_z = shmem_as_ndarray(shmem_coords_z)
        _coords = shmem_as_ndarray(shmem_coords).reshape((n, 3))
        
        
        # copy input data to shared memory
        _lons[:] = lons.ravel()
        _lats[:] = lats.ravel()
        
        # set up a scheduler to load balance the query        
        scheduler = Scheduler(n, self._nprocs, chunk=self._chunk, schedule=self._schedule)
        
        # Projection with multiple processes
        transform_call_args = [scheduler, shmem_lons, shmem_lats, n, shmem_coords]
        
        _run_jobs(_parallel_transform, transform_call_args, self._nprocs)
        return _coords.copy()
        

def _run_jobs(target, args, nprocs):
    """Run process pool
    """

    # return status in shared memory
    # access to these values are serialized automatically
    ierr = mp.Value(ctypes.c_int, 0)
    err_msg = mp.Array(ctypes.c_char, 1024)
    
    args.extend((ierr, err_msg))
    
    pool = [mp.Process(target=target, args=args) for n in range(nprocs)]
    for p in pool: p.start()
    for p in pool: p.join()
    if ierr.value != 0:
        raise RuntimeError,\
                ('%d errors in worker processes. Last one reported:\n%s'%\
                 (ierr.value, err_msg.value))
                
# This is executed in an external process:
def _parallel_query(scheduler, # scheduler for load balancing
                    data, ndata, ndim, leafsize, # data needed to reconstruct the kd-tree
                    x, nx, d, i, # query data and results
                    k, eps, p, dub, # auxillary query parameters
                    ierr, err_msg): # return values (0 on success)
    
    try:     
        # View shared memory as ndarrays.
        _data = shmem_as_ndarray(data).reshape((ndata, ndim))
        _x = shmem_as_ndarray(x).reshape((nx, ndim))
        if k == 1:
            _d = shmem_as_ndarray(d)
            _i = shmem_as_ndarray(i)
        else:
            _d = shmem_as_ndarray(d).reshape((nx, k))
            _i = shmem_as_ndarray(i).reshape((nx, k))

        # Reconstruct the kd-tree from the data.        
        kdtree = sp.cKDTree(_data, leafsize=leafsize)

        # Query for nearest neighbours, using slice ranges,
        # from the load balancer.
        for s in scheduler:
            if k == 1:
                _d[s], _i[s] = kdtree.query(_x[s,:], k=1, eps=eps, p=p,\
                                                distance_upper_bound=dub)
            else:
                _d[s,:], _i[s,:] = kdtree.query(_x[s,:], k=k, eps=eps, p=p,\
                                                distance_upper_bound=dub)
    # An error occured, increment the return value ierr.
    # Access to ierr is serialized by multiprocessing.
    except Exception, e:
        ierr.value += 1
        err_msg.value = e.message  
        
def _parallel_proj(scheduler, data1, data2, res1, res2, proj_args, proj_kwargs,\
                   inverse, radians, errcheck, ierr, err_msg):
    try:
        # View shared memory as ndarrays.
        _data1 = shmem_as_ndarray(data1)
        _data2 = shmem_as_ndarray(data2)
        _res1 = shmem_as_ndarray(res1)
        _res2 = shmem_as_ndarray(res2)
        
        #Initialise pyproj
        proj = pyproj.Proj(*proj_args, **proj_kwargs)
        
        #Reproject data segment
        for s in scheduler:
            _res1[s], _res2[s] = proj(_data1[s], _data2[s], inverse=inverse,\
                                       radians=radians, errcheck=errcheck)
    
    # An error occured, increment the return value ierr.
    # Access to ierr is serialized by multiprocessing.
    except Exception, e:
        ierr.value += 1
        err_msg.value = e.message  
        
def _parallel_transform(scheduler, lons, lats, n, coords, ierr, err_msg):
    try:
        # View shared memory as ndarrays.
        _lons = shmem_as_ndarray(lons)
        _lats = shmem_as_ndarray(lats)
        _coords = shmem_as_ndarray(coords).reshape((n, 3))
        
        #Transform to cartesian coordinates
        for s in scheduler:
            _coords[s, 0] = R*np.cos(np.radians(_lats[s]))*np.cos(np.radians(_lons[s]))
            _coords[s, 1] = R*np.cos(np.radians(_lats[s]))*np.sin(np.radians(_lons[s]))
            _coords[s, 2] = R*np.sin(np.radians(_lats[s]))
    # An error occured, increment the return value ierr.
    # Access to ierr is serialized by multiprocessing.
    except Exception, e:
        ierr.value += 1
        err_msg.value = e.message  
