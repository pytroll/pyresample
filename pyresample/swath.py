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

"""Handles reprojection of satellite swath data. Several types of resampling are supported"""

import types

import numpy as np
import scipy.spatial as sp

import geometry
import data_reduce
import _spatial_mp
#import _resample_iter

#Earth radius
R = 6371000

#class _AreaDefContainer:
#    
#    def __init__(self, target_area_def, nprocs=1):
#        #Check if target_area_def is preprocessed grid coordinates
#        if isinstance(target_area_def, geometry.AreaDefinition):
#            self.is_cartesian_grid = False
#            self.is_lonlat_grid = False
#        elif isinstance(target_area_def, np.ndarray):
#            if target_area_def.ndim == 3:
#                if target_area_def.shape[2] == 2:
#                    grid_lons = target_area_def[:, :, 0]
#                    grid_lats = target_area_def[:, :, 1]
#                    self.is_cartesian_grid = False                
#                    self.is_lonlat_grid = True
#                elif target_area_def.shape[2] == 3:
#                    grid_coords = target_area_def
#                    self.is_cartesian_grid = True
#                    self.is_lonlat_grid = False
#                else:
#                    raise TypeError(('Third dimension of grid expected '
#                                    'to be of length 2 or 3 not %s') % 
#                                    target_area_def.shape[2])            
#            else:
#                raise TypeError('target_area_def as numpy array must be of 3 '
#                                'dimensions')
#        else:
#            raise TypeError('target_area_def must be either AreaDefinition or '
#                            'numpy array')
#
#        if nprocs > 1:
#            cartesian = _spatial_mp.Cartesian_MP(nprocs)
#        else:
#            cartesian = _spatial_mp.Cartesian()
#            
#        
#        if self.is_cartesian_grid:
#            #Cartesian grid has been passed as target_area_def
#            self.cartesian_grid = grid_coords
#            self.grid_size = grid_coords[:, :, 0].size
#            self.grid_shape = grid_coords[:, :, 0].shape
#        
#            #Make grid coords array. Copy needed to avoid silent failure of kd-tree query
#            self.grid_coords = np.column_stack((grid_coords[:, :, 0].ravel(), 
#                                                grid_coords[:, :, 1].ravel(), 
#                                                grid_coords[:, :, 2].ravel())
#                                                ).copy()
#            
#        else:
#            if self.is_lonlat_grid:
#                #Lons and lats has been passed as target_area_def
#                self.grid_lons = grid_lons
#                self.grid_lats = grid_lats
#            else:
#                #Get lons and lats from area_def dict
#                self.grid_lons, self.grid_lats = \
#                                    target_area_def.get_lonlats(nprocs)
#                
#            
#            #Get grid size and shape
#            self.grid_size = self.grid_lons.size
#            self.grid_shape = self.grid_lons.shape
#            
#            #Transform grid to cartesian
#            self.grid_coords = cartesian.transform_lonlats(self.grid_lons.ravel(),
#                                                           self.grid_lats.ravel())
#            
#    def get_valid_index(self, lons, lats, radius_of_influence):
#        """Returns array of relevant indices of lons and lats
#        :Parameters:
#        lons : numpy array
#            1d array of swath lons
#        lats : numpy array               
#            1d array of swath lats
#        radius_of_influence : float 
#            Cut off distance in meters
#            
#        :Returns:
#        valid_index : numpy array
#            Boolean array of same size as lons and lats indicating relevant indices
#        """
#        
#        if self.is_cartesian_grid:
#            #Cartesian grid has been passed as target_area_def
#            valid_index = data_reduce.get_valid_index_from_cartesian_grid(self.cartesian_grid, lons,
#                                                                          lats, 
#                                                                          radius_of_influence)
#        else:
#            #Grid is lon lat
#            valid_index = data_reduce.get_valid_index_from_lonlat_grid(self.grid_lons, 
#                                                                       self.grid_lats, 
#                                                                       lons, lats,  
#                                                                       radius_of_influence)
#        return valid_index            
        
        
def resample_nearest(source_geo_def, data, target_geo_def,
                     radius_of_influence, epsilon=0,
                     fill_value=0, reduce_data=True, nprocs=1):
    """Resamples swath using kd-tree nearest neighbour approach

    :Parameters:
    lons : numpy array
        1d array of swath lons
    lats : numpy array               
        1d array of swath lats
    data : numpy array               
        1d array of single channel data points or
        (swath_size, k) array of k channels of datapoints
    target_area_def : AreaDefinition or numpy array
        Target area definition as instance of AreaDefinition 
        or array containing lons and lats of the target area 
        according to:
        target_area_def[:, :, 0] = lons and
        target_area_def[:, :, 1] = lats
        or array containing cartesian coordinates the target
        area according to:
        target_area_def[:, :, 0] = X and
        target_area_def[:, :, 1] = Y and
        target_area_def[:, :, 2] = Z
    radius_of_influence : float 
        Cut off distance in meters
    neighbours : int, optional 
        The number of neigbours to consider for each swath point
    epsilon : float, optional
        Allowed uncertainty in meters. Increasing uncertainty
        reduces execution time
    fill_value : {int, None}, optional 
            Set undetermined pixels to this value.
            If fill_value is None a masked array is returned 
            with undetermined pixels masked    
    reduce_data : bool, optional
        Perform initial coarse reduction of swath dataset in order
        to reduce execution time
    nprocs : int, optional
        Number of processor cores to be used
               
    :Returns: 
    data : numpy array 
        Swath data resampled to target grid
    """
    
    return _resample(source_geo_def, data, target_geo_def, 'nn',
                     radius_of_influence, neighbours=1,
                     epsilon=epsilon, fill_value=fill_value,
                     reduce_data=reduce_data, nprocs=nprocs)

def resample_gauss(source_geo_def, data, target_geo_def,
                   radius_of_influence, sigmas, neighbours=8, epsilon=0,
                   fill_value=0, reduce_data=True, nprocs=1):
    """Resamples swath using kd-tree gaussian weighting neighbour approach

    :Parameters:
    lons : numpy array
        1d array of swath lons
    lats : numpy array               
        1d array of swath lats
    data : numpy array               
        1d array of single channel data points or
        (swath_size, k) array of k channels of datapoints
    target_area_def : AreaDefinition or numpy array
        Target area definition as instance of AreaDefinition 
        or array containing lons and lats of the target area 
        according to:
        target_area_def[:, :, 0] = lons and
        target_area_def[:, :, 1] = lats
        or array containing cartesian coordinates the target
        area according to:
        target_area_def[:, :, 0] = X and
        target_area_def[:, :, 1] = Y and
        target_area_def[:, :, 2] = Z
    radius_of_influence : float 
        Cut off distance in meters
    sigmas : list of floats or float            
        List of sigmas to use for the gauss weighting of each 
        channel 1 to k, w_k = exp(-dist^2/sigma_k^2).
        If only one channel is resampled sigmas is a single float value.
    neighbours : int, optional 
        The number of neigbours to consider for each grid point
    epsilon : float, optional
        Allowed uncertainty in meters. Increasing uncertainty
        reduces execution time
    fill_value : {int, None}, optional 
            Set undetermined pixels to this value.
            If fill_value is None a masked array is returned 
            with undetermined pixels masked    
    reduce_data : bool, optional
        Perform initial coarse reduction of swath dataset in order
        to reduce execution time
    nprocs : int, optional
        Number of processor cores to be used
    
    :Returns: 
    data : numpy array 
        Swath data resampled to target grid                  
    """
    
    def gauss(sigma):
        return lambda r: np.exp(-r**2 / float(sigma)**2)
    
    is_multi_channel = False
    try:
        sigmas.__iter__()
        sigma_list = sigmas
        is_multi_channel = True
    except:
        sigma_list = [sigmas] 
        
        
    for sigma in sigma_list:
        if not isinstance(sigma, (long, int, float)):
            raise TypeError('sigma must be number')    
    
    if is_multi_channel:
        weight_funcs = map(gauss, sigma_list) 
    else:
        weight_funcs = gauss(sigmas)
        
    return _resample(source_geo_def, data, target_geo_def, 'custom',
                     radius_of_influence, neighbours=neighbours,
                     epsilon=epsilon, weight_funcs=weight_funcs, fill_value=fill_value,
                     reduce_data=reduce_data, nprocs=nprocs)

def resample_custom(source_geo_def, data, target_geo_def,
                    radius_of_influence, weight_funcs, neighbours=8,
                    epsilon=0, fill_value=0, reduce_data=True, nprocs=1):
    """Resamples swath using kd-tree custom radial weighting neighbour approach

    :Parameters:
    lons : numpy array
        1d array of swath lons
    lats : numpy array               
        1d array of swath lats
    data : numpy array               
        1d array of single channel data points or
        (swath_size, k) array of k channels of datapoints
    target_area_def : AreaDefinition or numpy array
        Target area definition as instance of AreaDefinition 
        or array containing lons and lats of the target area 
        according to:
        target_area_def[:, :, 0] = lons and
        target_area_def[:, :, 1] = lats
        or array containing cartesian coordinates the target
        area according to:
        target_area_def[:, :, 0] = X and
        target_area_def[:, :, 1] = Y and
        target_area_def[:, :, 2] = Z
    radius_of_influence : float 
        Cut off distance in meters
    weight_funcs : list function objects or function object       
        List of weight functions f(dist) to use for the weighting 
        of each channel 1 to k.
        If only one channel is resampled weight_funcs is
        a single function object.
    neighbours : int, optional 
        The number of neigbours to consider for each grid point
    epsilon : float, optional
        Allowed uncertainty in meters. Increasing uncertainty
        reduces execution time
    fill_value : {int, None}, optional 
            Set undetermined pixels to this value.
            If fill_value is None a masked array is returned 
            with undetermined pixels masked    
    reduce_data : bool, optional
        Perform initial coarse reduction of swath dataset in order
        to reduce execution time
    nprocs : int, optional
        Number of processor cores to be used
    
    :Returns: 
    data : numpy array 
        Swath data resampled to target grid
    """
    
    if data.ndim > 1:
        for weight_func in weight_funcs:
            if not isinstance(weight_func, types.FunctionType):
                raise TypeError('weight_func must be function object')
    else:
        if not isinstance(weight_funcs, types.FunctionType):
            raise TypeError('weight_func must be function object')
        
    return _resample(source_geo_def, data, target_geo_def, 'custom',
                     radius_of_influence, neighbours=neighbours,
                     epsilon=epsilon, weight_funcs=weight_funcs,
                     fill_value=fill_value, reduce_data=reduce_data,
                     nprocs=nprocs)

def _resample(source_geo_def, data, target_geo_def, resample_type,
             radius_of_influence, neighbours=8, epsilon=0, weight_funcs=None,
             fill_value=0, reduce_data=True, nprocs=1):
    """Resamples swath using kd-tree approach"""

    #Check validity of input
#    if not (isinstance(lons, np.ndarray) and \
#            isinstance(lats, np.ndarray) and \
#            isinstance(data, np.ndarray)):
#        raise TypeError('lons, lats and data must be numpy arrays')
#    elif lons.ndim != 1 or lons.ndim != 1:
#        raise TypeError('lons and lats must one dimensional arrays')
#    elif lons.size != lats.size or lats.size != data.shape[0] or \
#         data.shape[0] != lons.size:
#        raise TypeError('Not the same number of datapoints in lons, lats and '
#                        'data')
        
    #area_def_con = _AreaDefContainer(target_area_def, nprocs)
    valid_input_index, valid_output_index, index_array, distance_array = \
                                 get_neighbour_info(source_geo_def, 
                                                    target_geo_def, 
                                                    radius_of_influence, 
                                                    neighbours=neighbours, 
                                                    epsilon=epsilon, 
                                                    reduce_data=reduce_data, 
                                                    nprocs=nprocs)
    import h5py
    h5out = h5py.File('/home/esn/tmp/06.h5', 'w')
    h5out['index_array'] = index_array
    h5out.close()
    
    return get_sample_from_neighbour_info(resample_type, 
                                          target_geo_def.shape, 
                                          data, valid_input_index, 
                                          valid_output_index, index_array, 
                                          distance_array=distance_array, 
                                          weight_funcs=weight_funcs, 
                                          fill_value=fill_value)
    
#def get_neighbour_info(lons, lats, target_area_def, radius_of_influence, 
#                       neighbours=8, epsilon=0, reduce_data=True, nprocs=1):
#    """Returns information on nearest neighbours and relevant swath data points
#    
#    :Parameters:
#    lons : numpy array
#        1d array of swath lons
#    lats : numpy array               
#        1d array of swath lats
#    target_area_def : AreaDefinition or numpy array
#        Target area definition as instance of AreaDefinition 
#        or array containing lons and lats of the target area 
#        according to:
#        target_area_def[:, :, 0] = lons and
#        target_area_def[:, :, 1] = lats
#        or array containing cartesian coordinates the target
#        area according to:
#        target_area_def[:, :, 0] = X and
#        target_area_def[:, :, 1] = Y and
#        target_area_def[:, :, 2] = Z
#    radius_of_influence : float 
#        Cut off distance in meters
#    neighbours : int, optional 
#        The number of neigbours to consider for each grid point
#    epsilon : float, optional
#        Allowed uncertainty in meters. Increasing uncertainty
#        reduces execution time
#    epsilon : float, optional
#        Allowed uncertainty in meters. Increasing uncertainty
#        reduces execution time
#        
#    :Returns: 
#    (valid_index, index_array, distance_array) : tuple of numpy array 
#        valid_index: boolean array of relevant swath indices
#        index_array: array of neighbour indices in reduced swath array
#        distance_array: array of distances to neighbours
#    """
#     
#    area_def_con = _AreaDefContainer(target_area_def, nprocs)
#    return _get_neighbour_info(lons, lats, area_def_con, radius_of_influence, 
#                               neighbours=neighbours, epsilon=epsilon, 
#                               reduce_data=reduce_data, nprocs=nprocs)
    
def get_neighbour_info(source_geo_def, target_geo_def, radius_of_influence, 
                       neighbours=8, epsilon=0, reduce_data=True, nprocs=1):    
    """Returns neighbour info"""

    if not isinstance(source_geo_def, geometry.BaseDefinition):
        raise typeError('source_geo_def must be of geometry type')
    elif not isinstance(target_geo_def, geometry.BaseDefinition):
        raise typeError('target_geo_def must be of geometry type')
    
    lons = source_geo_def.lons.ravel()
    lats = source_geo_def.lats.ravel()
    
    #Check validity of input
#    if not (isinstance(lons, np.ndarray) and 
#            isinstance(lats, np.ndarray)):
#        raise TypeError('lons and lats must be numpy arrays')
#    elif lons.ndim != 1 or lons.ndim != 1:
#        raise TypeError('lons and lats must one dimensional arrays')
#    elif lons.size != lats.size:
#        raise TypeError('Not the same number of datapoints in lons and lats')
    
    if not isinstance(radius_of_influence, (long, int, float)):
        raise TypeError('radius_of_influence must be number')
    
    if not isinstance(neighbours, int):
        raise TypeError('neighbours must be integer')
    
    if not isinstance(epsilon, (long, int, float)):
        raise TypeError('epsilon must be number')
    
    #Find invalid data points 
    valid_data = (lons >= -180) * (lons <= 180) * (lats <= 90) * (lats >= -90)
    valid_input_index = np.ones(source_geo_def.size).astype(np.bool)
    valid_output_index = np.ones(target_geo_def.size).astype(np.bool)
    
    if reduce_data:
        #valid_index = target_area_def.get_valid_index(lons, lats, radius_of_influence)
        if (isinstance(source_geo_def, geometry.CoordinateDefinition) and 
            isinstance(target_geo_def, (geometry.GridDefinition, 
                                       geometry.AreaDefinition))) or \
           (isinstance(source_geo_def, (geometry.GridDefinition, 
                                        geometry.AreaDefinition)) and
            isinstance(target_geo_def, (geometry.GridDefinition, 
                                        geometry.AreaDefinition))):
            valid_input_index = \
                data_reduce.get_valid_index_from_cartesian_grid(
                                            target_geo_def.cartesian_coords, 
                                            lons, lats, radius_of_influence)
        elif isinstance(source_geo_def, (geometry.GridDefinition, 
                                         geometry.AreaDefinition)) and \
             isinstance(target_geo_def, geometry.CoordinateDefinition):
            valid_output_index = \
                data_reduce.get_valid_index_from_lonlat_grid(
                                            source_geo_def.lons,
                                            source_geo_def.lats, 
                                            target_geo_def.lons.ravel(), 
                                            target_geo_def.lats.ravel(), 
                                            radius_of_influence)
            valid_output_index = valid_output_index.astype(np.bool)
        else:
            pass
    
    #Find valid data points    
    valid_input_index = (valid_data & valid_input_index)
    if(isinstance(valid_input_index, np.ma.core.MaskedArray)):
        valid_input_index = valid_input_index.filled(False)
    lons = lons[valid_input_index]
    lats = lats[valid_input_index]
    
     
    if nprocs > 1:
        cartesian = _spatial_mp.Cartesian_MP(nprocs)
    else:
        cartesian = _spatial_mp.Cartesian()
    
    
    #Transform reduced swath dataset to cartesian
    swath_size = lons.size
    swath_coords = cartesian.transform_lonlats(lons, lats)
        
    del(lons)
    del(lats)
    
    #Build kd-tree on swath
    if nprocs > 1:        
        resample_kdtree = _spatial_mp.cKDTree_MP(swath_coords,
                                                 nprocs=nprocs)
    else:
        resample_kdtree = sp.cKDTree(swath_coords)
        
    del(swath_coords)
    
    #Query kd-tree with target grid
    #Find nearest neighbours
    target_shape = target_geo_def.shape
    if len(target_shape) > 1:
        grid_coords = \
            target_geo_def.cartesian_coords.reshape(target_shape[0] * 
                                                     target_shape[1], 3)
    else:
        grid_coords = target_geo_def.cartesian_coords
    
    grid_coords = grid_coords[valid_output_index] 
            
    distance_array, index_array = resample_kdtree.query(grid_coords, 
                                                        k=neighbours,
                                                        eps=epsilon,
                                                        distance_upper_bound=
                                                        radius_of_influence)
       
    return valid_input_index, valid_output_index, index_array, distance_array   

def get_sample_from_neighbour_info(resample_type, grid_shape, data, 
                                   valid_input_index, valid_output_index, 
                                   index_array, distance_array=None, 
                                   weight_funcs=None, fill_value=0):
    """Resamples swath based on neighbour info
    
    :Parameters:
    resample_type : {'nn', 'custom'}
        'nn': Use nearest neighbour resampling
        'custom': Resample based on weight_funcs
    grid_shape : (int, int)
        Shape of target grid as (rows, cols)
    lons : numpy array
        1d array of swath lons
    lats : numpy array               
        1d array of swath lats
    valid_index : numpy array
        valid_index from get_neighbour_info
    index_array : numpy array
        index_array from get_neighbour_info
    distance_array : numpy array, optional
        distance_array from get_neighbour_info
        Not needed for 'nn' resample type
    weight_funcs : list function objects or function object, optional       
        List of weight functions f(dist) to use for the weighting 
        of each channel 1 to k.
        If only one channel is resampled weight_funcs is
        a single function object.
        Must be supplied when using 'custom' resample type
    fill_value : {int, None}, optional 
        Set undetermined pixels to this value.
        If fill_value is None a masked array is returned 
        with undetermined pixels masked
        
    :Returns: 
    data : numpy array 
        Swath data resampled to target grid    
    """
    
    #Get size of grid and reduced swath
    swath_size = valid_input_index.sum()
    if len(grid_shape) > 1:
        grid_size = grid_shape[0] * grid_shape[1]
    else:
        grid_size = grid_shape[0]
        
    #Check validity of input
    if not isinstance(data, np.ndarray):
        raise TypeError('data must be numpy array')
    elif valid_input_index.ndim != 1:
        raise TypeError('valid_index must be one dimensional array')
    elif data.shape[0] != valid_input_index.size:
        raise TypeError('Not the same number of datapoints in '
                        'valid_index and data')
    
    valid_types = ('nn', 'custom')
    if not resample_type in valid_types:
        raise TypeError('Invalid resampling type: %s' % resample_type)
    
    if resample_type == 'custom' and weight_funcs is None:
        raise ValueError('weight_funcs must be supplied when using '
                          'custom resampling')
    
    if not isinstance(fill_value, (long, int, float)) and fill_value is not None:
        raise TypeError('fill_value must be number or None')
    
    if index_array.ndim == 1:
        neighbours = 1
    else:
        neighbours = index_array.shape[1]
        if resample_type == 'nn':
            raise ValueError('index_array contains more neighbours than ' 
                             'just the nearest')
    
    #Reduce data    
    new_data = data[valid_input_index]    
    
    #Nearest neighbour resampling should conserve data type
    #Get data type
    conserve_input_data_type = False
    if resample_type == 'nn':
        conserve_input_data_type = True
        input_data_type = new_data.dtype
    
    #Handle masked array input
    is_multi_channel = (new_data.ndim > 1)
    is_masked_data = False
    if np.ma.is_masked(new_data):
        #Add the mask as channels to the dataset
        is_masked_data = True
        new_data = np.column_stack((new_data.data, new_data.mask))
    
    #Prepare weight_funcs argument for handeling mask data
    if weight_funcs is not None and is_masked_data:
        if is_multi_channel:
            weight_funcs = weight_funcs * 2
        else:
            weight_funcs = (weight_funcs,) * 2
    
    #Handle request for masking intead of using fill values        
    use_masked_fill_value = False
    if fill_value is None:
        use_masked_fill_value = True
        fill_value = _get_fill_mask_value(new_data.dtype)
    
    #Resample based on kd-tree query result
    if resample_type == 'nn' or neighbours == 1:
        #Get nearest neighbour using array indexing
        index_mask = (index_array == swath_size)
        new_index_array = np.where(index_mask, 0, index_array)
        result = new_data[new_index_array]
        result[index_mask] = fill_value
    else:
        #Calculate result using weighting
                
        #Get neighbours and masks of valid indices
        ch_neighbour_list = []
        index_mask_list = []
        for i in range(neighbours):
            index_ni = index_array[:, i].copy()
            index_mask_ni = (index_ni == swath_size)
            index_ni[index_mask_ni] = 0
            ch_ni = new_data[index_ni]
            ch_neighbour_list.append(ch_ni) 
            index_mask_list.append(index_mask_ni)
        
        #Calculate weights 
        weight_list = []
        for i in range(neighbours):
            #Set out of bounds distance to 1 in order to avoid numerical Inf
            distance = distance_array[:, i].copy()
            distance[index_mask_list[i]] = 1
            
            if new_data.ndim > 1:
                #Calculate weights for each channel
                weights = []
                for j in range(new_data.shape[1]):                    
                    calc_weight = weight_funcs[j](distance)
                    expanded_calc_weight = np.ones(grid_size) * calc_weight
                    weights.append(expanded_calc_weight)
                weight_list.append(np.column_stack(weights))
            else:
                weights = weight_funcs(distance)
                weight_list.append(weights)
                        
        result = 0
        norm = 0
        
        #Calculate result       
        for i in range(neighbours):   
            #Find invalid indices to be masked of from calculation
            if new_data.ndim > 1:
                inv_index_mask = np.expand_dims(np.invert(index_mask_list[i]), axis=1)
            else:
                inv_index_mask = np.invert(index_mask_list[i])
            
            #Aggregate result and norm
            result += inv_index_mask * ch_neighbour_list[i] * weight_list[i]
            norm += inv_index_mask * weight_list[i]
                                
        #Normalize result and set fillvalue
        new_valid_index = (norm > 0)
        result[new_valid_index] /= norm[new_valid_index]
        result[np.invert(new_valid_index)] = fill_value 
    
    #
    if new_data.ndim > 1:
        full_result = np.ones((grid_size, new_data.shape[1])) * fill_value
    else:
        full_result = np.ones(grid_size) * fill_value
    full_result[valid_output_index] = result 
    result = full_result
    
    #Reshape resampled data to correct shape    
    if new_data.ndim > 1:
        channels = new_data.shape[1]        
        shape = list(grid_shape)
        shape.append(channels)
        result = result.reshape(shape)
        
        #Remap mask channels to create masked output
        if is_masked_data:
            result = _remask_data(result)
    else:
        result = result.reshape(grid_shape)
        
    #Create masking of fill values
    if use_masked_fill_value:
        result = np.ma.masked_equal(result, fill_value)
        
    #Set output data type to input data type if relevant
    if conserve_input_data_type:
        result = result.astype(input_data_type)        
    return result

def _get_fill_mask_value(data_dtype):
    """Returns the maximum value of dtype"""
    
    if issubclass(data_dtype.type, np.floating):
        fill_value = np.finfo(data_dtype.type).max
    elif issubclass(data_dtype.type, np.integer):
        fill_value = np.iinfo(data_dtype.type).max
    else:
        raise TypeError('Type %s is unsupported for masked fill values' %
                        data_dtype.type)
    return fill_value

def _remask_data(data):
    """Interprets half the array as mask for the other half"""
    
    channels = data.shape[2]
    mask = data[:, :, (channels // 2):]            
    #All pixels affected by masked pixels are masked out
    mask = (mask != 0)
    data = np.ma.array(data[:, :, :(channels // 2)], mask=mask)
    if data.shape[2] == 1:
        data = data.reshape(data.shape[:2])
    return data

