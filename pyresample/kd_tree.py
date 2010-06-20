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

"""Handles reprojection of geolocated data. Several types of resampling are supported"""

import types

import numpy as np
import scipy.spatial as sp

import geometry
import data_reduce
import _spatial_mp
        
        
def resample_nearest(source_geo_def, data, target_geo_def,
                     radius_of_influence, epsilon=0,
                     fill_value=0, reduce_data=True, nprocs=1):
    """Resamples data using kd-tree nearest neighbour approach

    :Parameters:
    source_geo_def : object
        Geometry definition of source
    data : numpy array               
        1d array of single channel data points or
        (source_size, k) array of k channels of datapoints
    target_geo_def : object
        Geometry definition of target
    radius_of_influence : float 
        Cut off distance in meters
    epsilon : float, optional
        Allowed uncertainty in meters. Increasing uncertainty
        reduces execution time
    fill_value : {int, None}, optional 
            Set undetermined pixels to this value.
            If fill_value is None a masked array is returned 
            with undetermined pixels masked    
    reduce_data : bool, optional
        Perform initial coarse reduction of source dataset in order
        to reduce execution time
    nprocs : int, optional
        Number of processor cores to be used
               
    :Returns: 
    data : numpy array 
        Source data resampled to target geometry
    """
    
    return _resample(source_geo_def, data, target_geo_def, 'nn',
                     radius_of_influence, neighbours=1,
                     epsilon=epsilon, fill_value=fill_value,
                     reduce_data=reduce_data, nprocs=nprocs)

def resample_gauss(source_geo_def, data, target_geo_def,
                   radius_of_influence, sigmas, neighbours=8, epsilon=0,
                   fill_value=0, reduce_data=True, nprocs=1):
    """Resamples data using kd-tree gaussian weighting neighbour approach

    :Parameters:
    source_geo_def : object
        Geometry definition of source
    data : numpy array               
        1d array of single channel data points or
        (source_size, k) array of k channels of datapoints
    target_geo_def : object
        Geometry definition of target
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
        Perform initial coarse reduction of source dataset in order
        to reduce execution time
    nprocs : int, optional
        Number of processor cores to be used
    
    :Returns: 
    data : numpy array 
        Source data resampled to target geometry
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
    """Resamples data using kd-tree custom radial weighting neighbour approach

    :Parameters:
    source_geo_def : object
        Geometry definition of source
    data : numpy array               
        1d array of single channel data points or
        (source_size, k) array of k channels of datapoints
    target_geo_def : object
        Geometry definition of target
    radius_of_influence : float 
        Cut off distance in meters
    weight_funcs : list of function objects or function object       
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
        Perform initial coarse reduction of source dataset in order
        to reduce execution time
    nprocs : int, optional
        Number of processor cores to be used
    
    :Returns: 
    data : numpy array 
        Source data resampled to target geometry
    """
    
    if data.ndim > 1:
        for weight_func in weight_funcs:
            if not isinstance(weight_func, types.FunctionType):
                raise TypeError('weight_func must be function object')
    elif not isinstance(weight_funcs, types.FunctionType):
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

    valid_input_index, valid_output_index, index_array, distance_array = \
                                 get_neighbour_info(source_geo_def, 
                                                    target_geo_def, 
                                                    radius_of_influence, 
                                                    neighbours=neighbours, 
                                                    epsilon=epsilon, 
                                                    reduce_data=reduce_data, 
                                                    nprocs=nprocs)
    
    return get_sample_from_neighbour_info(resample_type, 
                                          target_geo_def.shape, 
                                          data, valid_input_index, 
                                          valid_output_index, index_array, 
                                          distance_array=distance_array, 
                                          weight_funcs=weight_funcs, 
                                          fill_value=fill_value)
    
def get_neighbour_info(source_geo_def, target_geo_def, radius_of_influence, 
                       neighbours=8, epsilon=0, reduce_data=True, nprocs=1):    
    """Returns neighbour info
    
    :Parameters:
    source_geo_def : object
        Geometry definition of source
    target_geo_def : object
        Geometry definition of target
    radius_of_influence : float 
        Cut off distance in meters
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
        Perform initial coarse reduction of source dataset in order
        to reduce execution time
    nprocs : int, optional
        Number of processor cores to be used
        
    :Returns:
    (valid_input_index, valid_output_index, 
    index_array, distance_array) : tuple of numpy arrays
        Neighbour resampling info
    """

    #Check validity of input
    if not isinstance(source_geo_def, geometry.BaseDefinition):
        raise typeError('source_geo_def must be of geometry type')
    elif not isinstance(target_geo_def, geometry.BaseDefinition):
        raise typeError('target_geo_def must be of geometry type')    
    elif not isinstance(radius_of_influence, (long, int, float)):
        raise TypeError('radius_of_influence must be number')
    elif not isinstance(neighbours, int):
        raise TypeError('neighbours must be integer')
    elif not isinstance(epsilon, (long, int, float)):
        raise TypeError('epsilon must be number')
    
    s_lons = source_geo_def.lons.ravel()
    s_lats = source_geo_def.lats.ravel()
    
    t_lons = target_geo_def.lons.ravel()
    t_lats = target_geo_def.lats.ravel()

    
    #Find invalid data points 
    valid_data = ((s_lons >= -180) * (s_lons <= 180) * 
                  (s_lats <= 90) * (s_lats >= -90))
    valid_input_index = np.ones(source_geo_def.size).astype(np.bool)
    valid_output_index = np.ones(target_geo_def.size).astype(np.bool)
    
    if reduce_data:
        if (isinstance(source_geo_def, geometry.CoordinateDefinition) and 
            isinstance(target_geo_def, (geometry.GridDefinition, 
                                       geometry.AreaDefinition))) or \
           (isinstance(source_geo_def, (geometry.GridDefinition, 
                                        geometry.AreaDefinition)) and
            isinstance(target_geo_def, (geometry.GridDefinition, 
                                        geometry.AreaDefinition))):
            #Resampling from swath to grid or from grid to grid
            valid_input_index = \
                data_reduce.get_valid_index_from_lonlat_grid(
                                            target_geo_def.lons,
                                            target_geo_def.lats, 
                                            s_lons, s_lats, 
                                            radius_of_influence)
        elif isinstance(source_geo_def, (geometry.GridDefinition, 
                                         geometry.AreaDefinition)) and \
             isinstance(target_geo_def, geometry.CoordinateDefinition):
            #Resampling from grid to swath
            valid_output_index = \
                data_reduce.get_valid_index_from_lonlat_grid(
                                            source_geo_def.lons,
                                            source_geo_def.lats, 
                                            t_lons, 
                                            t_lats, 
                                            radius_of_influence)
            valid_output_index = valid_output_index.astype(np.bool)
        else:
            #Resampling from swath to swath. Do nothing
            pass
    
    valid_out = ((t_lons >= -180) * (t_lons <= 180) * 
                  (t_lats <= 90) * (t_lats >= -90))
    
    #Find valid output points
    valid_output_index = (valid_output_index & valid_out)
    
    #Find valid data points    
    valid_input_index = (valid_data & valid_input_index)
    if(isinstance(valid_input_index, np.ma.core.MaskedArray)):
        valid_input_index = valid_input_index.filled(False)
    s_lons = s_lons[valid_input_index]
    s_lats = s_lats[valid_input_index]
    
     
    if nprocs > 1:
        cartesian = _spatial_mp.Cartesian_MP(nprocs)
    else:
        cartesian = _spatial_mp.Cartesian()
    
    
    #Transform reduced swath dataset to cartesian
    input_size = s_lons.size
    input_coords = cartesian.transform_lonlats(s_lons, s_lats)
        
    del(s_lons)
    del(s_lats)
    
    #Build kd-tree on input
    if nprocs > 1:        
        resample_kdtree = _spatial_mp.cKDTree_MP(input_coords,
                                                 nprocs=nprocs)
    else:
        resample_kdtree = sp.cKDTree(input_coords)
        
    del(input_coords)
    
    #Query kd-tree with target coords
    #Find nearest neighbours
    target_shape = target_geo_def.shape
    if len(target_shape) > 1:
        output_coords = \
            target_geo_def.cartesian_coords.reshape(target_shape[0] * 
                                                    target_shape[1], 3)
    else:
        output_coords = target_geo_def.cartesian_coords
    
    output_coords = output_coords[valid_output_index] 
            
    distance_array, index_array = resample_kdtree.query(output_coords, 
                                                        k=neighbours,
                                                        eps=epsilon,
                                                        distance_upper_bound=
                                                        radius_of_influence)
       
    return valid_input_index, valid_output_index, index_array, distance_array   

def get_sample_from_neighbour_info(resample_type, output_shape, data, 
                                   valid_input_index, valid_output_index, 
                                   index_array, distance_array=None, 
                                   weight_funcs=None, fill_value=0):
    """Resamples swath based on neighbour info
    
    :Parameters:
    resample_type : {'nn', 'custom'}
        'nn': Use nearest neighbour resampling
        'custom': Resample based on weight_funcs
    output_shape : (int, int)
        Shape of output as (rows, cols)
    valid_input_index : numpy array
        valid_input_index from get_neighbour_info
    valid_output_index : numpy array
        valid_output_index from get_neighbour_info
    index_array : numpy array
        index_array from get_neighbour_info
    distance_array : numpy array, optional
        distance_array from get_neighbour_info
        Not needed for 'nn' resample type
    weight_funcs : list of function objects or function object, optional       
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
        Source data resampled to target geometry
    """
    
    #Get size of output and reduced input
    input_size = valid_input_index.sum()
    if len(output_shape) > 1:
        output_size = output_shape[0] * output_shape[1]
    else:
        output_size = output_shape[0]
        
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
        index_mask = (index_array == input_size)
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
            index_mask_ni = (index_ni == input_size)
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
                    expanded_calc_weight = np.ones(output_size) * calc_weight
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
        full_result = np.ones((output_size, new_data.shape[1])) * fill_value
    else:
        full_result = np.ones(output_size) * fill_value
    full_result[valid_output_index] = result 
    result = full_result
    
    #Reshape resampled data to correct shape    
    if new_data.ndim > 1:
        channels = new_data.shape[1]        
        shape = list(output_shape)
        shape.append(channels)
        result = result.reshape(shape)
        
        #Remap mask channels to create masked output
        if is_masked_data:
            result = _remask_data(result)
    else:
        result = result.reshape(output_shape)
        
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

