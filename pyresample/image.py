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

"""Handles gridded images"""

import numpy as np

import geometry, grid, swath 


class ImageContainer(object):
    """Holds image with area definition. 
    Allows indexing with linesample arrays.
    
    :Parameters:
    image_data : numpy array 
        Image data
    area_def : object 
        Area definition as AreaDefinition object
    
    :Attributes:
    image_data : numpy array 
        Image data
    area_def : object 
        Area definition as AreaDefinition object    
    """
        
    def __init__(self, image_data, area_def):
        if not isinstance(image_data, (np.ndarray, np.ma.core.MaskedArray)):
            raise TypeError('image_data must be either an ndarray'
                            ' or a masked array')
        if not isinstance(area_def, geometry.AreaDefinition):
            raise TypeError('area_def must be of type AreaDefinition')

        self.image_data = image_data
        self.area_def = area_def        
        
    def __str__(self):
        return 'Image:\n %s'%self.image_data.__str__()

    def __repr__(self): 
        return self.image_data.__repr__()
        
    def resample(self, *args, **kwargs):
        raise NotImplementedError('Method "resample" is not implemented ' 
                                  'in class %s' % self.__class__.__name__)

    def get_array_from_linesample(self, row_indices, col_indices,
                                  fill_value=0):
        """Samples from image based on index arrays.

        :Parameters:
        row_indices : numpy array
            Row indices. Dimensions must match col_indices
        col_indices : numpy array of
            Col indices. Dimensions must match row_indices
        fill_value : {int, None} optional 
            Set undetermined pixels to this value.
            If fill_value is None a masked array is returned         
        
        :Returns: 
        image_data : numpy_array
            Resampled image data
        """
        
        return grid.get_image_from_linesample(row_indices, col_indices,
                                              self.image_data, fill_value)


class ImageContainerQuick(ImageContainer):
    """Holds image with area definition. '
    Allows quick resampling within area.
    
    :Parameters:
    image_data : numpy array 
        Image data
    area_def : object 
        Area definition as AreaDefinition object
    
    :Attributes:
    image_data : numpy array 
        Image data
    area_def : object 
        Area definition as AreaDefinition object    
    """

    def resample(self, target_area_def, fill_value=0, nprocs=1):
        """Resamples image to area definition using nearest neighbour 
        approach in projection coordinates
        
        :Parameters:
        target_area_def : object 
            Target area definition as AreaDefinition object
        fill_value : {int, None} optional 
            Set undetermined pixels to this value.
            If fill_value is None a masked array is returned 
            with undetermined pixels masked
        nprocs : int, optional 
            Number of processor cores to be used
        
        :Returns: 
        image_container : object
            ImageContainer object of resampled area   
        """
        
        if not isinstance(target_area_def, geometry.AreaDefinition):
            raise TypeError('target_area_def must be of type '
                            'geometry.AreaDefinition')        
        
        resampled_image = grid.get_resampled_image(target_area_def,
                                                   self.area_def,
                                                   self.image_data,
                                                   fill_value,
                                                   nprocs)

        return ImageContainerQuick(resampled_image, target_area_def)
    

class ImageContainerNearest(ImageContainer):
    """Holds image with area definition. 
    Allows nearest neighbour resampling within area.
    
    :Parameters:
    image_data : numpy array 
        Image data
    area_def : object 
        Area definition as AreaDefinition object
    radius_of_influence : float 
        Cut off distance in meters    
    epsilon : float, optional
        Allowed uncertainty in meters. Increasing uncertainty
        reduces execution time
    
    :Attributes:
    image_data : numpy array 
        Image data
    area_def : object 
        Area definition as AreaDefinition object    
    """

    def __init__(self, image_data, area_def, radius_of_influence, epsilon=0):
        super(ImageContainerNearest, self).__init__(image_data, area_def)
        self.radius_of_influence = radius_of_influence
        self.epsilon = epsilon
        
    def resample(self, target_area_def, fill_value=0, nprocs=1):
        """Resamples image to area definition using nearest neighbour 
        approach
        
        :Parameters:
        target_area_def : object 
            Target area definition as AreaDefinition object        
        fill_value : {int, None} optional 
            Set undetermined pixels to this value.
            If fill_value is None a masked array is returned 
            with undetermined pixels masked
        nprocs : int, optional 
            Number of processor cores to be used
        
        :Returns: 
        image_container : object
            ImageContainer object of resampled area   
        """
        
        #lons, lats = self.area_def.get_lonlats(nprocs)
        if self.image_data.ndim > 2:
            image_data = self.image_data.reshape(self.image_data.shape[0] * 
                                                 self.image_data.shape[1], 
                                                 self.image_data.shape[2])
        else:
            image_data = self.image_data.ravel()
                   
        resampled_image = swath.resample_nearest(self.area_def, 
                                                 image_data, 
                                                 target_area_def,
                                                 self.radius_of_influence, 
                                                 epsilon=self.epsilon,
                                                 fill_value=fill_value, 
                                                 nprocs=nprocs)
        return ImageContainerNearest(resampled_image, target_area_def, 
                                     self.radius_of_influence, self.epsilon)