# pyresample, Resampling of remote sensing image data in python
#
# Copyright (C) 2010, 2015  Esben S. Nielsen
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Handles resampling of images with assigned geometry definitions"""

from __future__ import absolute_import

import numpy as np

from pyresample import geometry, grid, kd_tree, bilinear


class ImageContainer(object):

    """Holds image with geometry definition.
    Allows indexing with linesample arrays.

    Parameters
    ----------
    image_data : numpy array
        Image data
    geo_def : object
        Geometry definition
    fill_value : int or None, optional
        Set undetermined pixels to this value.
        If fill_value is None a masked array is returned
        with undetermined pixels masked
    nprocs : int, optional
        Number of processor cores to be used

    Attributes
    ----------
    image_data : numpy array
        Image data
    geo_def : object
        Geometry definition
    fill_value : int or None
        Resample result fill value
    nprocs : int
        Number of processor cores to be used for geometry operations
    """

    def __init__(self, image_data, geo_def, fill_value=0, nprocs=1):
        if type(geo_def).__name__ == "DynamicAreaDefinition":
            geo_def = geo_def.freeze()
        if not isinstance(image_data, (np.ndarray, np.ma.core.MaskedArray)):
            raise TypeError('image_data must be either an ndarray'
                            ' or a masked array')
        elif ((image_data.ndim > geo_def.ndim + 1) or
              (image_data.ndim < geo_def.ndim)):
            raise ValueError(('Unexpected number of dimensions for '
                              'image_data: %s') % image_data.ndim)
        for i, size in enumerate(geo_def.shape):
            if image_data.shape[i] != size:
                raise ValueError(('Size mismatch for image_data. Expected '
                                  'size %s for dimension %s and got %s') %
                                 (size, i, image_data.shape[i]))

        self.shape = geo_def.shape
        self.size = geo_def.size
        self.ndim = geo_def.ndim
        self.image_data = image_data
        if image_data.ndim > geo_def.ndim:
            self.channels = image_data.shape[-1]
        else:
            self.channels = 1
        self.geo_def = geo_def
        self.fill_value = fill_value
        self.nprocs = nprocs

    def __str__(self):
        return 'Image:\n %s' % self.image_data.__str__()

    def __repr__(self):
        return self.image_data.__repr__()

    def resample(self, target_geo_def):
        """Base method for resampling"""

        raise NotImplementedError('Method "resample" is not implemented '
                                  'in class %s' % self.__class__.__name__)

    def get_array_from_linesample(self, row_indices, col_indices):
        """Samples from image based on index arrays.

        Parameters
        ----------
        row_indices : numpy array
            Row indices. Dimensions must match col_indices
        col_indices : numpy array
            Col indices. Dimensions must match row_indices

        Returns
        -------
        image_data : numpy_array
            Resampled image data
        """

        if self.geo_def.ndim != 2:
            raise TypeError('Resampling from linesamples only makes sense '
                            'on 2D data')

        return grid.get_image_from_linesample(row_indices, col_indices,
                                              self.image_data,
                                              self.fill_value)

    def get_array_from_neighbour_info(self, *args, **kwargs):
        """Base method for resampling from preprocessed data."""

        raise NotImplementedError('Method "get_array_from_neighbour_info" is '
                                  'not implemented in class %s' %
                                  self.__class__.__name__)


class ImageContainerQuick(ImageContainer):

    """Holds image with area definition. '
    Allows quick resampling within area.

    Parameters
    ----------
    image_data : numpy array
        Image data
    geo_def : object
        Area definition as AreaDefinition object
    fill_value : int or None, optional
        Set undetermined pixels to this value.
        If fill_value is None a masked array is returned
        with undetermined pixels masked
    nprocs : int, optional
        Number of processor cores to be used for geometry operations
    segments : int or None
        Number of segments to use when resampling.
        If set to None an estimate will be calculated

    Attributes
    ----------
    image_data : numpy array
        Image data
    geo_def : object
        Area definition as AreaDefinition object
    fill_value : int or None
        Resample result fill value
        If fill_value is None a masked array is returned
        with undetermined pixels masked
    nprocs : int
        Number of processor cores to be used
    segments : int or None
        Number of segments to use when resampling
    """

    def __init__(self, image_data, geo_def, fill_value=0, nprocs=1,
                 segments=None):
        if not isinstance(geo_def, geometry.AreaDefinition):
            raise TypeError('area_def must be of type '
                            'geometry.AreaDefinition')
        super(ImageContainerQuick, self).__init__(image_data, geo_def,
                                                  fill_value=fill_value,
                                                  nprocs=nprocs)
        self.segments = segments

    def resample(self, target_area_def):
        """Resamples image to area definition using nearest neighbour
        approach in projection coordinates.

        Parameters
        ----------
        target_area_def : object
            Target area definition as AreaDefinition object

        Returns
        -------
        image_container : object
            ImageContainerQuick object of resampled area
        """

        resampled_image = grid.get_resampled_image(target_area_def,
                                                   self.geo_def,
                                                   self.image_data,
                                                   fill_value=self.fill_value,
                                                   nprocs=self.nprocs,
                                                   segments=self.segments)

        return ImageContainerQuick(resampled_image, target_area_def,
                                   fill_value=self.fill_value,
                                   nprocs=self.nprocs, segments=self.segments)


class ImageContainerNearest(ImageContainer):

    """Holds image with geometry definition.
    Allows nearest neighbour to new geometry definition.

    Parameters
    ----------
    image_data : numpy array
        Image data
    geo_def : object
        Geometry definition
    radius_of_influence : float
        Cut off distance in meters
    epsilon : float, optional
        Allowed uncertainty in meters. Increasing uncertainty
        reduces execution time
    fill_value : int or None, optional
        Set undetermined pixels to this value.
        If fill_value is None a masked array is returned
        with undetermined pixels masked
    reduce_data : bool, optional
        Perform coarse data reduction before resampling in order
        to reduce execution time
    nprocs : int, optional
        Number of processor cores to be used for geometry operations
    segments : int or None
        Number of segments to use when resampling.
        If set to None an estimate will be calculated

    Attributes
    ----------

    image_data : numpy array
        Image data
    geo_def : object
        Geometry definition
    radius_of_influence : float
        Cut off distance in meters
    epsilon : float
        Allowed uncertainty in meters
    fill_value : int or None
        Resample result fill value
    reduce_data : bool
        Perform coarse data reduction before resampling
    nprocs : int
        Number of processor cores to be used
    segments : int or None
        Number of segments to use when resampling
    """

    def __init__(self, image_data, geo_def, radius_of_influence, epsilon=0,
                 fill_value=0, reduce_data=True, nprocs=1, segments=None):
        super(ImageContainerNearest, self).__init__(image_data, geo_def,
                                                    fill_value=fill_value,
                                                    nprocs=nprocs)
        self.radius_of_influence = radius_of_influence
        self.epsilon = epsilon
        self.reduce_data = reduce_data
        self.segments = segments

    def resample(self, target_geo_def):
        """Resamples image to area definition using nearest neighbour
        approach

        Parameters
        ----------
        target_geo_def : object
            Target geometry definition

        Returns
        -------
        image_container : object
            ImageContainerNearest object of resampled geometry
        """

        if self.image_data.ndim > 2 and self.ndim > 1:
            image_data = self.image_data.reshape(self.image_data.shape[0] *
                                                 self.image_data.shape[1],
                                                 self.image_data.shape[2])
        else:
            image_data = self.image_data.ravel()

        resampled_image = \
            kd_tree.resample_nearest(self.geo_def,
                                     image_data,
                                     target_geo_def,
                                     self.radius_of_influence,
                                     epsilon=self.epsilon,
                                     fill_value=self.fill_value,
                                     nprocs=self.nprocs,
                                     reduce_data=self.reduce_data,
                                     segments=self.segments)
        return ImageContainerNearest(resampled_image, target_geo_def,
                                     self.radius_of_influence,
                                     epsilon=self.epsilon,
                                     fill_value=self.fill_value,
                                     reduce_data=self.reduce_data,
                                     nprocs=self.nprocs,
                                     segments=self.segments)


class ImageContainerBilinear(ImageContainer):

    """Holds image with geometry definition.
    Allows bilinear to new geometry definition.

    Parameters
    ----------
    image_data : numpy array
        Image data
    geo_def : object
        Geometry definition
    radius_of_influence : float
        Cut off distance in meters
    epsilon : float, optional
        Allowed uncertainty in meters. Increasing uncertainty
        reduces execution time
    fill_value : int or None, optional
        Set undetermined pixels to this value.
        If fill_value is None a masked array is returned
        with undetermined pixels masked
    reduce_data : bool, optional
        Perform coarse data reduction before resampling in order
        to reduce execution time
    nprocs : int, optional
        Number of processor cores to be used for geometry operations
    segments : int or None
        Number of segments to use when resampling.
        If set to None an estimate will be calculated

    Attributes
    ----------

    image_data : numpy array
        Image data
    geo_def : object
        Geometry definition
    radius_of_influence : float
        Cut off distance in meters
    epsilon : float
        Allowed uncertainty in meters
    fill_value : int or None
        Resample result fill value
    reduce_data : bool
        Perform coarse data reduction before resampling
    nprocs : int
        Number of processor cores to be used
    segments : int or None
        Number of segments to use when resampling
    """

    def __init__(self, image_data, geo_def, radius_of_influence, epsilon=0,
                 fill_value=0, reduce_data=False, nprocs=1, segments=None,
                 neighbours=32):
        super(ImageContainerBilinear, self).__init__(image_data, geo_def,
                                                     fill_value=fill_value,
                                                     nprocs=nprocs)
        self.radius_of_influence = radius_of_influence
        self.epsilon = epsilon
        self.reduce_data = reduce_data
        self.segments = segments
        self.neighbours = neighbours

    def resample(self, target_geo_def):
        """Resamples image to area definition using bilinear approach

        Parameters
        ----------
        target_geo_def : object
            Target geometry definition

        Returns
        -------
        image_container : object
            ImageContainerBilinear object of resampled geometry
        """

        if self.image_data.ndim > 2 and self.ndim > 1:
            image_data = self.image_data.reshape(self.image_data.shape[0] *
                                                 self.image_data.shape[1],
                                                 self.image_data.shape[2])
        else:
            image_data = self.image_data.ravel()

        try:
            mask = image_data.mask.copy()
            image_data = image_data.data.copy()
            image_data[mask] = np.nan
        except AttributeError:
            pass

        resampled_image = \
            bilinear.resample_bilinear(image_data,
                                       self.geo_def,
                                       target_geo_def,
                                       radius=self.radius_of_influence,
                                       neighbours=self.neighbours,
                                       epsilon=self.epsilon,
                                       fill_value=self.fill_value,
                                       nprocs=self.nprocs,
                                       reduce_data=self.reduce_data,
                                       segments=self.segments)
        try:
            resampled_image = resampled_image.reshape(target_geo_def.shape)
        except ValueError:
            # The input data was 3D
            shp = target_geo_def.shape
            new_shp = [shp[0], shp[1], image_data.shape[-1]]
            resampled_image = resampled_image.reshape(new_shp)

        return ImageContainerBilinear(resampled_image, target_geo_def,
                                      self.radius_of_influence,
                                      epsilon=self.epsilon,
                                      fill_value=self.fill_value,
                                      reduce_data=self.reduce_data,
                                      nprocs=self.nprocs,
                                      segments=self.segments)
