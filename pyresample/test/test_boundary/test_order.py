#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 15:48:41 2023

@author: ghiggi
"""
import pytest 
from pyresample.boundary.geographic_boundary import _is_clockwise_order

class Test_Is_Clockwise_Order:
    
    def test_vertical_edges(self):
        """Test with vertical polygon edges."""
        # North Hemisphere
        first_point = (0, 10)
        second_point = (0, 20)
        point_inside = (1, 15)
        assert _is_clockwise_order(first_point, second_point, point_inside)
        assert not _is_clockwise_order(second_point, first_point , point_inside)
       
        # South Hemisphere
        first_point = (0, -20)   
        second_point = (0, -10)
        point_inside = (1, - 15)
        assert _is_clockwise_order(first_point, second_point, point_inside)
        assert not _is_clockwise_order(second_point, first_point , point_inside)
    
    @pytest.mark.parametrize("lon",[-180, -90, 0, 90, 180])
    def test_horizontal_edges(self, lon):
        """Test with horizontal polygon edges."""
        # Point in northern hemisphere 
        first_point = (lon, 0)
        second_point = (lon-10, 0)
        point_inside = (1, 15) 
        assert _is_clockwise_order(first_point, second_point, point_inside)
        assert not _is_clockwise_order(second_point, first_point , point_inside)
       
        # Point in northern hemisphere 
        first_point = (lon, 0)
        second_point = (lon+10, 0)
        point_inside = (1, -15) 
        assert _is_clockwise_order(first_point, second_point, point_inside)
        assert not _is_clockwise_order(second_point, first_point , point_inside)
    
    def test_diagonal_edges(self):
        """Test with diagonal polygon edges."""
        point_inside = (20, 15) 
        
        # Edge toward right (above point) --> clockwise 
        first_point = (0, 0)
        second_point = (20, 20)
        assert _is_clockwise_order(first_point, second_point, point_inside)
        assert not _is_clockwise_order(second_point, first_point , point_inside)
        
        # Edge toward right (below point) --> not clockwise
        first_point = (0, 0)
        second_point = (20, 10)
        assert not _is_clockwise_order(first_point, second_point, point_inside)
                
    def test_polygon_edges_on_antimeridian(self):
        """Test polygon edges touching the antimeridian edges."""
        ## Right side of antimeridian 
        # North Hemisphere
        first_point = (-180, 10)
        second_point = (-180, 20)
        point_inside = (-179, 15)
        assert _is_clockwise_order(first_point, second_point, point_inside)
        assert not _is_clockwise_order(second_point, first_point , point_inside)
       
        # South Hemisphere
        first_point = (-180, -20)   
        second_point = (-180, -10)
        point_inside = (-179, - 15)
        assert _is_clockwise_order(first_point, second_point, point_inside)
        assert not _is_clockwise_order(second_point, first_point , point_inside)
        
        ## Left side of antimeridian
        # North Hemisphere
        first_point = (-180, 20)
        second_point = (-180, 10)
        point_inside = (179, 15)
        assert _is_clockwise_order(first_point, second_point, point_inside)
        assert not _is_clockwise_order(second_point, first_point , point_inside)
       
        # South Hemisphere
        first_point = (-180, -10)   
        second_point = (-180,-20)
        point_inside = (179, - 15)
        assert _is_clockwise_order(first_point, second_point, point_inside)
        assert not _is_clockwise_order(second_point, first_point , point_inside)
        
    @pytest.mark.parametrize("lon",[179, 180, -180, -179])
    def test_polygon_around_antimeridian(self, lon):
        """Test polygon edges crossing antimeridian."""
        # North Hemisphere
        first_point = (170, 10)  
        second_point = (-170, 10)
        point_inside = (lon, 5)
        assert _is_clockwise_order(first_point, second_point, point_inside)
        assert not _is_clockwise_order(second_point, first_point , point_inside)
       
        # South Hemisphere
        first_point = (-170, -10)   
        second_point = (170, -10)
        point_inside = (lon, - 5)
        assert _is_clockwise_order(first_point, second_point, point_inside)
        assert not _is_clockwise_order(second_point, first_point , point_inside)
    
    @pytest.mark.parametrize("lon_pole",[-180, 90, 45, 0, 45, 90, 180])
    @pytest.mark.parametrize("lat",[85, 0, -85])
    def test_polygon_around_north_pole(self, lon_pole, lat):
        """Test polygon edges around north pole (right to left)."""
        point_inside = (lon_pole, 90)
        first_point = (0, lat)  
        second_point = (-10, lat)       
        assert _is_clockwise_order(first_point, second_point, point_inside)
        assert not _is_clockwise_order(second_point, first_point , point_inside)
         
    @pytest.mark.parametrize("lon_pole",[-180, 90, 45, 0, 45, 90, 180])
    @pytest.mark.parametrize("lat",[85, 0, -85])
    def test_polygon_around_south_pole(self, lon_pole, lat):
        """Test polygon edges around south pole (left to right)."""
        point_inside = (lon_pole, -90)
        first_point = (0, lat)  
        second_point = (10, lat)       
        assert _is_clockwise_order(first_point, second_point, point_inside)
        assert not _is_clockwise_order(second_point, first_point , point_inside)
        
        


 