#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 15:26:08 2023

@author: andy
"""

import numpy as np

def SphericalAberration(x_p, y_p, x_slope, y_slope):
    Bx = x_p.getCoefficient([0, 3, 0, 0])*x_slope**3 + \
         x_p.getCoefficient([0, 2, 0, 1])*x_slope**2*y_slope**1 + \
         x_p.getCoefficient([0, 1, 0, 2])*x_slope**1*y_slope**2 + \
         x_p.getCoefficient([0, 0, 0, 3])*y_slope**3
    
    By = y_p.getCoefficient([0, 3, 0, 0])*x_slope**3 + \
         y_p.getCoefficient([0, 2, 0, 1])*x_slope**2*y_slope**1 + \
         y_p.getCoefficient([0, 1, 0, 2])*x_slope**1*y_slope**2 + \
         y_p.getCoefficient([0, 0, 0, 3])*y_slope**3

    return Bx, By

def Coma(x_p, y_p, x_pos, y_pos, x_slope, y_slope):
    Fx = x_p.getCoefficient([1, 2, 0, 0])*x_pos*x_slope**2 + \
         x_p.getCoefficient([1, 1, 0, 1])*x_pos*x_slope*y_slope + \
         x_p.getCoefficient([1, 0, 0, 2])*x_pos*y_slope**2 + \
         x_p.getCoefficient([0, 2, 1, 0])*x_slope**2*y_pos + \
         x_p.getCoefficient([0, 1, 1, 1])*x_slope*y_pos*y_slope + \
         x_p.getCoefficient([0, 0, 1, 2])*y_pos*y_slope**2

    Fy = y_p.getCoefficient([1, 2, 0, 0])*x_pos*x_slope**2 + \
         y_p.getCoefficient([1, 1, 0, 1])*x_pos*x_slope*y_slope + \
         y_p.getCoefficient([1, 0, 0, 2])*x_pos*y_slope**2 + \
         y_p.getCoefficient([0, 2, 1, 0])*x_slope**2*y_pos + \
         y_p.getCoefficient([0, 1, 1, 1])*x_slope*y_pos*y_slope + \
         y_p.getCoefficient([0, 0, 1, 2])*y_pos*y_slope**2

    return Fx, Fy

def FieldCurvature_and_Astigmatism(x_p, y_p, x_pos, y_pos, x_slope, y_slope):
    CDx = x_p.getCoefficient([2, 1, 0, 0])*x_pos**2*x_slope + \
          x_p.getCoefficient([1, 1, 1, 0])*x_pos*x_slope*y_pos + \
          x_p.getCoefficient([0, 1, 2, 0])*x_slope*y_pos**2 + \
          x_p.getCoefficient([2, 0, 0, 1])*x_pos**2*y_slope + \
          x_p.getCoefficient([1, 0, 1, 1])*x_pos*y_pos*y_slope + \
          x_p.getCoefficient([0, 0, 2, 1])*y_pos**2*y_slope

    CDy = y_p.getCoefficient([2, 1, 0, 0])*x_pos**2*x_slope + \
          y_p.getCoefficient([1, 1, 1, 0])*x_pos*x_slope*y_pos + \
          y_p.getCoefficient([0, 1, 2, 0])*x_slope*y_pos**2 + \
          y_p.getCoefficient([2, 0, 0, 1])*x_pos**2*y_slope + \
          y_p.getCoefficient([1, 0, 1, 1])*x_pos*y_pos*y_slope + \
          y_p.getCoefficient([0, 0, 2, 1])*y_pos**2*y_slope
    
    return CDx, CDy

def Distortion(x_p, y_p, x_pos, y_pos):
    Ex = x_p.getCoefficient([3, 0, 0, 0])*x_pos**3 + \
         x_p.getCoefficient([2, 0, 1, 0])*x_pos**2*y_pos**1 + \
         x_p.getCoefficient([1, 0, 2, 0])*x_pos**1*y_pos**2 + \
         x_p.getCoefficient([0, 0, 3, 0])*y_pos**3
    
    Ey = y_p.getCoefficient([3, 0, 0, 0])*x_pos**3 + \
         y_p.getCoefficient([2, 0, 1, 0])*x_pos**2*y_pos**1 + \
         y_p.getCoefficient([1, 0, 2, 0])*x_pos**1*y_pos**2 + \
         y_p.getCoefficient([0, 0, 3, 0])*y_pos**3
    
    return Ex, Ey

def spot_diagram(x_f, ax, color, num_spots = 5, x_lim = [-1e-4, 1e-4], y_lim = [-1e-4, 1e-4], semi_angle = 1e-3, count = 0, ):
    
    x_spots = np.linspace(x_lim[0], x_lim[1], num_spots, endpoint = True)
    y_spots = np.linspace(y_lim[0], y_lim[1], num_spots, endpoint = True)
    
    x_radians = np.cos(np.linspace(0, 2*np.pi, 100))*semi_angle
    y_radians = np.sin(np.linspace(0, 2*np.pi, 100))*semi_angle
    x_slopes = np.tan(x_radians)
    y_slopes = np.tan(y_radians)
    
    for x in x_spots:
        for y in y_spots:
            
            x_pos = np.ones(len(x_slopes))*x
            y_pos = np.ones(len(y_slopes))*y
            x0 = np.array([x_pos, x_slopes, y_pos, y_slopes]).T
            xf = np.zeros(x0.shape)
            
            for idx, row in enumerate(x0):
                 xf[idx, :] = x_f.eval(row)
                
            if count == 0:
                ax[1].plot(np.average(xf[:, 0])*1e3, np.average(xf[:, 2])*1e3, 'o', markersize = 4)
                
            ax[1].plot(xf[:, 0]*1e3, xf[:, 2]*1e3, color = color)
                 
    return ax

def spherical_aberration_diagram(x_f, ax, color, semi_angle = 5e-3):
    
    x_radians = np.cos(np.linspace(0, 2*np.pi, 100))*semi_angle
    y_radians = np.sin(np.linspace(0, 2*np.pi, 100))*semi_angle
    x_slopes = np.tan(x_radians)
    y_slopes = np.tan(y_radians)
    
    Bx, By = SphericalAberration(x_f[0], x_f[2], x_slopes, y_slopes)
    
    ax.plot(Bx*1e3, By*1e3, color = color)
    
    return ax
    
def coma_diagram(x_f, ax, color, num_spots = 4, outer_spot_pos = 1e-4, angle_of_spots =  (1/4)*np.pi, semi_angle = 5e-3):
    
    x_radians = np.cos(np.linspace(0, 2*np.pi, 100))*semi_angle
    y_radians = np.sin(np.linspace(0, 2*np.pi, 100))*semi_angle
    x_slopes = np.tan(x_radians)
    y_slopes = np.tan(y_radians)
    
    #Create an array of the circle centres along the x axis
    pos_x = np.linspace(0, outer_spot_pos, num_spots, endpoint = True)
    pos_y = np.linspace(0, 0, num_spots)
    
    #Join them 
    A = np.array([pos_x, pos_y])
    
    #Make rotation matrix
    c, s = np.cos(angle_of_spots), np.sin(angle_of_spots)
    R = np.array(((c, -s), (s, c)))

    #Rotate A
    A = (R @ A).T

    for row in A:
        x_pos = np.ones(100)*row[0]
        y_pos = np.ones(100)*row[1]
        Fx, Fy = Coma(x_f[0], x_f[2], x_pos, y_pos, x_slopes, y_slopes)
        
        ax.plot(Fx*1e3, Fy*1e3, color = color)

    return ax

def fieldcurvature_and_astigmatism_diagram(x_f, ax, color, num_spots = 3, outer_spot_pos = 1e-3, angle_of_spots =  (1/4)*np.pi, semi_angle = 5e-3):
    
    semi_angles = np.linspace(0, semi_angle, num_spots, endpoint = True)
    for semi_angle in semi_angles:
        x_pos = np.ones(100)*outer_spot_pos 
        y_pos = np.ones(100)*outer_spot_pos
        
        x_radians = np.cos(np.linspace(0, 2*np.pi, 100))*semi_angle
        y_radians = np.sin(np.linspace(0, 2*np.pi, 100))*semi_angle
        x_slopes = np.tan(x_radians)
        y_slopes = np.tan(y_radians)

        CDx, CDy = FieldCurvature_and_Astigmatism(x_f[0], x_f[2], x_pos, y_pos, x_slopes, y_slopes)
        ax.plot(CDx*1e3, CDy*1e3, color = color)

    return ax
                
def distortion_diagram(x_f, ax, color, top_left = (-1e-4, 1e-4), l = 2e-4, n = 100):
    #https://stackoverflow.com/a/53549359/20214963
    top = np.stack(
        [np.linspace(top_left[0], top_left[0] + l, n//4 + 1),
         np.full(n//4 + 1, top_left[1])],
         axis=1
    )[:-1]
    left = np.stack(
        [np.full(n//4 + 1, top_left[0]),
         np.linspace(top_left[1], top_left[1] - l, n//4 + 1)],
         axis=1
    )
    right = left.copy()
    right[:, 0] += l
    bottom = top.copy()
    bottom[:, 1] -= l
    square = np.concatenate([top, right, np.flip(bottom, axis = 0), np.flip(left, axis = 0)])
    
    Ex, Ey = Distortion(x_f[0], x_f[2], square[:, 0], square[:, 1])

    # gaussian_squre_x = x_f[0].getCoefficient([1, 0, 0, 0])*square[:, 0]
    # gaussian_squre_y = x_f[2].getCoefficient([0, 0, 1, 0])*square[:, 1]

    # ax.plot(gaussian_squre_x, gaussian_squre_y, color = 'gray', linestyle = '--')
    ax.plot(Ex*1e3, Ey*1e3, color = color)

    return ax
                
                
                
                