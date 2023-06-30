#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 15:26:32 2022

@author: andy
"""
import numpy as np
from bfieldtools.line_magnetics import magnetic_field
import pyqtgraph.opengl as gl

def LensGeometry(R, L, Position, n):
    
    md = gl.MeshData.cylinder(rows=10, cols=20, radius=[R, R], length=L/2)
    
    mesh = gl.GLMeshItem(meshdata=md, smooth=True, color = [1, 1, 1, 0.35])
    mesh.setGLOptions('translucent')
    
    mesh.resetTransform()
    mesh.translate(Position[0], Position[1], Position[2]-(L/2)/2)
    
    return mesh

def SaddleGeometry(R, L, PHI, Position, offset, n_arc, n_lengths):
    
    points = np.empty((3, 0), int)
    
    theta = np.linspace(-PHI + offset, PHI + offset, n_arc, endpoint = False) 
    r = R*np.ones(np.size(theta))
    z = L/2*np.ones(np.size(theta))
    
    points_upper_arc = np.array([r*np.cos(theta), r*np.sin(theta), z])
    
    z = -L/2*np.ones(np.size(theta))
    theta = np.linspace(PHI + offset, -PHI + offset, n_arc, endpoint = False) 
    points_lower_arc = np.array([r*np.cos(theta), r*np.sin(theta), z]).T
    
    z = np.linspace(L/2, -L/2, n_lengths, endpoint = False)
    r = R*np.ones(np.size(z))
    
    theta = PHI*np.ones(np.size(z)) + offset
    points_str_1 = np.array([r*np.cos(theta), r*np.sin(theta), z]).T
    
    z = np.linspace(-L/2, L/2, n_lengths, endpoint = True)
    r = R*np.ones(np.size(z))
    
    theta = -PHI*np.ones(np.size(z)) + offset
    points_str_2 = np.array([r*np.cos(theta), r*np.sin(theta), z]).T

    points_saddle = np.hstack((points, points_upper_arc)).T
    points_saddle = np.vstack((points_saddle, points_str_1, points_lower_arc, points_str_2)).T
    
    points_saddle = points_saddle.T + Position
    
    return points_saddle

def LineGeometry(R, L, Position, theta, n_lengths):
    
    z = np.linspace(L/2, -L/2, n_lengths, endpoint = True)
    r = R*np.ones(np.size(z))
    theta = np.zeros(np.size(z)) + theta
    points_line = np.array([r*np.cos(theta), r*np.sin(theta), z]).T + Position

    return points_line

def SaddleBField(points, Position, n, sign, extent):
    
    xx, x_step = np.linspace(extent[0, 0], extent[0, 1], n, retstep = True)
    yy, y_step = np.linspace(extent[1, 0], extent[1, 1], n, retstep = True)
    zz, z_step = np.linspace(extent[2, 0], extent[2, 1], n, retstep = True)
    
    X, Y, Z = np.meshgrid(xx, yy, zz, indexing="ij")
    
    xyz_grid = np.stack((xx, yy, zz), axis=0)
    xyz_grid_step = np.array((abs(x_step), abs(y_step), abs(z_step)))
                
    x = X.ravel()
    y = Y.ravel()
    z = Z.ravel()

    b_points = np.array([x, y, z]).T

    B = np.zeros(b_points.shape)

    B += magnetic_field(points, b_points)*sign

    B_matrix = B.reshape((n, n, n, 3))
    
    return B_matrix, xyz_grid, xyz_grid_step