#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 16:59:47 2023

@author: andy
"""
import numpy as np
import matplotlib.pyplot as plt

mesh_default_02V = np.loadtxt('Nanomi_Lens_Default_Mesh_On_ZAxis_0.02Vacuum.txt')
u_z = mesh_default_02V[:, 1]
z = (mesh_default_02V[:, 0]-50)*1e-3

plt.plot(z, u_z, '.', alpha = 0.5, label = 'Default Mesh Size. 0.02 Vacuum')

mesh_1 = np.loadtxt('Nanomi_Lens_0.1_Mesh_On_ZAxis.txt')
u_z = mesh_1[:, 1]
z = (mesh_1[:, 0]-50)*1e-3

plt.plot(z, u_z, '.r', alpha = 0.5, label = '0.1 Mesh Size')

mesh_01 = np.loadtxt('Nanomi_Lens_0.01_Mesh_On_ZAxis.txt')
u_z = mesh_01[:, 1]
z = (mesh_01[:, 0]-50)*1e-3

plt.plot(z, u_z, '.g', alpha = 0.5, label = '0.01 Mesh Size')

mesh_001 = np.loadtxt('Nanomi_Lens_0.001_Mesh_On_ZAxis.txt')
u_z = mesh_001[:, 1]
z = (mesh_001[:, 0]-50)*1e-3

plt.plot(z, u_z, '.m', alpha = 0.5, label = '0.001 Mesh Size')

mesh_0002 = np.loadtxt('Nanomi_Lens_0.0002_Mesh_On_ZAxis.txt')
u_z = mesh_0002[:, 1]
z = (mesh_0002[:, 0]-50)*1e-3

plt.plot(z, u_z, '.m', alpha = 0.5, label = '0.0002 Mesh Size')

mesh_001_1V = np.loadtxt('Nanomi_Lens_0.001_Mesh_On_ZAxis_0.1Vacuum.txt')
u_z = mesh_001_1V[:, 1]
z = (mesh_001_1V[:, 0]-50)*1e-3

plt.plot(z, u_z, '.y', alpha = 0.5, label = '0.001 Mesh Size, 0.1 Vacuum')

mesh_default = np.loadtxt('Nanomi_Lens_Default_Mesh.txt')
u_z = mesh_default[:, 1]
z = (mesh_default[:, 0]-50)*1e-3

plt.plot(z, u_z, '.b', alpha = 0.5, label = 'Default Mesh Size')
plt.legend()


