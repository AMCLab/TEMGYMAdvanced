# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 10:24:35 2023

@author: User
"""
import numpy as np
import matplotlib.pyplot as plt
from daceypy import DA, array  
from numba import njit
import sympy as sp
from Laplace import AnalyticalLaplace
from odedopri import odedopri, odedopri_store
from tqdm import trange

def ray_matrix(x, x_slope):
    return np.array([[x, x_slope, 1]]).T

def lens_matrix(f):
    return np.array([[1, 0, 0],
                    [-1/f, 1, 0],
                    [0, 0, 1]])

def propagation_matrix(d):
    return np.array([[1, d, 0],
                     [0, 1, 0],
                     [0, 0, 1]])

def biprism_matrix(s):
    return np.array([[1, 0, 0],
                     [0, 1, s],
                     [0, 0, 1]])

def lens_da(r, f):
    r[1] = r[0]*(-1/f) + r[1]
    
    return r

def propagation_da(r, d):
    r[0] = r[0]+r[1]*d
    
    return r

def biprism_da(r, s):
    r[1] = r[1]+r[2]*s
    
    return r

def run(source_ray, source_z, biprism_z, lens_z, biprism, lens, Cs = 0):
    source_ray_biprism_dist = propagation_matrix(source_z-biprism_z)
    biprism_ray = np.matmul(biprism, np.matmul(source_ray_biprism_dist, source_ray))
    biprism_ray[1] = biprism_ray[1]
    
    biprism_ray_lens_dist = propagation_matrix(biprism_z-lens_z)
    lens_ray = np.matmul(lens, np.matmul(biprism_ray_lens_dist, biprism_ray))

    
    lens_ray_screen_dist = propagation_matrix(lens_z)
    screen_ray = np.matmul(lens_ray_screen_dist, lens_ray)
    
    return biprism_ray, lens_ray, screen_ray

def image_dist(u, f):
    return (1/f-1/u)**-1

def euler_dz_single(x, u0, E, U, z, dz, steps, biprism_z, tol):
    for _ in trange(1, steps):
        
        if (x[0](0)**2 + (z-biprism_z)**2) > 1e-3:
            Ex, Ey, Ez  = 0, 0, 0
            u = U(0, 0, 0, 0)
        else:
            Ex, Ey, Ez = E(x[0], 0, z, x[2])
            u = U(x[0], 0, z, x[2])

        v_ = 1 + x[1]**2
    
        x[1] += (1/(2*(u-u0))*v_*((Ex) - x[1]*Ez))*dz
        x[0] += x[1]*dz
        
        if abs(z - biprism_z) < tol:
            x[1] = x[0]*(-1/0.5) + x[1]
        
        
        z -= dz
         
    return x, z


def euler_dz_array(x, u0, E, U, z, dz, steps, biprism_z, tol):
    this_portion_is_executed=False
    for i in range(1, steps):
        Ex, Ey, Ez = E(x[i-1, 0], 0, z[i-1], 90)
        u = U(x[i-1, 0], 0, z[i-1], 90)
        
        v_ = 1 + x[i-1, 1]**2
    
        x[i, 1] = x[i-1 , 1] + (1/(2*(u-u0))*v_*((Ex) - x[i-1, 1]*Ez))*dz
        x[i, 0] = x[i-1 , 0] + x[i, 1]*dz
        
        if z[i-1] - biprism_z < 0 and not this_portion_is_executed:
            x[i, 1] = x[i, 0]*(-1/0.5) + x[i, 1]
            this_portion_is_executed=True
        
        z[i] = z[i-1]-dz
         
    return x, z

def model_da(z, x, u0, E, U):
    
    if (x[0](0)**2 + (z-0.66666666)**2) > 1e-3:
        Ex, Ey, Ez  = 0, 0, 0
        u = U(0, 0, 0, 0)
    else:
        Ex, Ey, Ez = E(x[0], 0, z, x[2])
        u = U(x[0], 0, z, x[2])

    Ex, Ey, Ez = E(x[0], 0, z, x[2])
    u = U(x[0], 0, z, x[2])
    v_ = 1 + x[1]**2
    
    return np.array([x[1], (1/(2*(u-u0))*v_*((Ex) - x[1]*Ez)), 1])


def model(z, x, u0, E, U):
    Ex, Ey, Ez = E(x[0], 0, z, 90)
    u = U(x[0], 0, z, 90)
    v_ = 1 + x[1]**2
    
    return np.array([x[1], (1/(2*(u-u0))*v_*((Ex) - x[1]*Ez))])

lens_val = 0.5
biprism_val = -7

object_dist = 2

lens_z = image_dist(object_dist, lens_val)
source_z = lens_z + object_dist
biprism_z = lens_z


biprism = biprism_matrix(biprism_val)
lens = lens_matrix(lens_val)
source_ray = ray_matrix(0, 1)

biprism_ray, lens_ray, screen_ray = run(source_ray, source_z, biprism_z, lens_z, biprism, lens, Cs = 0.001)

plt.figure()
plt.plot(source_ray[0], source_z, 'og')

# plt.plot([biprism_ray[0], biprism_ray[0]], [source_z, biprism_z], linestyle = '-.', color = 'g')
# plt.plot(biprism_ray[0], source_z, 'o', markerfacecolor = 'w', markeredgecolor = 'g')

plt.plot(0, biprism_z, 'ob')
plt.hlines(lens_z, -0.5, 0.5, 'k')
plt.hlines(0, -0.5, 0.5, 'k')
plt.plot([source_ray[0], biprism_ray[0], lens_ray[0], screen_ray[0]], [source_z, biprism_z, lens_z, 0], 'g')

# print([source_ray[0], biprism_ray[0], lens_ray[0], screen_ray[0]])
source_ray = ray_matrix(0, 2)
biprism_ray, lens_ray, screen_ray = run(source_ray, source_z, biprism_z, lens_z, biprism, lens)
plt.plot([source_ray[0], biprism_ray[0], lens_ray[0], screen_ray[0]], [source_z, biprism_z, lens_z, 0], 'r')
print([source_ray[0], biprism_ray[0], lens_ray[0], screen_ray[0]])

print(screen_ray)

DA.init(3, 3)
r = array([0 + DA(1), 2 + DA(2), 1 + DA(3)])

r = propagation_da(r, source_z-biprism_z)
r = biprism_da(r, biprism_val)
r = propagation_da(r, biprism_z - lens_z)
r = lens_da(r, lens_val)
r = propagation_da(r, lens_z)

#r[0] = r[0] + 0.1*(1+DA(2))**3
print(r[0])
print(r[1])
print(r[2])

Vf = 90
r_bi = 0.25e-6
R_bi = 1e-3
A = AnalyticalLaplace(0)
phi = A.SeptierBiprism(z_offset = biprism_z)
E_lambda, U_lambda = A.BiprismFieldCartE(phi)

z0 = source_z

dz = 1e-5
l_max = source_z

steps = int(l_max/dz+dz)


u0 = 3e4

x = np.zeros((steps, 2))
z = np.zeros(steps)

z[0] = z0
x[0, 0] = 0
x[0, 1] = 1e-4
x, z = euler_dz_array(x, u0, E_lambda, U_lambda, z, dz, steps, biprism_z, tol = 1e-5)

plt.figure()
plt.plot(0, biprism_z, 'ob')
plt.hlines(lens_z, -1e-3, 1e-3, 'k')
plt.hlines(0, -1e-3, 1e-3, 'k')
plt.plot(x[:, 0], z, 'g')

x = np.zeros((steps, 2))
z = np.zeros(steps)

z[0] = z0
x[0, 0] = 0
x[0, 1] = 2e-4
x, z = euler_dz_array(x, u0, E_lambda, U_lambda, z, dz, steps, biprism_z, tol = 1e-5)
plt.plot(x[:, 0], z, 'b')

# x = array([0 + DA(1),  1e-5 + DA(2), 90+DA(3)])
# z, x, end = odedopri_store(model, source_z,  np.array([0, 1e-4]),  0,  1e-7,  1e-5,  1e-14, 1000000, (u0, E_lambda, U_lambda))


# DA.init(3, 3)
# x = array([0 + DA(1),  1e-5 + DA(2), 90+DA(3)])
# z, x = odedopri(model_da, source_z,  x,  0,  1e-1,  1e-2,  1e-15, 1000000, (u0, E_lambda, U_lambda))

# DA.init(3, 3)
# dz = 1e-4
# steps = int(l_max/dz+dz)
# x = array([0 + DA(1),  1e-5 + DA(2), 90+DA(3)])
# x, z = euler_dz_single(x, u0, E_lambda, U_lambda, source_z, dz, steps, biprism_z, tol = 1e-7)

# print(x[0])
# print(x[1])

# DA.init(3, 3)
# dz = 1e-4
# steps = int(l_max/dz+dz)
# x = array([0 + DA(1),  1e-4 + DA(2), 90+DA(3)])
# x, z = euler_dz_single(x, u0, E_lambda, U_lambda, source_z, dz, steps, biprism_z, tol = 1e-7)

# print(x[0])
# print(x[1])

# DA.init(3, 3)
# dz = 1e-4
# steps = int(l_max/dz+dz)
# x = array([0 + DA(1),  1e-3 + DA(2), 90+DA(3)])
# x, z = euler_dz_single(x, u0, E_lambda, U_lambda, source_z, dz, steps, biprism_z, tol = 1e-7)

# print(x[0])
# print(x[1])



    

