#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 16:50:17 2022

@author: andy
"""

import numpy as np
import sympy as sp
from numba import njit, jit
from tqdm import tqdm
from daceypy import DA, array
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from Laplace import AnalyticalLaplace
from scipy.special import ellipj, ellipkinc

#Linearised Equation of Motion Solver via the Euler Cromer method
@njit
def euler_paraxial_electrostatic(r, v, z, dz, steps):
    for i in range(1, steps):
        
        v[i] = v[i-1] + (-1*(U_(z[i-1]))/(2*U(z[i-1]))*v[i-1] - U__(z[i-1])/(4*(U(z[i-1])))*r[i-1])*dz
        r[i] = r[i-1] + v[i]*dz
        
    return r, v

def euler_dz(x, E, U, z, dz, steps):
    for _ in tqdm(range(1, steps)):
        
        Ex, Ey, Ez = E(x[0], x[1], z)
        u = U(x[0], x[1], z)
        v_ = 1 + x[2]**2 + x[3]**2
    
        x[2] += (1/(2*-1*u)*v_*((Ex) - x[2]*Ez))*dz
        x[3] += (1/(2*-1*u)*v_*((Ey) - x[3]*Ez))*dz
        
        x[0] += x[2]*dz
        x[1] += x[3]*dz
        
        z += dz
        
    print(z)
     
    return x

A = AnalyticalLaplace(0)
a = 0.025
phi_0 = 5
k = 0.5**(1/2)

phi = A.SchiskeField(a = a, phi_0 = phi_0, k = k)
phi_ = phi.diff(A.z, 1)
phi__ = phi.diff(A.z, 2)
phi____ = phi.diff(A.z, 4)

U = jit(sp.lambdify(A.z, phi))
U_ = jit(sp.lambdify(A.z, phi_))
U__ = jit(sp.lambdify(A.z, phi__))
U____ = jit(sp.lambdify(A.z, phi____))

z0 = -0.5
w = np.sqrt(1-(k*k)/2)
psi_0 = np.arctan(a/z0)
eta_1 = ellipkinc(psi_0, k*k) - np.pi/w
psi_1 = ellipj(eta_1, k*k)[3]

z_g = abs(a/np.tan(psi_1)) #Analytical Calculation


dz = 1e-7
z_ = np.arange(z0, z_g+dz, dz)
steps = len(z_)

r = np.zeros(steps)
v = np.zeros(steps)

r[0] = 0
v[0] = 1

h, h_ = euler_paraxial_electrostatic(r.copy(), v.copy(), z_, dz, steps)


def h_f(z):
    z_i = np.abs(z_- z).argmin()
    return h[z_i]

def h__f(z):
    z_i = np.abs(z_- z).argmin()
    return h_[z_i]

r[0] = 1
v[0] = 0

g, g_  = euler_paraxial_electrostatic(r.copy(), v.copy(), z_, dz, steps)

def g_f(z):
    z_i = np.abs(z_- z).argmin()
    return g[z_i]

def g__f(z):
    z_i = np.abs(z_- z).argmin()
    return g_[z_i]

M_ = g[-1]

# plt.figure()
# plt.plot(z_, h, 'k', label = 'h')
# plt.plot(z_, g, 'r', label = 'g')


# plt.hlines(r[0], z_[0], z_[-1], 'k', linestyles = '--')
# plt.hlines(0, z0, z_g, 'k')
# plt.vlines(0, 0, 1, 'gray', label = 'Lens Centre')

# plt.title('Principal Rays of Schiske Electrostatic Field')
# plt.xlabel('z (m)')
# plt.ylabel('r (m)')

def L(z):
    return (1/(32*np.sqrt(U(z))))*((U__(z)**2)/(U(z))-U____(z))

def M(z):
    return (1/(8*np.sqrt(U(z))))*(U__(z))

def N(z):
    return 1/2*(np.sqrt(U(z)))

# def F_200(z):
#     return (L(z)/4)*g*g*g*g + (M(z)/2)*g*g*g_*g_ + (N(z)/4)*g_*g_*g_*g_

def F_110(z):
    return 2*((L(z)/4)*h*h*g*g + (M(z)/2)*(h*h*g_*g_+g*g*h_*h_)/2 + (N(z)/4)*h_*h_*g_*g_)

def F_101(z):
    return 4*((L(z)/4)*h*g*g*g + (M(z)/2)*(h*g*g_*g_+g*g*h_*g_)/2 + (N(z)/4)*h_*g_*g_*g_)

def F_020(z):
    return (L(z)/4)*h*h*h*h + (M(z)/2)*h*h*h_*h_ + (N(z)/4)*h_*h_*h_*h_

def F_011(z):
    return 4*((L(z)/4)*h*h*h*g + (M(z)/2)*(h*g*h_*h_+h*h*h_*g_)/2 + (N(z)/4)*h_*h_*h_*g_)

def F_002(z):
    return 4*((L(z)/4)*h*h*g*g + (M(z)/2)*h*g*h_*g_ + (N(z)/4)*h_*h_*g_*g_)


B_val = 4/np.sqrt(U(z0))*simpson(F_020(z_), z_)*M_
F_val = 1/np.sqrt(U(z0))*simpson(F_011(z_), z_)*M_
C_val = 1/np.sqrt(U(z0))*simpson(F_002(z_), z_)*M_
D_val = 2/np.sqrt(U(z0))*simpson(F_110(z_), z_)*M_
E_val = 1/np.sqrt(U(z0))*simpson(F_101(z_), z_)*M_

int_aber = [B_val, F_val, C_val, D_val, E_val]

A = AnalyticalLaplace(3)
phi = A.SchiskeField()
E_jit, U_jit, E_lambda, U_lambda = A.RoundLensFieldCartE(phi)
phi_func = sp.lambdify(A.z, phi)

dz = 1e-7
z_ = np.arange(z0, z_g, dz)
steps = len(z_)

DA.init(3, 4)

x = array([0+ DA(1), 0 + DA(2), 1e-7 + DA(3), 0 + DA(4)])

x_f = euler_dz(x, E_lambda, U_lambda, z0, dz, steps)
print(x_f[0].getCoefficient([1, 0, 0, 0]))

diff = z_g - z_[-1]
dz = 1e-8
steps = int(diff/dz) 

x_f = euler_dz(x_f, E_lambda, U_lambda, z_[-1], dz, steps)
print(x_f[0].getCoefficient([1, 0, 0, 0]))

B = x_f[0].getCoefficient([0, 0, 3, 0]) #Spherical
F = x_f[0].getCoefficient([1, 0, 0, 2]) #Coma
C = x_f[0].getCoefficient([1, 1, 0, 1])/2 #Astigmatism
D = x_f[0].getCoefficient([0, 2, 1, 0]) #Field Curvature
E = x_f[0].getCoefficient([3, 0, 0, 0]) #Distortion

da_aber = [B, F, C, D, E]

res = "\n".join("{} {}".format(x, y) for x, y in zip(int_aber, da_aber))
print(res)

