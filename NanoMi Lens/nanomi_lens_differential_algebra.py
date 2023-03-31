#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 15:27:44 2022

@author: andy
"""

import pyvista as pv
from pyvista import examples
import pyvista 
import sympy as sp
import numpy as np
from sum_of_norms import sum_of_norms, norm, symbolic_norm
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from generate_expression import generate_n_gauss_expr
from sum_of_norms import sum_of_norms, norm
import numba
from numba import jit
from Laplace import AnalyticalLaplace
from odedopri import odedopri_store, odedopri
from daceypy import DA, array

def get_n_closest(array, value, k):
    return np.argsort(abs(array - value))[:k]
    
# %%
u_l = -1000

# '''#1 PYFEMM Test'''
pyfemm_data = np.loadtxt('pyfemm_nanomi_lens_insulator_rod_axisymmetric_cad.txt')
u_z = pyfemm_data[:, 1]*-1
z = (pyfemm_data[:, 0]-50)*1e-3

plt.figure()
plt.plot(z, u_z, '-r', alpha = 0.5)#, label = 'FEMM')
plt.legend()

# %%
'''Sum of Norms Fit'''
n_gaussians = 150
w_best, rms, means, sigmas = sum_of_norms(z, u_z, n_gaussians,
                                          spacing='linear',
                                      full_output=True)

norms = w_best * norm(z[:, None], means, sigmas)

phi_fit = generate_n_gauss_expr(w_best, means, sigmas)
phi_lambda = sp.lambdify(sp.abc.z, phi_fit)

# plot the results
plt.plot(z, u_z, '-k', label='input potential')
plt.ylabel('Voltage (V)')
plt.xlabel('Z axis (m)')
ylim = plt.ylim()

plt.plot(z, norms, ls='-', c='#FFAAAA')
plt.plot(z, norms.sum(1), '-r', label='sum of gaussians')

plt.text(0.97, 0.8,
          "rms error = %.2g" % rms,
          ha='right', va='top', transform=plt.gca().transAxes)
plt.title("Fit to Potential with a Sum of %i Gaussians" % n_gaussians)
plt.title('Nanomi Lens Potential - V central_electrode = 1000V')
plt.legend(loc=0)


def complete_ODE(z, x, E, U):
    Ex, Ey, Ez = E(x[0], x[2], z)
    u = U(x[0], x[2], z)
    v_ = 1 + x[1]**2 + x[3]**2
        
    return np.array([x[1], ((1/(-2*u)*v_*((Ex) - x[1]*Ez))), x[3], (1/(-2*u)*v_*((Ey) - x[3]*Ez))])

def complete_ODE(z, x, w_best, means, sigmas, U_0, U, U_, U__, U___, U____):
    # i = get_n_closest(means, z, 25)
    Usum = sum(U(z, means, sigmas, w_best)) - U_0
    Usum_ = sum(U_(z, means, sigmas, w_best))
    Usum__ = sum(U__(z, means, sigmas, w_best))
    Usum___ = sum(U___(z, means, sigmas, w_best))
    Usum____ = sum(U____(z, means, sigmas, w_best))
    
    # i = get_n_closest(means, z, 25)
    # Usum = sum(U(z, means[i], sigmas[i], w_best[i])) - U_0
    # Usum_ = sum(U_(z, means[i], sigmas[i], w_best[i]))
    # Usum__ = sum(U__(z, means[i], sigmas[i], w_best[i]))
    # Usum___ = sum(U___(z, means[i], sigmas[i], w_best[i]))
    # Usum____ = sum(U____(z, means[i], sigmas[i], w_best[i]))
    
    u = -(x[0]**2 + x[2]**2)*Usum__/4 + Usum
    Ex = -x[0]*(x[0]**2 + x[2]**2)*Usum____/16 + x[0]*Usum__/2
    Ey = -x[2]*(x[0]**2 + x[2]**2)*Usum____/16 + x[2]*Usum__/2
    Ez =  (x[0]**2 + x[2]**2)*Usum___/4 - Usum_
    
    # Ex, Ey, Ez = E_l(x[0], x[2], z)
    # u = U_l(x[0], x[2], z)
    v_ = 1 + x[1]**2 + x[3]**2
        
    return np.array([x[1], ((1/(-2*u)*v_*((Ex) - x[1]*Ez))), x[3], (1/(-2*u)*v_*((Ey) - x[3]*Ez))])

# %%
U_0 = -1000
phi = phi_fit-U_0
Laplace = AnalyticalLaplace(1)
# E_jit, U_jit, E_l, U_l = Laplace.RoundLensFieldCartE(phi)
sym_z, sym_z_0, sym_mu, sym_sigma = sp.symbols('z z_0 mu sigma')

# %%

phi = symbolic_norm(sym_z, sym_z_0, sym_mu, sym_sigma)
phi_ = phi.diff(Laplace.z, 1)
phi__ = phi.diff(Laplace.z, 2)
phi___ = phi.diff(Laplace.z, 3)
phi____ = phi.diff(Laplace.z, 4)

U = jit(sp.lambdify((sym_z, sym_z_0, sym_mu, sym_sigma), phi))
U_ = jit(sp.lambdify((sym_z, sym_z_0, sym_mu, sym_sigma), phi_))
U__ = jit(sp.lambdify((sym_z, sym_z_0, sym_mu, sym_sigma), phi__))
U___ = jit(sp.lambdify((sym_z, sym_z_0, sym_mu, sym_sigma), phi___))
U____ = jit(sp.lambdify((sym_z, sym_z_0, sym_mu, sym_sigma), phi____))


# %%
z_g = 0.010
z_0 = -0.050

x0 = 0
y0 = 0

y0_slope = 0
x0_slope = 1e-4

y = np.array([x0, x0_slope, y0, y0_slope])
z, Y, i = odedopri_store(complete_ODE,  z_0,  y,  z_g,  1e-5,  1e-1,  1e-10,  int(1e3), (w_best, means, sigmas, U_0, U, U_, U__, U___, U____))

fig, ax = plt.subplots() 
ax.plot(z[:i], Y[:i, 0], '-r')

DA.init(3, 4)

x0 = 0
y0 = 0

x0_slope = 0
y0_slope = 0

x = array([x0 + DA(1), x0_slope + DA(2), y0 + DA(3), y0_slope + DA(4)])
z_g = 0.0037464120394716145

with DA.cache_manager():
    zf, x_f = odedopri(complete_ODE,  z_0,  x,  z_g,  1e-4, 1e-4, 1e-12,  int(1e6), (w_best, means, sigmas, U_0, U, U_, U__, U___, U____))

Mag = x_f[0].getCoefficient([1, 0, 0, 0])
B = x_f[0].getCoefficient([0, 3, 0, 0])
F = x_f[0].getCoefficient([1, 0, 0, 2])
C = x_f[0].getCoefficient([1, 0, 1, 1])
D = x_f[0].getCoefficient([0, 1, 2, 0])
E = x_f[0].getCoefficient([3, 0, 0, 0])

da_aber = np.array([B, F, C, D, E])

print(da_aber)

