#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 23:07:50 2022

@author: andy
"""
import sympy as sp
from Laplace import AnalyticalLaplace
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import numba
from scipy.special import ellipj, ellipkinc
from scipy.integrate import solve_ivp
from odedopri import odedopri_store
plt.rc('font', family='Helvetica')

@njit
def linear_ODE(z, x, U, U_, U__):
    return np.array([x[1], (-1*(U_(z))/(2*U(z))*x[1] - U__(z)/(4*U(z))*x[0])])

A = AnalyticalLaplace(0)
a = 0.025
phi_0 = 5
k = 0.5**(1/2)

phi = A.SchiskeField(a = a, phi_0 = phi_0, k = k)
phi_ = phi.diff(A.z)
phi__ = phi_.diff(A.z)

U = numba.jit(sp.lambdify(A.z, phi))
U_ = numba.jit(sp.lambdify(A.z, phi_))
U__ = numba.jit(sp.lambdify(A.z, phi__))

zg = 0.81728749 #From Paper
z0 = -0.5

w = np.sqrt(1-(k*k)/2)
psi_0 = np.arctan(a/z0)
eta_1 = ellipkinc(psi_0, k*k) - np.pi/w
psi_1 = ellipj(eta_1, k*k)[3]

zg_calc = abs(a/np.tan(psi_1)) #Analytical Calculation
M_calc = -1*np.sin(psi_0)/np.sin(psi_1)
f_calc = 1/((1/a)*(np.sqrt(5)/np.sqrt(U(zg)))*(np.sin(psi_0)*np.cos(psi_1)*np.sqrt(1-(k*k)*np.sin(psi_1)**2)-
                                           np.cos(psi_0)*np.sin(psi_1)*np.sqrt(1-(k*k)*np.sin(psi_0)**2)))

l_max = np.abs(z0) + zg_calc
dz = 1e-8

z = np.arange(z0, zg_calc, dz)
steps = len(z)

# sol_g = solve_ivp(linear_ODE, t_span = [z[0], z[-1]], y0 = [1, 0], 
#                 args = (U, U_, U__), t_eval = z, method = 'RK45', rtol = 1e-13, atol = 1e-13, max_step = 1e-3)
# g, g_ = sol_g.y
# sol_h = solve_ivp(linear_ODE, t_span = [z[0], z[-1]], y0 = [0, 1], 
#                 args = (U, U_, U__), t_eval = z, method = 'RK45', rtol = 1e-13, atol = 1e-13, max_step = 1e-3)
# h, h_ = sol_h.y

z_g_ray, G, ig = odedopri_store(linear_ODE,  z[0],  np.array([1, 0]),  zg_calc,  1e-13,  1e-3,  1e-15, 1000000, (U, U_, U__))
z_g_ray, g, g_ = z_g_ray[:ig], G[:ig, 0], G[:ig, 1]
# plt.plot(z_g_ray[:ig], g, '.', color = 'k', label = 'Linearised ODE - g')
z_h_ray, H, ih = odedopri_store(linear_ODE,  z[0],  np.array([0, 1]),  zg_calc,  1e-13,  1e-3,  1e-15, 1000000, (U, U_, U__))
z_h_ray, h, h_ = z_h_ray[:ih], H[:ih, 0], H[:ih, 1]
# plt.plot(z_h_ray[:ih], h, '.', color = 'gray', label = 'Linearised ODE - h')

fig, ax1 = plt.subplots(figsize = (10,5))
ax2 = ax1.twinx()

ax1.plot(z_g_ray, g, color = 'dodgerblue', linewidth = 2, alpha = 0.7, zorder = 0, label = 'g')
ax1.plot(z_h_ray, h, color = 'r', linewidth = 2, alpha = 0.7, zorder = 0, label = 'h')

n_g = np.where(g<0)[0][0] #index just after where ray crossed axis
x0, y0, x1, y1 = z_g_ray[n_g-1], g[n_g-1], z_g_ray[n_g], g[n_g]

#Get Focal Length when ray crossed radial axis via linear interpolation. 
zinterp = (x0*(y1-0) + x1*(0-y0))/(y1-y0)
principal_ray_slope = g_[-1]
principal_ray_intercept = -principal_ray_slope*zinterp
z_po = (1-principal_ray_intercept)/(principal_ray_slope)
zf = abs(z_po)+zinterp

# x0, y0, x1, y1 = z[-2], g[-2], z[-1], g[-1]
# M_interp = (y0*(x1-zg_calc) + y1*(zg_calc-x0))/(x1-x0)

principal_ray_x = np.linspace(z_po, z[-1], 100)
principal_ray_y = principal_ray_slope*principal_ray_x+principal_ray_intercept

ax1.plot(principal_ray_x, principal_ray_y, 'k--', alpha = 0.25, zorder = 0)
ax1.hlines(0, z[0], z[-1], 'k', linestyles = '-', alpha = 0.25, zorder = 0)
ax1.hlines(1, z[0], z_po, 'k', linestyles = '--', alpha = 0.25, zorder = 0, label = 'Principal Ray')
ax1.vlines(z_po, 0, 1, 'g', alpha = 0.5, label = 'Principal Plane', zorder = 0)
ax2.vlines(0, 0, U(0), 'gray', alpha = 0.7, zorder = 0)
ax1.plot([], [], color = 'gray', linestyle = '-', label = 'Potential', zorder = 0, alpha = 0.7)
ax1.arrow(z0, 0, 0, 1, color = 'k', width = 0.005, length_includes_head = True, head_length = 0.1, zorder = 10)
ax1.arrow(zg, 0, 0, g[-1], color = 'k', width = 0.005, length_includes_head = True, head_length = 0.1, zorder = 10)
ax1.scatter([],[], c='k',marker=r'$\uparrow$',s=20, label='Object')
ax1.scatter([],[], c='k',marker=r'$\downarrow$',s=20, label='Image')

fig.legend(fontsize = '8', bbox_to_anchor=(0.9, 0.9))
ax2.plot(z, abs(U(z)-phi_0), color = 'gray', linestyle = '-', label = 'Potential', zorder = 0, alpha = 0.7)
# ax1.plot(zg_calc, M_calc, '.r')
# ax1.plot(zg_calc, M_interp, '.b')

axin = ax1.inset_axes([0.25, 0.15, 0.1, 0.3])
axin.plot(principal_ray_x, principal_ray_y, 'k--', alpha = 0.25, zorder = 0)
axin.hlines(0, z[0], z[-1], 'k', linestyles = '-', alpha = 0.25, zorder = 0)
axin.hlines(1, z[0], z_po, 'k', linestyles = '--', alpha = 0.25, zorder = 0)
axin.vlines(z_po, 0, 1, 'g', alpha = 0.5, label = 'Principal Plane', zorder = 0)
axin.vlines(0, 0, U(0), 'gray', alpha = 0.7, zorder = 0)

# Plot the data on the inset axis and zoom in on the important part
axin.set_xlim(z_po-0.005, z_po+0.005)
axin.set_ylim(0.8, 1.1)
xticklabels = [round(z_po, 4), 0]
axin.set_xticks([z_po, 0])
axin.set_xticklabels(xticklabels, rotation = 45, ha="right", fontsize = 8)
axin.get_xticklabels()[1].set_ha('center')
axin.get_xticklabels()[1].set_rotation(0)
yticklabels = [0.8, 1, 1.1]
axin.set_yticks(yticklabels)
axin.set_yticklabels(yticklabels, rotation = 0, fontsize = 8)

# Add the lines to indicate where the inset axis is coming from
ax1.indicate_inset_zoom(axin)
ax1.set_title("First Order Properties of Schiske's Electrostatic Lens")
ax1.set_xlabel('z (m)')
ax1.set_ylabel('r (m)')
ax2.set_xlabel('z (m)')
ax2.set_ylabel('U (V)')
ax2.set_ylim([-3, 3])
ax1.set_ylim([-2.5, 2.5])

ax1.spines.right.set_visible(False)
ax1.spines.top.set_visible(False)

for axis in ['right']:
    ax2.spines[axis].set_linewidth(1)
    ax2.spines[axis].set_linestyle('--')
    ax2.spines[axis].set_alpha(0.5)

ax2.tick_params(axis = 'y', direction='out', length=5, width=0.5, colors='k')
ax2.spines.top.set_visible(False)

plt.savefig('Electrostatic_Lens - First Order.svg', dpi = 800)
print('Focal Length - Analytical Solution', f_calc)
print('Focal Length - Linear Equation of Motion', zf)

print('Magnification - Analytical Solution', M_calc)
print('Magnification - Linear Equation of Motion', g[-1])

f_calc = round(f_calc, 14)
zf = round(zf, 14)
focal_diff = round(f_calc-zf, 14)

M_calc = round(M_calc, 14)
M_ode = round(g[-1], 14)
M_diff = round(abs(M_calc-M_ode), 14)

res = np.array([[f_calc, zf, focal_diff],[M_calc, M_ode, M_diff]])
np.savetxt("linear_electrostatic_results.csv", res, delimiter=",", fmt='%.14f, %.14f, %.14f') 