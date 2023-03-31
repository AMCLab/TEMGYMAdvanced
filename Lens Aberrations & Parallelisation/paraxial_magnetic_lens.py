#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 10:18:44 2022

@author: andy
"""
import sympy as sp
from Laplace import AnalyticalLaplace
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, jit
from scipy.integrate import solve_ivp
from odedopri import odedopri_store

plt.rc('font', family='Helvetica')

@njit
def linear_ODE(z, x, alpha, B):
    return np.array([x[1], (-alpha*B(z)**2*x[0])])

#Define constants
q = 1.60217662e-19
m = 9.10938356e-31
c = 2.99792458e8

#Define Step Size
dz = 1e-8

#Set V accelerating and other variables for relativistic voltage acceleration
V = 1000
V = (V+(abs(q)/(2*m*(c**2)))*(V)**2)
eta = np.sqrt(abs(q)/(2*m))
alpha = eta**2/(4*V)

#Set gapsize and tesla of lens
a = 0.0075
Bmag = 0.01

#Define funky variables from Hawkes and Karehs book (7.10)
k_sqr = q*(Bmag**2)*(a**2)/(8*m*V)

w = np.sqrt(1+k_sqr)
K = np.sqrt(k_sqr)
z0 = -0.1 #Starting position in metres

#Convert to coordinate system for principal image plane
psi_0 = np.arctan(a/z0)
psi_1 = psi_0 - (np.pi/w)

#Calculate gaussian image plane, z_p focal length (w.r.t principal plane), zm focal length (w.r.t centre) and 
#the principal plane
zg_calc = abs(a/np.tan(psi_1))
f_calc = 1/((1/a)*(np.sin(np.pi/np.sqrt(k_sqr+1))))
M_calc = -1*np.sin(psi_0)/np.sin(psi_1)
zm_calc = -1*a*(1/np.tan(np.pi/w))
z_po = zm_calc - f_calc

#This time we use the analytical calculation to determine where to put the ray, although
#we could also use our g ray to determine this first. This is more conveninet for now. 
zg = zg_calc

#Set up fields
Laplace = AnalyticalLaplace(0)
glaser = Laplace.GlaserBellField(a=a, zpos=0)
glaser_ = glaser.diff(Laplace.z)
glaser__ = glaser_.diff(Laplace.z)

B = jit(sp.lambdify(Laplace.z, Bmag*glaser))

#Set number of steps
z = np.arange(z0, zg_calc+dz, dz)

#Set number of steps
steps = len(z)

r = np.zeros(steps)
v = np.zeros(steps)

#Trace ray to get gaussian image plane numerically

sol_g = solve_ivp(linear_ODE, t_span = [z[0], z[-1]], y0 = [1, 0], 
                args = (alpha, B), t_eval = z, method = 'RK45', rtol = 1e-13, atol = 1e-13, max_step = 1e-3)
g, g_ = sol_g.y
sol_h = solve_ivp(linear_ODE, t_span = [z[0], z[-1]], y0 = [0, 1], 
                args = (alpha, B), t_eval = z, method = 'RK45', rtol = 1e-13, atol = 1e-13, max_step = 1e-3)
h, h_ = sol_h.y

z_g_ray, G, ig = odedopri_store(linear_ODE,  z[0],  np.array([1, 0]),  zg_calc,  1e-13,  1e-3,  1e-15, 1000000, (alpha, B))
z_g_ray, g, g_ = z_g_ray[:ig], G[:ig, 0], G[:ig, 1]
# plt.plot(z_g_ray[:ig], g, '.', color = 'k', label = 'Linearised ODE - g')
z_h_ray, H, ih = odedopri_store(linear_ODE,  z[0],  np.array([0, 1]),  zg_calc,  1e-13,  1e-3,  1e-15, 1000000, (alpha, B))
z_h_ray, h, h_ = z_h_ray[:ih], H[:ih, 0], H[:ih, 1]

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
ax1.plot([], [], color = 'gray', linestyle = '-', label = 'B Field', zorder = 0, alpha = 0.7)
ax1.arrow(z0, 0, 0, 1, color = 'k', width = 0.001, length_includes_head = True, head_length = 0.1, zorder = 10)
ax1.arrow(zg, 0, 0, g[-1], color = 'k', width = 0.001, length_includes_head = True, head_length = 0.1, zorder = 10)
ax1.scatter([],[], c='k',marker=r'$\uparrow$',s=20, label='Object')
ax1.scatter([],[], c='k',marker=r'$\downarrow$',s=20, label='Image')

fig.legend(fontsize = '8', bbox_to_anchor=(0.9, 0.9))
ax2.plot(z, abs(B(z)), color = 'gray', linestyle = '-', label = 'B-Field', zorder = 0, alpha = 0.7)
ax2.vlines(0, 0, B(0), 'gray', alpha = 0.7, zorder = 0)
# ax1.plot(zg_calc, M_calc, '.r')
# ax1.plot(zg_calc, M_interp, '.b')

axin = ax1.inset_axes([0.45, 0.15, 0.1, 0.3])
axin.plot(principal_ray_x, principal_ray_y, 'k--', alpha = 0.25, zorder = 0)
axin.hlines(0, z[0], z[-1], 'k', linestyles = '-', alpha = 0.25, zorder = 0)
axin.hlines(1, z[0], z_po, 'k', linestyles = '--', alpha = 0.25, zorder = 0)
axin.vlines(z_po, 0, 1, 'g', alpha = 0.5, label = 'Principal Plane', zorder = 0)
axin.vlines(0, 0, 2, 'gray', alpha = 0.7, zorder = 0)
axin.plot(z_g_ray, g, color = 'dodgerblue', linewidth = 2, alpha = 0.7, zorder = 0, label = 'g')

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
ax1.set_title("First Order Properties of Glaser's Bell-Shaped Magnetic Lens")
ax1.set_xlabel('z (m)')
ax1.set_ylabel('r (m)')
ax2.set_xlabel('z (m)')
ax2.set_ylabel('B (T)')
ax2.set_ylim([-0.02, 0.02])
ax1.set_ylim([-1.5, 1.5])

ax1.spines.right.set_visible(False)
ax1.spines.top.set_visible(False)

for axis in ['right']:
    ax2.spines[axis].set_linewidth(1)
    ax2.spines[axis].set_linestyle('--')
    ax2.spines[axis].set_alpha(0.5)

ax2.tick_params(axis = 'y', direction='out', length=5, width=0.5, colors='k')
ax2.spines.top.set_visible(False)

plt.savefig('Magnetic_Lens - First Order.svg', dpi = 800)
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


