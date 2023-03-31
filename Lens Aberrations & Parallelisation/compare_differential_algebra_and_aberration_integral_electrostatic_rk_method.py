#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 16:50:17 2022

@author: andy
"""

import numpy as np
import sympy as sp
from numba import jit
from daceypy import DA, array
from Laplace import AnalyticalLaplace
from scipy.special import ellipj, ellipkinc
from odedopri import odedopri, odedopri_store
from scipy.integrate import solve_ivp, simpson
from scipy.interpolate import CubicSpline


def linear_ODE(z, x, U, U_, U__):
    return np.array([x[1], (-1*(U_(z))/(2*U(z))*x[1] - U__(z)/(4*U(z))*x[0])])


def complete_ODE(z, x, E, U):
    Ex, Ey, Ez = E(x[0], x[2], z)
    u = U(x[0], x[2], z)
    v_ = 1 + x[1]**2 + x[3]**2

    return np.array([x[1], ((1/(2*-1*u)*v_*((Ex) - x[1]*Ez))), x[3], (1/(2*-1*u)*v_*((Ey) - x[3]*Ez))])


A = AnalyticalLaplace(0)
a = 0.025
phi_0 = 5
k = 0.5**(1/2)

phi = A.SchiskeField(a=a, phi_0=phi_0, k=k)
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
M_ = -1*np.sin(psi_0)/np.sin(psi_1)
z_g = abs(a/np.tan(psi_1))


dz = 1e-7
z = np.arange(z0, z_g, dz)
steps = len(z)

r = np.zeros(steps)
v = np.zeros(steps)

r[0] = 0
v[0] = 1

# sol_g = solve_ivp(linear_ODE, t_span=[z[0], ], y0=[1, 0],
#                   args=(U, U_, U__), t_eval=z, method='RK45', rtol=1e-9, atol=1e-9, max_step=1e-3)
# g, g_ = sol_g.y
# sol_h = solve_ivp(linear_ODE, t_span=[z[0], z[-1]], y0=[0, 1],
#                   args=(U, U_, U__), t_eval=z, method='RK45', rtol=1e-9, atol=1e-9, max_step=1e-3)
# h, h_ = sol_h.y

z_g_ray, G, ig = odedopri_store(linear_ODE,  z[0],  np.array([1, 0]),  z[-1],  1e-12,  1e-1,  1e-15, 10000, (U, U_, U__))
g, g_ = G[:ig, 0], G[:ig, 1]
# plt.plot(z_g_ray[:ig], g, '.', color = 'k', label = 'Linearised ODE - g')
z_h_ray, H, ih = odedopri_store(linear_ODE,  z[0],  np.array([0, 1]),  z[-1],  1e-12,  1e-1,  1e-15, 10000, (U, U_, U__))
h, h_ = H[:ih, 0], H[:ih, 1]
# plt.plot(z_h_ray[:ih], h, '.', color = 'gray', label = 'Linearised ODE - h')

fg = CubicSpline(z_g_ray[:ig], g)
fg_ = CubicSpline(z_g_ray[:ig], g_)
fh = CubicSpline(z_h_ray[:ih], h)
fh_ = CubicSpline(z_h_ray[:ih], h_)

g, g_, h, h_ = fg(z), fg_(z), fh(z), fh_(z)


def L(z):
    return (1/(32*np.sqrt(U(z))))*((U__(z)**2)/(U(z))-U____(z))


def M(z):
    return (1/(8*np.sqrt(U(z))))*(U__(z))


def N(z):
    return 1/2*(np.sqrt(U(z)))


def F_200(z):
    return (L(z)/4)*g*g*g*g + (M(z)/2)*g*g*g_*g_ + (N(z)/4)*g_*g_*g_*g_


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


B_val = 4/np.sqrt(U(z0))*simpson(F_020(z), z)*M_
F_val = 1/np.sqrt(U(z0))*simpson(F_011(z), z)*M_
C_val = 1/np.sqrt(U(z0))*simpson(F_002(z), z)*M_*2
D_val = 2/np.sqrt(U(z0))*simpson(F_110(z), z)*M_
E_val = 1/np.sqrt(U(z0))*simpson(F_101(z), z)*M_

int_aber = np.round([B_val, F_val, C_val, D_val, E_val], 11)

A = AnalyticalLaplace(1)
phi = A.SchiskeField()
E_jit, U_jit, E_lambda, U_lambda = A.RoundLensFieldCartE(phi)

DA.init(3, 4)

x0 = 0
y0 = 0

x0_slope = 0
y0_slope = 0

x = array([x0 + DA(1), x0_slope + DA(2), y0 + DA(3), y0_slope + DA(4)])

zf, x_f = odedopri(complete_ODE,  z0,  x,  z_g,  1e-8,  1e-3,
                   1e-15,  int(1e6), (E_lambda, U_lambda))

B = x_f[0].getCoefficient([0, 3, 0, 0])
F = x_f[0].getCoefficient([1, 0, 0, 2])
C = x_f[0].getCoefficient([1, 0, 1, 1])
D = x_f[0].getCoefficient([0, 1, 2, 0])
E = x_f[0].getCoefficient([3, 0, 0, 0])

da_aber = np.round([B, F, C, D, E], 11)

diff = np.round(np.abs(np.array(int_aber) - np.array(da_aber)), 11)

res = np.asarray([int_aber, da_aber, diff]).T
np.savetxt("electrostatic_da_results.csv", res, delimiter=",", fmt='%.11f, %.11f, %.11f')
