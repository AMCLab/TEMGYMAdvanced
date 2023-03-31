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
from scipy.constants import e, m_e
from scipy import integrate
from Laplace import AnalyticalLaplace
from odedopri import odedopri
from odedopri import odedopri, odedopri_store
from scipy.integrate import solve_ivp, simpson
from scipy.interpolate import CubicSpline

def linear_model(z, x, alpha, Bz):
    return np.array([x[1], (-alpha*Bz(z)**2*x[0])])

def complete_model(z, x, eta, rel_pot, B, B_z, d_B_z, B_mag):
    # Obtain B
    Bx, By, Bz = B(x[0], x[2], z)

    Bx, By, Bz = B_mag*Bx, B_mag*By, B_mag*Bz
    
    theta_ = (eta/2)*(B_mag*B_z(z))/(np.sqrt(rel_pot))
    theta__ = (eta/2)*(B_mag*d_B_z(z))/(np.sqrt(rel_pot))
    
    # Increment velocity and position
    p = np.sqrt(1+(x[1]-theta_*x[2])**2+(x[3]+theta_*x[0])**2)

    Bt = (1/p)*(Bz + (x[1]-theta_*x[2]) * Bx + (x[3]+theta_*x[0])*By)

    Fx = 2*theta_*x[3] + (theta_**2)*x[0] + theta__*x[2]
    Fy = -2*theta_*x[1] + (theta_**2)*x[2] - theta__*x[0]

    a_x = (eta*(p**2))/(np.sqrt(rel_pot)) * (p*By - (x[3]+(theta_*x[0]))*Bt) + Fx

    a_y = (eta*(p**2))/(np.sqrt(rel_pot)) * (-p*Bx + (x[1]-(theta_*x[2]))*Bt) + Fy
    
    return np.array([x[1], a_x, x[3], a_y])

q = e
m = m_e

#Define Step Size
dz = 1e-7

#Set V accelerating and other variables
V = 1000
eta = np.sqrt(q/(2*m))
alpha = (eta**2)/(4*V)

#Set gapsize and tesla of lens
a = 0.01
Bmag = 0.01

#Define funky variables from Hawkes and Karehs book (7.10)
k_sqr = q*(Bmag**2)*(a**2)/(8*m*V)

w = np.sqrt(1+k_sqr)
K = np.sqrt(k_sqr)
zm = -1*a*1/(np.tan(np.pi/w))
zpo = a*1/(np.tan(np.pi/(2*w)))
f =  1/((1/a)*(np.sin(np.pi/np.sqrt(k_sqr+1))))
z0 = (f/-1000)-zm

psi_0 = np.arctan(a/z0)
psi_1 = psi_0 - (np.pi/w)

M = -1*np.sin(psi_0)/np.sin(psi_1)

#Calculate gaussian image plane, z_p focal length (w.r.t principal plane), zm focal length (w.r.t centre) and 
#the principal plane
zg = a/np.tan(psi_1)

#Set up fields
Laplace = AnalyticalLaplace(0)

glaser = Bmag*Laplace.GlaserBellField(a=a, zpos=0)
glaser_ = glaser.diff(Laplace.z)
glaser__ = glaser_.diff(Laplace.z)

B = jit(sp.lambdify(Laplace.z, glaser))
B_ = jit(sp.lambdify(Laplace.z, glaser_))
B__ = jit(sp.lambdify(Laplace.z, glaser__))

z_ = np.arange(z0, zg, dz)

#Set number of steps
steps = len(z_)

# sol = solve_ivp(linear_model, t_span = [z_[0], z_[-1]], y0 = [1, 0], 
#                 args = (alpha, B), t_eval=z_, method = 'RK45', rtol = 1e-13, atol = 1e-13)
# g, g_ = sol.y
# sol = solve_ivp(linear_model, t_span = [z_[0], z_[-1]], y0 = [0, 1], 
#                 args = (alpha, B), t_eval=z_, method = 'RK45', rtol = 1e-13, atol = 1e-13)
# h, h_ = sol.y

z_g_ray, G, ig = odedopri_store(linear_model,  z_[0],  np.array([1, 0]),  z_[-1],  1e-12, 1e-1,  1e-15, 10000, (alpha, B))
g, g_ = G[:ig, 0], G[:ig, 1]
# plt.plot(z_g_ray[:ig], g, '.', color = 'k', label = 'Linearised ODE - g')
z_h_ray, H, ih = odedopri_store(linear_model,  z_[0],  np.array([0, 1]),  z_[-1],  1e-12,  1e-1,  1e-15, 10000, (alpha, B))
h, h_ = H[:ih, 0], H[:ih, 1]
# plt.plot(z_h_ray[:ih], h, '.', color = 'gray', label = 'Linearised ODE - h')

fg = CubicSpline(z_g_ray[:ig], g)
fg_ = CubicSpline(z_g_ray[:ig], g_)
fh = CubicSpline(z_h_ray[:ih], h)
fh_ = CubicSpline(z_h_ray[:ih], h_)

g, g_, h, h_ = fg(z_), fg_(z_), fh(z_), fh_(z_)


def K(z):
    return(eta)**2*(B(z)**2)/(8*V)

def L(z):
    return ((eta)**4)*(B(z)**4)/(32*(V**2)) - (eta)**2*B(z)*B__(z)/(8*V)

def Q(z):
    return (eta*B(z))/(4*(V**(1/2)))

def P(z):
    return (eta*eta*eta)*(B(z)*B(z)*B(z))/(16*(V**(3/2))) - (eta*B__(z))/(16*(V**(1/2)))
    
N = 1/2

gh = g*h
gh_ = np.gradient(gh, z_)

def B_func(z):
    return L(z)*h*h*h*h + 2*K(z)*h*h*h_*h_ + N*h_*h_*h_*h_

def F_func(z):
    return L(z)*h*h*h*g + K(z)*h*h_*gh_ + N*g_*h_*h_*h_

def C_func(z):
    return L(z)*g*g*h*h + 2*K(z)*g*g_*h*h_ + N*g_*g_*h_*h_ - K(z)

def D_func(z):
    return L(z)*g*g*h*h + K(z)*(g*g*h_*h_+g_*g_*h*h) + N*g_*g_*h_*h_ + 2*K(z)

def E_func(z):
    return L(z)*g*g*g*h + K(z)*g*g_*gh_ + N*g_*g_*g_*h_

def f_func(z):
    return P(z)*h*h + Q(z)*h_*h_

def c_func(z):
    return P(z)*g*h + Q(z)*g_*h_

def e_func(z):
    return P(z)*g*g + Q(z)*g_*g_

B_val = integrate.simpson(B_func(z_), z_)*(M)
F_val = integrate.simpson(F_func(z_), z_)*(M)
C_val = integrate.simpson(C_func(z_), z_)*(M)
D_val = integrate.simpson(D_func(z_), z_)*(M)
E_val = integrate.simpson(E_func(z_), z_)*(M)
c_val = integrate.simpson(c_func(z_), z_)*(M)*2
e_val = integrate.simpson(e_func(z_), z_)*(M)
f_val = integrate.simpson(f_func(z_), z_)*(M)

int_aber = np.round([B_val, F_val, C_val, D_val, E_val, f_val, c_val, e_val], 11)

Laplace = AnalyticalLaplace(1)

B_z = Laplace.GlaserBellField(a=a, zpos=0)
B_z_lambda = sp.lambdify(Laplace.z, B_z)
d_B_z_lambda = sp.lambdify(Laplace.z, B_z.diff(Laplace.z, 1))

B_jit, B_lambda, _ = Laplace.RoundLensFieldCart(B_z)

x0 = 0
y0 = 0

x0_slope = 0
y0_slope = 0

DA.init(3, 4)

x = array([x0 + DA(1), x0_slope + DA(2), y0 + DA(3), y0_slope + DA(4)])

zf, x_f = odedopri(complete_model,  z0,  x,  zg,  1e-8,  1e-1,  1e-15,  int(1e6), (eta, V, B_lambda, B_z_lambda, d_B_z_lambda, Bmag))

B = x_f[0].getCoefficient([0, 3, 0, 0])
F = x_f[0].getCoefficient([1, 0, 0, 2])
C = x_f[0].getCoefficient([1, 0, 1, 1])/2
D = x_f[0].getCoefficient([0, 1, 2, 0])
E = x_f[0].getCoefficient([3, 0, 0, 0])
f = x_f[2].getCoefficient([1, 0, 0, 2])/3
c = x_f[2].getCoefficient([1, 0, 1, 1])/2
e = x_f[2].getCoefficient([3, 0, 0, 0])

da_aber = np.round([B, F, C, D, E, f, c, e], 11)

diff = np.round(np.abs(np.array(int_aber) - np.array(da_aber)), 11)

print(diff)
res = np.asarray([int_aber, da_aber, diff]).T
np.savetxt("magnetic_da_results.csv", res, delimiter=",", fmt='%.11f, %.11f, %.11f') 
