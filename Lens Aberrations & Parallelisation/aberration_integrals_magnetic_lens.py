#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 10:18:44 2022

@author: andy
"""
from Laplace import AnalyticalLaplace
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import sympy as sp
from scipy import integrate
from scipy.integrate import odeint
from scipy.constants import e, m_e, c


#Linearised Equation of Motion Solver via the Euler Cromer method
@jit
def euler_linear_magnetic(r, v, z, alpha, Bz, dz, steps):
    for i in range(1, steps):
        v[i] = v[i-1] + (-alpha*Bz(z[i-1])**2*r[i-1])*dz
        r[i] = r[i-1] + v[i]*dz
        
    return r, v


def model(x, z, alpha, B):
    return [x[1], (-alpha*B(z)**2*x[0])]

def main():
    q = e
    m = m_e
    
    #Define Step Size
    decimalplaces = 7
    dz = 1e-7
    
    #Set V accelerating and other variables
    V = 1000
    #V = (V+(abs(q)/(2*m*(c**2)))*(V)**2)
    eta = np.sqrt(q/(2*m))
    alpha = (eta**2)/(4*V)
    
    #Set gapsize and tesla of lens
    a = 0.01
    Bmag = 0.01
    
    #Define funky variables from Hawkes and Karehs book (7.10)
    k_sqr = q*(Bmag**2)*(a**2)/(8*m*V)
    
    w = np.sqrt(1+k_sqr)
    K = np.sqrt(k_sqr)
    z0 = -0.05#-0.032698493992741405#-0.05
    
    psi_0 = np.arctan(a/z0)
    psi_1 = psi_0 - (np.pi/w)
    
    #Calculate gaussian image plane, z_p focal length (w.r.t principal plane), zm focal length (w.r.t centre) and 
    #the principal plane
    zg = round(a/np.tan(psi_1), decimalplaces)
    
    f =  1/((1/a)*(np.sin(np.pi/np.sqrt(k_sqr+1))))
    zm = -1*a*1/(np.tan(np.pi/w))
    zpo = a*1/(np.tan(np.pi/(2*w)))
    M = (f/(z0+zm))
    
    #Set up fields
    Laplace = AnalyticalLaplace(0)
    
    glaser = Bmag*Laplace.GlaserBellField(a=a, zpos=0)
    glaser_ = glaser.diff(Laplace.z)
    glaser__ = glaser_.diff(Laplace.z)
    
    B = jit(sp.lambdify(Laplace.z, glaser))
    B_ = jit(sp.lambdify(Laplace.z, glaser_))
    B__ = jit(sp.lambdify(Laplace.z, glaser__))
    
    z_ = np.arange(z0, zg+dz, dz)
    
    #Set number of steps
    steps = len(z_)
    
    r = np.zeros(steps)
    z = np.zeros(steps)
    v = np.zeros(steps)
    
    #Trace ray to get gaussian image plane numerically
    r[0] = 0
    v[0] = 1
    z[0] = z0
    
    h, h_ = euler_linear_magnetic(r.copy(), v.copy(), z_, alpha, B, dz, steps)
    
    
    def h_f(z):
        z_i = np.abs(z_- z).argmin()
        return h[z_i]
    
    def h__f(z):
        z_i = np.abs(z_- z).argmin()
        return h_[z_i]
    
    r[0] = 1
    v[0] = 0
    g, g_ = euler_linear_magnetic(r.copy(), v.copy(), z_, alpha, B, dz, steps)
    
    def g_f(z):
        z_i = np.abs(z_- z).argmin()
        return g[z_i]
    
    def g__f(z):
        z_i = np.abs(z_- z).argmin()
        return g_[z_i]
    
    M_ = g[-1]
    farg = np.argmin(abs(g[100:]))+100
    f_ = (z_[farg]+ z_[farg -1])/2
    #h, h_ = odeint(model, [0, 1], z_, (alpha, B)).T   
    #g, g_ = odeint(model, [1, 0], z_, (alpha, B)).T   
    
    # h = h.copy()
    # h_ = h_.copy()
    
    # plt.figure()
    plt.plot(z_, h, 'k')
    plt.plot(z_, g, 'r')
    
    #get where it crosses the z-axis
    zg_ = z_[np.argmin(abs(h[100:]))+100]
    print('Gaussian Image Plane -  Analytical Solution: ', zg)
    print('Gaussian Image Plane - Linear Equation of Motion: ', zg_)
    
    d = a
    a_0 = (np.pi*k_sqr)/(4*(1+k_sqr)**(3/2))+(1/8)*((4*k_sqr-3)/(4*k_sqr+3))*np.sin((2*np.pi)/(1+k_sqr)**(1/2))
    a_1 = -(1/2)*((4*k_sqr-3)/(4*k_sqr+3))*(np.sin((np.pi)/(1+k_sqr)**(1/2)))**2
    a_2 = (np.pi*k_sqr)/(2*(1+k_sqr)**(3/2))
    a_3 = -(1/2)*((4*k_sqr-3)/(4*k_sqr+3))*(np.sin((np.pi)/(1+k_sqr)**(1/2)))**2
    a_4 = (np.pi*k_sqr)/(4*(1+k_sqr)**(3/2))-(1/8)*((4*k_sqr-3)/(4*k_sqr+3))*np.sin((2*np.pi)/(1+k_sqr)**(1/2))
    
    # #Cs calculation Numerical
    Cs = d*((a_4*(z0/d)**4)+(a_3*(z0/d)**3)+(a_2*(z0/d)**2)+(a_1*(z0/d))+a_0)*M
    

    def K(z):
        return(eta)**2*(B(z)**2)/(8*V)
    
    def L(z):
        return ((eta)**4)*(B(z)**4)/(32*(V**2)) - (eta)**2*B(z)*B__(z)/(8*V)
    
    def P(z):
        return (eta*eta*eta)*(B(z)*B(z))/(16*(V**(3/2))) - (eta*B__(z))/(16*(V**(1/2)))
        
    def Q(z):
        return (eta*B(z))/(4*(V**(1/2)))
        
    N = 1/2
        
    def B_func(z):
        return L(z)*h*h*h*h + 2*K(z)*h*h*h_*h_ + N*h_*h_*h_*h_
    
    def F_func(z):
        return L(z)*h*h*h*g + K(z)*h*h_*(g*h) + N*g_*h_*h_*h_
    
    def C_func(z):
        return L(z)*g*g*h*h + 2*K(z)*g*g_*h*h_ + N*g_*g_*h_*h_ - K(z)
    
    def D_func(z):
        return L(z)*g*g*h*h + K(z)*(g*g*h_*h_+g_*g_*h*h) + N*g_*g_*h_*h_ + 2*K(z)
    
    def E_func(z):
        return L(z)*g*g*g*h + K(z)*g*g_*(g_*h) + N*g_*g_*g_*h_
    
    def f_func(z):
        return P(z)*h*h + Q(z)*h_*h_
    
    def c_func(z):
        return P(z)*g*h + Q(z)*g_*h_
    
    def e_func(z):
        return P(z)*g*g + Q(z)*g_*g_
    
    def test_f(z):
        return (((eta*B__(z))/np.sqrt(V))+(2*(eta**3)*(B(z)**3))/(V**(3/2)))*h*h
    
    B_val = integrate.simpson(B_func(z_), z_)*(M_)
    F_val = integrate.simpson(F_func(z_), z_)*(M_)
    C_val = integrate.simpson(C_func(z_), z_)*(M_)
    D_val = integrate.simpson(D_func(z_), z_)*(M_)
    E_val = integrate.simpson(E_func(z_), z_)*(M_)
    c_val = integrate.simpson(c_func(z_), z_)*(M_)*2
    e_val = integrate.simpson(e_func(z_), z_)*(M_)
    f_val = integrate.simpson(f_func(z_), z_)*(M_)
    test_f = integrate.simpson(test_f(z_), z_)*(M_)*(1/16)
    
    
    print('M - Magnification - Analytical Solution:', M)
    print('M - Magnification - Numerical Solution:', M_)
    print('-1/f - focal length - Analytical Solution', f)
    print('-1/f - focal length - Numerical Solution', f_+zpo)
    print('B - Spherical Aberration - Analytical Solution', Cs)
    print('B - Spherical Aberration - Numerical Solution', B_val)
    print('F - Isotropic Coma - Numerical Solution', F_val)
    print('C - Isotropic Asigmation - Numerical Solution', C_val)
    print('D - Isotropic Field Curvature - Numerical Solution', D_val)
    print('E - Isotropic Distortion - Numerical Solution', E_val)
    print('c - Anisotropic Asigmation - Numerical Solution', c_val)
    print('e - Anisotropic Distortion - Numerical Solution', e_val)
    print('f - Anisotropic Coma - Numerical Solution', f_val)
    print('test_f - Anisotropic Coma - Numerical Solution', test_f)
    
    aberrations = [M, f, B_val, F_val, C_val, D_val, E_val, c_val, e_val, f_val]
    
    return aberrations

if __name__ == "__main__":
    main()
