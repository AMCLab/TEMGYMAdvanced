#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 16:48:14 2023

@author: andy
"""
import numpy as np
import matplotlib.pyplot as plt
from mycolorpy import colorlist as mcp
from scipy.optimize import curve_fit 

def spherical_reducued_poly(m, C0, C1, C2):
    return C0*(1+1/(m**4))+C1*((1/m**3)+1/m)+C2*(1/m**2)

colors=mcp.gen_color(cmap="Set1",n=7)

convert_to_cs_image = lambda m, f, Sf, Sg: -1*((1+m**2)*Sg + 2*m*Sf)*((1+m)**2)*(f/4)
convert_to_cs_object = lambda m, f, Sf, Sg: -1*((1+(1/(m**2)))*Sg + (2/m)*Sf)*((1+1/m)**2)*(f/4)

fig, ax = plt.subplots(figsize = (8, 5))
# ax.invert_yaxis()

ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

ax.set_yscale('symlog')
ax.set_title('Symmetric Lens - $U_L/U_0$ & Cso')
# ax.set_xlabel('$U_L/U_0$')
ax.set_xlabel('Magnification')
ax.set_ylabel('$C_so$ (m)')

linestyle = '-'


rempfer_data = np.loadtxt("nanomi_lens_SfSg.txt", delimiter=" ")
UL_U0, Sf, Sg = rempfer_data[:, 0], rempfer_data[:, 1], rempfer_data[:, 2]

test = convert_to_cs_image(5.2, 0.0508, -50, -45)*1000

data = np.zeros((6,6,7))
our_data = np.loadtxt('Measured_Cs_aberint_and_DA_focal_length.txt', delimiter=" ")
data[:, :, 0] = our_data

our_data = np.loadtxt('Measured_Cs_aberint_and_DA_-0.01.txt', delimiter=" ")
data[:, :, 1] = our_data

our_data = np.loadtxt('Measured_Cs_aberint_and_DA_-0.05.txt', delimiter=" ")
data[:, :, 2] = our_data

our_data = np.loadtxt('Measured_Cs_aberint_and_DA_-0.1.txt', delimiter=" ")
data[:, :, 3] = our_data

our_data = np.loadtxt('Measured_Cs_aberint_and_DA_-0.2.txt', delimiter=" ")
data[:, :, 4] = our_data

our_data = np.loadtxt('Measured_Cs_aberint_and_DA_-0.3.txt', delimiter=" ")
data[:, :, 5] = our_data

our_data = np.loadtxt('Measured_Cs_aberint_and_DA_-0.4.txt', delimiter=" ")
data[:, :, 6] = our_data

m = data[4, 4, 1:]
cs_object = data[4, 1, 1:]/m

ax.scatter(m, cs_object)

popt, pcov = curve_fit(spherical_reducued_poly, m, cs_object)

m_fit = np.linspace(-0.8, -0.001, 100)
ax.plot(m_fit, spherical_reducued_poly(m_fit, *popt))

