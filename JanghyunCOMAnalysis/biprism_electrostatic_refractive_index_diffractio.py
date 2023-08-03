from diffractio import degrees, mm, plt, sp, um, np
from diffractio.scalar_masks_X import Scalar_mask_X
from diffractio.scalar_masks_XZ import Scalar_mask_XZ
from diffractio.scalar_sources_X import Scalar_source_X

import matplotlib.pyplot as plt
from scipy.constants import c, epsilon_0, e, m_e, h
import numpy as np
import sympy as sp

X, Z = sp.symbols('X Z')

phi_0 = 0.5
wavelength = h/(2*abs(e)*m_e*phi_0)**(1/2)/1e-6

V = 0.1
r = 0.125e-6/1e-6
R = 1e-5/1e-6

phi = 1+V*(sp.log((X*X+(Z)**2)/(R**2)))/(2*sp.log(r/R))/phi_0
n = str(phi**(0.5)).replace("log", "np.log")

x0 = np.linspace(-0.4 * um, 0.4 * um, 1024 * 1)
z0 = np.linspace(-2 * um, 3 * um, 1024 * 1)

u0 = Scalar_source_X(x=x0, wavelength=wavelength)
u0.plane_wave(A=1, z0=0 * um, theta=0. * degrees)

t0 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength, n_background=1.0)
t0.incident_field(u0)

t0.sphere(r0=(0, 0),
             radius=(R * um, R * um),
             refraction_index=n,
             angle=0 * degrees)

t0.sphere(r0=(0, 0),
             radius=(r * um, r * um),
             refraction_index=0 + 1j,
             angle=0 * degrees)

t0.draw_refraction_index(draw_borders=True, colorbar_kind='vertical')

t0.WPM(verbose=True, has_edges=False)
t0.draw(kind='intensity', draw_borders=True)
t0.draw(kind='phase', draw_borders=True)

u_field_last_plane = t0.profile_transversal(z0= 3 * um, normalize = 'maximum')
plt.xlim(-0.4 * um, 0.4 * um)
plt.legend()



