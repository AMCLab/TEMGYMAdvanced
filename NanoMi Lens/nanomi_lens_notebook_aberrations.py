# %%
import sys
sys.path.insert(0,'py-simple-fem-elstatics-master/py-simple-fem-elstatics-master')

import sympy as sp
import numpy as np
from Laplace import AnalyticalLaplace
from scipy.integrate import simpson

import matplotlib.pyplot as plt

from generate_expression import generate_n_gauss_expr
from sum_of_norms import sum_of_norms, norm
import numba
from numba import njit
from daceypy import DA, array
from tqdm import tqdm

u_l = -1000

# '''Load PYFEMM Data'''
pyfemm_data = np.loadtxt('pyfemm_nanomi_lens_insulator_rod_axisymmetric_cad.txt')
u_z = pyfemm_data[:, 1]*-1
z = (pyfemm_data[:, 0]-40)*1e-3

plt.figure()
plt.plot(z, u_z, '-r', alpha = 0.5, label = 'FEMM')
plt.legend()

# %%
'''Sum of Norms Fit'''
n_gaussians = 75
w_best, rms, means, sigmas = sum_of_norms(z, u_z, n_gaussians,
                                          spacing='linear', widths = None,
                                      full_output=True)

norms = w_best * norm(z[:, None], means, sigmas)

phi = generate_n_gauss_expr(w_best, means, sigmas)
phi_lambda = sp.lambdify(sp.abc.z, phi)

# plot the results
plt.plot(z, u_z, '-k', label='input potential')
ylim = plt.ylim()

plt.plot(z, norms, ls='-', c='#FFAAAA')
plt.plot(z, norms.sum(1), '-r', label='sum of gaussians')

plt.text(0.97, 0.8,
          "rms error = %.2g" % rms,
          ha='right', va='top', transform=plt.gca().transAxes)
plt.title("Fit to Potential with a Sum of %i Gaussians" % n_gaussians)

plt.legend(loc=0)

@njit
def euler_paraxial_electrostatic_kareh(r, v, z, U, U_, U__, dz):
    for i in range(1, r.shape[0]):
        
        v[i] = v[i-1] + (-1*(U_(z[i-1]))/(2*U(z[i-1]))*v[i-1] - U__(z[i-1])/(4*(U(z[i-1])))*r[i-1])*dz
        r[i] = r[i-1] + v[i]*dz
        z[i] = z[i-1] + dz 
        
    return r, v, z

@njit
def euler_paraxial_electrostatic_kareh_fast(r, v, z, U, U_, U__, dz, steps):
    for _ in range(1, steps):
        
        v += (-1*(U_(z))/(2*U(z))*v - U__(z)/(4*(U(z)))*r)*dz
        
        if r+v*dz < 0:
            return r, v, z
        
        r += v*dz
        z += dz 
        
    return r, v, z

# %%
A = AnalyticalLaplace(0)
U_0 = -1000

phi = phi-U_0
phi_ = phi.diff(A.z)
phi__ = phi_.diff(A.z)
phi___ = phi__.diff(A.z)
phi____ = phi___.diff(A.z)

U = numba.jit(sp.lambdify(A.z, phi))
U_ = numba.jit(sp.lambdify(A.z, phi_))
U__ = numba.jit(sp.lambdify(A.z, phi__))
U___ = numba.jit(sp.lambdify(A.z, phi___))
U____ = numba.jit(sp.lambdify(A.z, phi____))

# %%
z_f = 0.02
z_init = -0.03

l_max = np.abs(z_init) + z_f

dz = 1e-4
steps = int(round((abs(l_max))/dz))+1

r = np.zeros(steps)
v = np.zeros(steps)
z = np.zeros(steps)

r[0] = 1
v[0] = 0
z[0] = z_init

r_f, v_f, z_f = euler_paraxial_electrostatic_kareh_fast(1, 0, z_init, U, U_, U__, dz, steps)
print(z_f)

r[0] = 1
v[0] = 0
r_g, v_g, z_g = euler_paraxial_electrostatic_kareh_fast(0, 1, z_init, U, U_, U__, dz, steps)
print(z_g)

l_max = np.abs(z_init) + z_g

dz = 1e-5
steps = int(round((abs(l_max))/dz))+1

r = np.zeros(steps)
v = np.zeros(steps)
z = np.zeros(steps)

r[0] = 1
v[0] = 0
z[0] = z_init

g, g_, z_out_g = euler_paraxial_electrostatic_kareh(r.copy(), v.copy(), z.copy(), U, U_, U__, dz)


r[0] = 0
v[0] = 1

h, h_, z_out_h = euler_paraxial_electrostatic_kareh(r.copy(), v.copy(), z.copy(), U, U_, U__, dz)

plt.figure()

plt.plot(z_out_g, g, color = 'k', label = 'Linearised ODE - g')
plt.plot(z_out_h, h, color = 'gray', label = 'Linearised ODE - h')
plt.legend()

plt.hlines(0, z_init, z_f, 'k', alpha = 0.5)
plt.xlabel('z (m)')
plt.ylabel('r (m)')
plt.title('Linear Solution to Lens Potential')

# %%
n_g = -1
n_h = -1
x0, y0, x1, y1 = z_out_g[n_g-1], g[n_g-1], z_out_g[n_g], g[n_g]

#Get Focal Length when ray crossed radial axis via linear interpolation. 
focal_length = (x0*(y1-0) + x1*(0-y0))/(y1-y0)

x0, y0, x1, y1 = z_out_h[n_h-1], h[n_h-1], z_out_h[n_h], h[n_h]
gaussian_image_plane = (x0*(y1-0) + x1*(0-y0))/(y1-y0)

magnification = g[n_h]

# # %%
# #draw lines which will numerically tell me the principal image plane
principal_ray_slope = g_[n_h]
principal_ray_intercept = g[n_h]-(g_[n_h]*z_out_g[n_h])

z_po = (1 - principal_ray_intercept)/principal_ray_slope
zf = focal_length + abs(z_po)

print('Focal Length Wrt Principal Image Plane = ', zf)
print('Focal Length = ', focal_length)
print('Gaussian Image Plane = ', gaussian_image_plane)
print('Magnification = ', magnification)

principal_ray_x = np.linspace(z_init, gaussian_image_plane, 100)
principal_ray_y = principal_ray_slope*principal_ray_x+principal_ray_intercept

plt.plot(focal_length, 0, '.r')
plt.plot(0, principal_ray_intercept, '.b', label = 'Principal Ray Intercept')
plt.plot(principal_ray_x, principal_ray_y, 'k--')
plt.hlines(1, z_init, 0, 'r')
plt.vlines(z_po, 0, 1, 'r')
plt.hlines(0, z_init, gaussian_image_plane, 'k')
plt.vlines(0, 0, 1, 'gray', label = 'Lens Centre')
plt.vlines(gaussian_image_plane, 0, magnification, 'r')
plt.legend()

# %%
# Calculate focal length via aberration integral
@njit
def L1(z):
    return (1/(32*np.sqrt(U(z))))*((U__(z)**2)/(U(z))-U____(z))

@njit
def L2(z):
    return (1/(8*np.sqrt(U(z))))*(U__(z))

@njit
def L3(z):
    return 1/2*(np.sqrt(U(z)))

@njit
def C_func(z):
    return L1(z)*h*h*h*h + 2*L2(z)*h*h*h_*h_ + L3(z)*h_*h_*h_*h_

C_func(0)

B_val = 1/np.sqrt(U(z_init))*simpson(C_func(z_out_h), z_out_h)
print('Spherical Abberation Integral', B_val)

# %%
#Differential algebra & Spherical Aberration
def euler_dz(x, E, U, z, dz, steps):
    for i in tqdm(range(1, steps)):

        Ex, Ey, Ez = E(x[0], x[1], z)
        u = U(x[0], x[1], z)
        v_ = 1 + x[2]**2 + x[3]**2
    
        x[2] += (1/(2*-1*u)*v_*((Ex) - x[2]*Ez))*dz
        x[3] += (1/(2*-1*u)*v_*((Ey) - x[3]*Ez))*dz
        
        x[0] += x[2]*dz
        x[1] += x[3]*dz
        
        z += dz
         
    return x, z

@njit
def euler_dz_trace(r, v, z, E, U, dz):
    for i in range(1, r.shape[0]):
        Ex, _, Ez = E(r[i-1], 0, z[i-1])
        u = U(r[i-1], 0, z[i-1])

        v_ = 1 + v[i-1]**2
        v[i] = v[i-1] + (1/(2*-u)*v_*(Ex - v[i-1]*Ez))*dz
        r[i] = r[i-1] + v[i]*dz
        z[i] = z[i-1] + dz 

    return r, v, z

A = AnalyticalLaplace(1)
E_jit, U_jit, E_lambda, U_lambda = A.RoundLensFieldCartE(phi)

z_end = gaussian_image_plane

l_max = np.abs(z_init) + z_end

dz = 1e-5
steps = int(round((abs(l_max))/dz))+1
r = np.zeros(steps)
v = np.zeros(steps)
z = np.zeros(steps)

r[0] = 0
v[0] = 1e-5
z[0] = z_init

# euler_dz_trace(np.zeros(1), np.zeros(1), np.zeros(1), E_jit, U_jit, dz)
# r_o, v_o, z_o = euler_dz_trace(r, v, z, E_jit, U_jit, dz)

# plt.figure()
# # plt.plot(z_out_h, h, 'k')
# plt.plot(z_o, r_o, color = 'g', linewidth = 3, alpha = 0.6, label = 'Full Equation of Motion')

DA.init(3, 2)

x = array([0+ DA(1), 0, 1e-5 + DA(2), 0])

x_f, z_end = euler_dz(x, E_lambda, U_lambda, z_init, dz, steps)
print(x_f[0])
print(x_f[2])

F = -1/x_f[2].getCoefficient([1, 0])
M = x_f[0].getCoefficient([1, 0])
C = x_f[0].getCoefficient([0, 3])

# print('Final Position (ODE) = ', r_o[-1])
print('Final Position (DA) =', x_f[0].getCoefficient([0, 0]))
# print('Final Slope (ODE) =', v_o[-1])
print('Final Slope (DA) =', x_f[2].getCoefficient([0, 0]))
print('Focal Length (ODE)', zf)
print('Focal Length (DA)', F)
print('Magnification (ODE)', magnification)
print('Magnification (DA)', M)
print('Spherical Abberation (DA)', x_f[0].getCoefficient([0, 3])/M)
print('Spherical Abberation (Integral)', B_val)

spherical_aberration_data = np.loadtxt("nanomi_lens_SfSg.txt", delimiter=" ")

UL_U0, Sf, Sg = spherical_aberration_data[:, 0], spherical_aberration_data[:, 1], spherical_aberration_data[:, 2]

convert_to_cs_image = lambda m, f, Sf, Sg: -1*((1+(m**2))*Sg + 2*m*Sf)*((1+m)**2)*(f/4)
convert_to_cs_object = lambda m, f, Sf, Sg: -1*((1+(1/(m**2)))*Sg + (2/m)*Sf)*((1+(1/m))**2)*(f/4)

Cso = convert_to_cs_object(-1*M, F, Sf, Sg)
Csi = convert_to_cs_image(-1*M, F, Sf, Sg)
print(Cso, Csi)


