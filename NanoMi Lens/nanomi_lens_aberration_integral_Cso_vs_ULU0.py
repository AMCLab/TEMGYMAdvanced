# %%

import sympy as sp
import numpy as np

from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from generate_expression import generate_n_gauss_expr
from sum_of_norms import sum_of_norms, norm, symbolic_norm
from numba import jit
from Laplace import AnalyticalLaplace
from odedopri import odedopri_store, odedopri
from daceypy import DA, array

u_l = -1000

# '''#1 PYFEMM Test'''
pyfemm_data = np.loadtxt('Nanomi_Lens_Default_Mesh_On_ZAxis_0.02Vacuum.txt')
u_z = pyfemm_data[:, 1]*-1
z = (pyfemm_data[:, 0]-50)*1e-3

# plt.figure()
# plt.plot(z, u_z, '-r', alpha = 0.5)#, label = 'FEMM')
# plt.legend()

# %%
'''Sum of Norms Fit'''
n_gaussians = 201
w_best, rms, means, sigmas = sum_of_norms(z, u_z, n_gaussians,
                                          spacing='linear',
                                      full_output=True)

norms = w_best * norm(z[:, None], means, sigmas)

phi_fit = generate_n_gauss_expr(w_best, means, sigmas)
phi_lambda = sp.lambdify(sp.abc.z, phi_fit)

# plot the results
fig, ax = plt.subplots(1, 2, figsize = (12, 5))
ax[0].plot(z, u_z*-1, '-k', label='Input Potential')
ax[0].set_ylabel('Voltage (V)')
ax[0].set_xlabel('Z (m)')
ylim = plt.ylim()

ax[0].plot([],[], ls='-', c='#FFAAAA', label = 'Single Gaussian Function')
ax[0].plot(z, norms*-1, ls='-', c='#FFAAAA')
ax[0].plot(z, norms.sum(1)*-1, '-r', label='Sum of Gaussians')

ax[0].text(0.97, 0.8,
          "rms error = %.2g" % rms,
          ha='right', va='top', transform=plt.gca().transAxes)
ax[0].set_title('Nanomi Lens Potential - V = 1000V')

ax[1].plot(z, abs(norms.sum(1)-(u_z)), 'gray')
ax[1].set_ylabel('Error in Gaussian Fit (V)')
ax[1].set_xlabel('Z (m)')
ax[0].legend(loc=0, fontsize = 8)

ax[0].set_ylim([0, 1000])
ax[0].spines.right.set_visible(False)
ax[0].spines.top.set_visible(False)
ax[1].spines.right.set_visible(False)
ax[1].spines.top.set_visible(False)
ax[0].set_xlim([-0.02, 0.02])
ax[1].set_xlim([-0.02, 0.02])
ax[0].set_xticks(np.linspace(-0.02, 0.02, 5, endpoint = True))
ax[1].set_xticks(np.linspace(-0.02, 0.02, 5, endpoint = True))
plt.savefig('FitPotential.svg', dpi = 800)


def complete_ODE(z, x, w_best, means, sigmas, U_0, U, U_, U__, U___, U____):

    Usum = sum(U(z, means, sigmas, w_best))-U_0
    Usum_ = sum(U_(z, means, sigmas, w_best))
    Usum__ = sum(U__(z, means, sigmas, w_best))
    Usum___ = sum(U___(z, means, sigmas, w_best))
    Usum____ = sum(U____(z, means, sigmas, w_best))
    
    u = -(x[0]**2 + x[2]**2)*Usum__/4 + Usum
    Ex = -x[0]*(x[0]**2 + x[2]**2)*Usum____/16 + x[0]*Usum__/2
    Ey = -x[2]*(x[0]**2 + x[2]**2)*Usum____/16 + x[2]*Usum__/2
    Ez =  (x[0]**2 + x[2]**2)*Usum___/4 - Usum_
    
    v_ = 1 + x[1]**2 + x[3]**2
        
    return np.array([x[1], ((1/(-2*u)*v_*((Ex) - x[1]*Ez))), x[3], (1/(-2*u)*v_*((Ey) - x[3]*Ez))])

def model(z, x, w_best, means, sigmas, U_0, U, U_, U__):
    Usum = sum(U(z, means, sigmas, w_best))-U_0
    Usum_ = sum(U_(z, means, sigmas, w_best))
    Usum__ = sum(U__(z, means, sigmas, w_best))
    
    return np.array([x[1], ((-1*(Usum_)/(2*Usum )*x[1] - Usum__/(4*(Usum ))*x[0]))])
           

# %%
U_0 = -1000
phi = phi_fit-U_0

Laplace = AnalyticalLaplace(1)
sym_z, sym_z_0, sym_sigma, sym_w = sp.symbols('z z_0 mu w')

phi = symbolic_norm(sym_z, sym_z_0, sym_sigma, sym_w)
phi_ = phi.diff(Laplace.z, 1)
phi__ = phi.diff(Laplace.z, 2)
phi___ = phi.diff(Laplace.z, 3)
phi____ = phi.diff(Laplace.z, 4)

U = jit(sp.lambdify((sym_z, sym_z_0, sym_sigma, sym_w), phi))
U_ = jit(sp.lambdify((sym_z, sym_z_0, sym_sigma, sym_w), phi_))
U__ = jit(sp.lambdify((sym_z, sym_z_0, sym_sigma, sym_w), phi__))
U___ = jit(sp.lambdify((sym_z, sym_z_0, sym_sigma, sym_w), phi___))
U____ = jit(sp.lambdify((sym_z, sym_z_0, sym_sigma, sym_w), phi____))

# %%
z_f = 4.
z_init = -0.5

l_max = np.abs(z_init) + z_f

dz = 1e-6

z = np.arange(z_init, z_f + dz, dz)
steps = len(z)

# # %%
# Now we want to find many focal lengths for varying electron voltage parameters
u_0_values = u_l/np.linspace(0.8, 1.0, 3, endpoint = True)

colors = ['r', 'g', 'b', 'm', 'k', 'yellow']

u_l_over_u_0 = []
focal_lengths = []
magnifications_aber = []
magnifications_da = []
B_aber = []
B_da = []
z_object = []
plt.figure()

spot_fig, spot_ax = plt.subplots(nrows=1, ncols=2, figsize = (14, 6))
aber_fig, aber_ax = plt.subplots(nrows=2, ncols=2, figsize = (12, 12))

object_values = np.linspace(-0.01, -0.05, 20, endpoint = True)

for idx, U_0 in enumerate(u_0_values):
    for z_init in object_values:
        
        print('u_l/U_0 = ', u_l/U_0)

        z_g_ray, G, ig = odedopri_store(model,  -0.5,  np.array([1, 0]),  z_f,  1e-8,  1e-1,  1e-12, 10000, (w_best, means, sigmas, U_0, U, U_, U__))
        g, g_ = G[:ig, 0], G[:ig, 1]
        #plt.plot(z_g_ray[:ig], g, '.', color = 'k', label = 'Linearised ODE - g')
        z_h_ray, H, ih = odedopri_store(model,  z_init,  np.array([0, 1]),  z_f,  1e-8,  1e-1,  1e-12, 10000, (w_best, means, sigmas, U_0, U, U_, U__))
        h, h_ = H[:ih, 0], H[:ih, 1]
        #plt.plot(z_h_ray[:ih], h, '.', color = 'gray', label = 'Linearised ODE - h')
        
        fg = interpolate.CubicSpline(z_g_ray[:ig], g)
        fh = interpolate.CubicSpline(z_h_ray[:ih], h)
        
        #plt.plot(z_g_ray[:ig], fg(z_g_ray[:ig]))
        #plt.plot(z_h_ray[:ih], fh(z_h_ray[:ih]))
        
        #Get Focal Length when ray crossed radial axis via linear interpolation. 
        focal_length = fg.roots()[np.argmax(fg.roots()>0)]
        gaussian_image_plane = fh.roots()[np.argmax(fh.roots()>focal_length)]
        magnification = fg(gaussian_image_plane)
    
        z_g_ray, G, ig = odedopri_store(model,  z_init,  np.array([1, 0]),  gaussian_image_plane,  1e-8,  1e-1,  1e-15,  int(1e5), (w_best, means, sigmas, U_0, U, U_, U__))
        g, g_ = G[:ig, 0], G[:ig, 1]
        M_ = g[-1]
    
        z_h_ray, H, ih = odedopri_store(model,  z_init,  np.array([0, 1]),  gaussian_image_plane,  1e-8,  1e-1,  1e-15,  int(1e5), (w_best, means, sigmas, U_0, U, U_, U__))
        h, h_ = H[:ih, 0], H[:ih, 1]
    
        fg = interpolate.CubicSpline(z_g_ray[:ig], g)
        fg_ = interpolate.CubicSpline(z_g_ray[:ig], g_)
        fh = interpolate.CubicSpline(z_h_ray[:ih], h)
        fh_ = interpolate.CubicSpline(z_h_ray[:ih], h_)
        
        z = np.arange(z_init, gaussian_image_plane, dz)
        g, g_, h, h_ = fg(z), fg_(z), fh(z), fh_(z)
        
        focal_lengths.append(focal_length)
        magnifications_aber.append(M_)
        u_l_over_u_0.append(float(u_l/U_0))  
    
        Uz0 = U(z[0, None], means, sigmas, w_best).sum() - U_0
        U_val = U(z[:, None], means, sigmas, w_best).sum(1) - U_0
        U_val_ = U_(z[:, None], means, sigmas, w_best).sum(1)
        U_val__ = U__(z[:, None], means, sigmas, w_best).sum(1)
        U_val____ = U____(z[:, None], means, sigmas, w_best).sum(1)
        
        def L():
            return (1/(32*np.sqrt(U_val)))*((U_val__**2)/(U_val)-U_val____)
        
        def M():
            return (1/(8*np.sqrt(U_val)))*(U_val__)
        
        def N():
            return 1/2*(np.sqrt(U_val))
        
        def F_020():
            return (L()/4)*h*h*h*h + (M()/2)*h*h*h_*h_ + (N()/4)*h_*h_*h_*h_
        
        B_val = 4/np.sqrt(Uz0)*simpson(F_020(), z)
        B_aber.append(B_val) 
        
        DA.init(3, 4)
    
        x0 = 0
        y0 = 0
    
        x0_slope = 0
        y0_slope = 0
    
        x = array([x0 + DA(1), x0_slope + DA(2), y0 + DA(3), y0_slope + DA(4)])
    
        with DA.cache_manager():
            zf, x_f = odedopri(complete_ODE,  z_init,  x,  gaussian_image_plane,  1e-3, 1e-1, 1e-15,  int(1e6), (w_best, means, sigmas, U_0, U, U_, U__, U___, U____))
    
        Mag = x_f[0].getCoefficient([1, 0, 0, 0])
        B = x_f[0].getCoefficient([0, 3, 0, 0])
        
        z_object.append(z_init)
        B_da.append(B/Mag)
        magnifications_da.append(Mag)
        
X = np.array((u_l_over_u_0, B_da, B_aber, magnifications_da, magnifications_aber, focal_lengths, z_object)).T
#np.savetxt('Measured_Cs_aberint_and_DA_ULU0_0.8-1.0.txt', X)
np.savetxt('Measured_Cs_aberint_and_DA_ULU0_0.8-1.0_zobj_-0.01_to_-0.05.txt', X)

# plt.figure()
# plt.plot(z_object[:60], B_da[:60], label = 'UL/U0 = 0.5')
# plt.plot(z_object[60:120], B_da[60:120], label = 'UL/U0 = 0.6')
# plt.plot(z_object[120:180], B_da[120:180], label = 'UL/U0 = 0.7')
# plt.legend()
