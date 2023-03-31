# %%
import sympy as sp
import numpy as np

import matplotlib.pyplot as plt
from generate_expression import generate_n_gauss_expr
from sum_of_norms import sum_of_norms, norm, symbolic_norm
from numba import jit
from Laplace import AnalyticalLaplace
from odedopri import odedopri_store
from scipy import interpolate 
import time
from tqdm import tqdm

t0 = time.time()
u_l = -1000

# '''#1 PYFEMM Test'''
pyfemm_data = np.loadtxt('Nanomi_Lens_Default_Mesh_On_ZAxis_0.02Vacuum.txt')
u_z = pyfemm_data[:, 1]*-1
z = (pyfemm_data[:, 0]-50)*1e-3

plt.figure()
plt.plot(z, u_z, '-r', alpha = 0.5)#, label = 'FEMM')
plt.legend()

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
plt.plot(z, u_z, '-k', label='input potential')
plt.ylabel('Voltage (V)')
plt.xlabel('Z axis (m)')
ylim = plt.ylim()

# plt.plot(z, norms, ls='-', c='#FFAAAA')
plt.plot(z, norms.sum(1), '-r', label='sum of gaussians')

# plt.text(0.97, 0.8,
#           "rms error = %.2g" % rms,
#           ha='right', va='top', transform=plt.gca().transAxes)
#plt.title("Fit to Potential with a Sum of %i Gaussians" % n_gaussians)
plt.title('Nanomi Lens Potential - V central_electrode = 1000V')
# plt.legend(loc=0)

@jit(nopython = True)
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

U = jit(sp.lambdify((sym_z, sym_z_0, sym_sigma, sym_w), phi))
U_ = jit(sp.lambdify((sym_z, sym_z_0, sym_sigma, sym_w), phi_))
U__ = jit(sp.lambdify((sym_z, sym_z_0, sym_sigma, sym_w), phi__))

# %%
z_f = 5.
z_init = -0.05

l_max = np.abs(z_init) + z_f

dz = 1e-5

z = np.arange(z_init, z_f + dz, dz)
steps = len(z)

# # %%
# Now we want to find many focal lengths for varying electron voltage parameters
u_0_values = u_l/np.linspace(0.8, 1.0, 3, endpoint = True)

u_l_over_u_0 = []
real_focal_lengths = []
asymptotic_focal_lengths = []
real_magnifications = []
asymptotic_magnifications = []
z_object = []
z_image_real = []
z_image_asymp = []

#object_positions = np.linspace(-0.1, -0.6, 60)
object_positions = np.linspace(-0.0001, -0.125, 500)
fig, ax = plt.subplots()

colors = ['r', 'g', 'k']
i = 0
plt.figure()
for idx, U_0 in tqdm(enumerate(u_0_values)):
    print(idx)
    print('u_l/U_0 = ', u_l/U_0)

    z_g_ray, G, ig = odedopri_store(model,  z_init,  np.array([1, 0]),  z_f,  1e-8,  1e-1,  1e-12, 10000, (w_best, means, sigmas, U_0, U, U_, U__))
    g, g_ = G[:ig, 0], G[:ig, 1]
    
    fg = interpolate.CubicSpline(z_g_ray[:ig], g)
    focal_length = fg.roots()[np.argmax(fg.roots()>0)]
    
    ii = 0
    for object_position in tqdm(object_positions):
        print('u_l/U_0 = ', u_l/U_0)
        
        z_init = object_position-focal_length
        
        z_g_ray, G, ig = odedopri_store(model,  z_init,  np.array([1, 0]),  z_f,  1e-8,  1e-1,  1e-12, 10000, (w_best, means, sigmas, U_0, U, U_, U__))
        g, g_ = G[:ig, 0], G[:ig, 1]
        # plt.plot(z_g_ray[:ig], g, '.', color = 'k', label = 'Linearised ODE - g')
        z_h_ray, H, ih = odedopri_store(model,  z_init,  np.array([0, 1]),  z_f,  1e-8,  1e-1,  1e-12, 10000, (w_best, means, sigmas, U_0, U, U_, U__))
        h, h_ = H[:ih, 0], H[:ih, 1]
        # plt.plot(z_h_ray[:ih], h, '.', color = 'gray', label = 'Linearised ODE - h')
        
        fg = interpolate.CubicSpline(z_g_ray[:ig], g)
        # plt.plot(z_g_ray[:ig], fg(z_g_ray[:ig]))
        fh = interpolate.CubicSpline(z_h_ray[:ih], h)
        # plt.plot(z_h_ray[:ih], fh(z_h_ray[:ih]))
        
        #Get Focal Length when ray crossed radial axis via linear interpolation. 
        real_focal_length = fg.roots()[np.argmax(fg.roots()>0)]
        real_gaussian_image_plane = fh.roots()[np.argmax(fh.roots()>real_focal_length)]
        real_magnification = fg(real_gaussian_image_plane)
        
        # plt.plot(real_focal_length, 0, '.m')
        # plt.hlines(0, z_init, z_f, 'r')
        # plt.vlines(real_gaussian_image_plane, 0, real_magnification)
        
        principal_gray_slope = g_[-1]
        principal_gray_intercept = g[-1]-principal_gray_slope*z[-1]
    
        real_z_po = (1-principal_gray_intercept)/(principal_gray_slope)
        real_z_pi = -real_z_po
        real_zf = abs(real_z_po)+real_focal_length
        
        principal_hray_slope = h_[-1]
        principal_hray_intercept = h[-1]-principal_hray_slope*z[-1]
        
        asymptotic_focal_length = -principal_gray_intercept/principal_gray_slope
        asymptotic_gaussian_image_plane = -principal_hray_intercept/principal_hray_slope
        asymptotic_magnification = principal_gray_slope*asymptotic_gaussian_image_plane+principal_gray_intercept
        
        asymptotic_zf = abs(real_z_po)+asymptotic_focal_length
        
        print('z_init = ' + str(z_init))
        print('asympy_mag = ' + str(asymptotic_magnification))
        print('real_mag = ' + str(real_magnification))
        # print(asymptotic_zf-(asymptotic_focal_length)/asymptotic_magnification)
        # print(-real_zf-(-real_focal_length)/real_magnification)
        
        # plt.plot(asymptotic_focal_length, 0, '.g')
        # plt.plot(asymptotic_gaussian_image_plane, 0, '.b')
        # plt.vlines(asymptotic_gaussian_image_plane, 0, asymptotic_magnification, 'r')
        # plt.hlines(asymptotic_magnification, real_z_pi, asymptotic_gaussian_image_plane, linestyle = '--')
        # principal_ray_x = np.linspace(-0.1, real_focal_length, 100)
        # principal_ray_y = principal_ray_slope*principal_ray_x
        

        # ax.scatter(real_magnification, z_init, facecolor = 'w', edgecolor = colors[idx], label = 'Real Magnification', zorder = 1)
        # ax.scatter(asymptotic_magnification, z_init, facecolor = 'w', edgecolor = colors[idx], label = 'Asymptotic Magnification', zorder = 1)

        real_magnifications.append(real_magnification)
        real_focal_lengths.append(real_focal_length)
        asymptotic_magnifications.append(asymptotic_magnification)
        asymptotic_focal_lengths.append(asymptotic_focal_length)
        z_object.append(z_init)
        z_image_real.append(real_gaussian_image_plane)
        z_image_asymp.append(asymptotic_gaussian_image_plane)
        u_l_over_u_0.append(u_l/U_0)
        
        # plt.plot(z_object, z_image_real)
        # plt.plot(z_object, z_image_asymp)
        # fmag_real = interpolate.CubicSpline(real_magnifications, z_object)
        # fmag_asympy = interpolate.CubicSpline(asymptotic_magnifications, z_object)


        i=i+1
        

    # ax.plot(real_magnifications, fmag_real(real_magnifications), 'gray')   
    # ax.plot(asymptotic_magnifications, fmag_asympy(asymptotic_magnifications), 'r') 

#ax.set_title('Object Distance vs Magnification')
#ax.set_ylabel('Object Distance (m)')
#ax.set_xlabel('Magnificiation (m) (Real & Asymptotic)')
# ax.legend()
    
X = np.array([real_magnifications, real_focal_lengths, asymptotic_magnifications, asymptotic_focal_lengths, z_image_real, z_image_asymp, z_object, u_l_over_u_0]).T
#np.savetxt('RealMag_Realf_AsympMag_Asympf_zobject_0.8-1.0.txt', X)
np.savetxt('RealMag_Realf_AsympMag_Asympf_zobject_0.8-1.0_closer.txt', X)



