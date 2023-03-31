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

plt.rc('font', family='Helvetica')
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

dz = 1e-7

z = np.arange(z_init, z_f + dz, dz)
steps = len(z)

# # %%
# Now we want to find many focal lengths for varying electron voltage parameters
u_0_values = u_l/np.linspace(0.1, 1.0, 100, endpoint = True)

u_l_over_u_0 = []
focal_lengths = []
magnifications = []

plt.figure()

for U_0 in u_0_values:
    print('u_l/U_0 = ', u_l/U_0)
    
    z_g_ray, G, ig = odedopri_store(model,  z_init,  np.array([1, 0]),  z_f,  1e-5,  1e-1,  1e-12, 10000, (w_best, means, sigmas, U_0, U, U_, U__))
    g, g_ = G[:ig, 0], G[:ig, 1]
    plt.plot(z_g_ray[:ig], g, '.', color = 'k', label = 'Linearised ODE - g')
    z_h_ray, H, ih = odedopri_store(model,  z_init,  np.array([0, 1]),  z_f,  1e-5,  1e-1,  1e-12, 10000, (w_best, means, sigmas, U_0, U, U_, U__))
    h, h_ = H[:ih, 0], H[:ih, 1]
    plt.plot(z_h_ray[:ih], h, '.', color = 'gray', label = 'Linearised ODE - h')
    
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
    
    principal_ray_x = np.linspace(asymptotic_focal_length-0.01, z_f, 100)
    principal_ray_y = principal_hray_slope*principal_ray_x+principal_hray_intercept
    # plt.plot(principal_ray_x, principal_ray_y, '--k')
    
    principal_ray_x = np.linspace(real_focal_length-0.01, z_f, 100)
    principal_ray_y = principal_gray_slope*principal_ray_x+principal_gray_intercept
    # plt.plot(principal_ray_x, principal_ray_y, '--k')
    
    asymptotic_zf = abs(real_z_po)+asymptotic_focal_length
    
    print(z_init)
    print(asymptotic_zf-(asymptotic_focal_length)/asymptotic_magnification)
    print(-real_zf-(-real_focal_length)/real_magnification)
    
    # plt.plot(asymptotic_focal_length, 0, '.g')
    # plt.plot(asymptotic_gaussian_image_plane, 0, '.b')
    # plt.vlines(asymptotic_gaussian_image_plane, 0, asymptotic_magnification, 'r')
    # plt.hlines(asymptotic_magnification, real_z_pi, asymptotic_gaussian_image_plane, linestyle = '--')
    # plt.plot([-(asymptotic_zf-(asymptotic_focal_length)/asymptotic_magnification), -asymptotic_focal_length, real_z_pi], [1, 0, asymptotic_magnification], '--')
    # principal_ray_x = np.linspace(-0.1, real_focal_length, 100)
    # principal_ray_y = principal_ray_slope*principal_ray_x
    
    magnifications.append(asymptotic_magnification)
    focal_lengths.append(asymptotic_zf)
    u_l_over_u_0.append(u_l/U_0)  

X = np.hstack([u_l_over_u_0, focal_lengths])
np.savetxt('ULU0andFocalLengths.txt', X)
# # # # %%
# print(u_l_over_u_0)

# # %%
fig, ax = plt.subplots(figsize = (12, 6))
ax.set_xlim([0, 1.1])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
nanomi_data = np.loadtxt("nanomi_symmetric_lens_focal_length_data.txt")
rempfer_data = np.loadtxt("rempfer_symmetric_lens_focal_length_data.txt")

u_l_over_u_0_nanomi = np.round(nanomi_data[:, 0],1)
focal_lengths_nanomi = (nanomi_data[:, 1])
u_l_over_u_0_rempfer = rempfer_data[:, 0]
focal_lengths_rempfer = (rempfer_data[:, 1])*1e3

ax.plot(u_l_over_u_0, np.array(focal_lengths)*1e3, '-', color = 'blue', label = 'Our Focal Length - Linear Interpolation', zorder = 0)
ax.scatter(u_l_over_u_0, np.array(focal_lengths)*1e3, color = 'k', s = 2, label = 'Our Focal Length - Datapoint', zorder = 0)
ax.scatter(u_l_over_u_0_nanomi, focal_lengths_nanomi, color = 'r', alpha = 0.5, label = 'Nanomi Focal Length', edgecolor = 'k', zorder = 1, marker = 's')
ax.scatter(u_l_over_u_0_rempfer, focal_lengths_rempfer, color = 'g', alpha = 0.5, label = 'Rempfer Focal Length', edgecolor = 'k', zorder = 1)

# Create an inset axis in the bottom right corner
axin = ax.inset_axes([0.3, 0.2, 0.7, 0.7])

# Plot the data on the inset axis and zoom in on the important part
axin.plot(u_l_over_u_0, np.array(focal_lengths)*1e3, '-b', label = 'My data', zorder = 0)
axin.scatter(u_l_over_u_0, np.array(focal_lengths)*1e3, color = 'k', s = 4, zorder = 0)
axin.scatter(u_l_over_u_0_nanomi, focal_lengths_nanomi, color = 'r', alpha = 0.5, label = 'Nanomi data', edgecolor = 'k', zorder = 1, marker = 's')
axin.scatter(u_l_over_u_0_rempfer, focal_lengths_rempfer, color = 'g', alpha = 0.5, label = 'Rempfer data', edgecolor = 'k', zorder = 1)

axin.set_xlim(0.49, 1.02)
axin.set_ylim(0., 60)

# Add the lines to indicate where the inset axis is coming from
ax.indicate_inset_zoom(axin)
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
axin.spines.right.set_visible(False)
axin.spines.top.set_visible(False)
                            
ax.set_xticks(np.linspace(0., 1., 11, endpoint = True))
ax.set_xlabel('$U_L/U_0$')
ax.set_ylabel('Focal Length (mm)')
ax.set_title('Symmetric Lens Focal Length')
ax.legend(loc='upper center', bbox_to_anchor=(1.05, 1.1),
          ncol=1, fancybox=True, shadow=True, fontsize = 8)
fig.savefig('FocalLength.svg', dpi = 800)

t1 = time.time()
print(t1-t0)
# # %%

# # %% plot rempfers' Cs Values
# spherical_aberration_data = np.loadtxt("nanomi_lens_SfSg.txt", delimiter=" ")

# UL_U0, Sf, Sg = spherical_aberration_data[:, 0], spherical_aberration_data[:, 1], spherical_aberration_data[:, 2]

# convert_to_cs_image = lambda m, f, Sf, Sg: -1*((1+m**2)*Sg + 2*m*Sf)*((1+m)**2)*(f/4)
# convert_to_cs_object = lambda m, f, Sf, Sg: -1*((1+(1/(m**2)))*Sg + (2/m)*Sf)*((1+1/m)**2)*(f/4)

# focal_lengths = np.array(focal_lengths)
# magnifications = abs(np.array(magnifications))

# test = convert_to_cs_image(5.2, 0.0508, -50, -45)*1000
# plt.figure()
# plt.title('Cs wrt Image')
# plt.xlabel('UL/U0')
# plt.ylabel('Csi (m)')
# cs_image = convert_to_cs_image(magnifications, focal_lengths, Sf, Sg)
# plt.plot(UL_U0, cs_image)

# plt.figure()
# plt.title('Cs wrt Object')
# plt.xlabel('UL/U0')
# plt.ylabel('Cso (m)')
# cs_object = convert_to_cs_object(magnifications, focal_lengths, Sf, Sg)
# plt.plot(UL_U0, cs_object)

