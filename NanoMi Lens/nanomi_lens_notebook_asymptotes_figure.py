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
import matplotlib
from matplotlib.patches import Rectangle
    
def zoom_outside(srcax, roi, dstax, color="red", linewidth=1, roiKwargs={}, arrowKwargs={}):
    '''Create a zoomed subplot outside the original subplot
    
    srcax: matplotlib.axes
        Source axis where locates the original chart
    dstax: matplotlib.axes
        Destination axis in which the zoomed chart will be plotted
    roi: list
        Region Of Interest is a rectangle defined by [xmin, ymin, xmax, ymax],
        all coordinates are expressed in the coordinate system of data
    roiKwargs: dict (optional)
        Properties for matplotlib.patches.Rectangle given by keywords
    arrowKwargs: dict (optional)
        Properties used to draw a FancyArrowPatch arrow in annotation
    '''
    roiKwargs = dict([("fill", False), ("linestyle", "dashed"),
                      ("color", color), ("linewidth", linewidth)]
                     + list(roiKwargs.items()))
    arrowKwargs = dict([("arrowstyle", "-"), ("color", color),
                        ("linewidth", linewidth)]
                       + list(arrowKwargs.items()))
    # draw a rectangle on original chart
    srcax.add_patch(Rectangle([roi[0], roi[1]], roi[2]-roi[0], roi[3]-roi[1], 
                            **roiKwargs))
    # get coordinates of corners
    srcCorners = [[roi[0], roi[1]], [roi[0], roi[3]],
                  [roi[2], roi[1]], [roi[2], roi[3]]]
    dstCorners = dstax.get_position().corners()
    srcBB = srcax.get_position()
    dstBB = dstax.get_position()
    # find corners to be linked
    if srcBB.max[0] <= dstBB.min[0]: # right side
        if srcBB.min[1] < dstBB.min[1]: # upper
            corners = [1, 2]
        elif srcBB.min[1] == dstBB.min[1]: # middle
            corners = [0, 1]
        else:
            corners = [0, 3] # lower
    elif srcBB.min[0] >= dstBB.max[0]: # left side
        if srcBB.min[1] < dstBB.min[1]:  # upper
           corners = [0, 3]
        elif srcBB.min[1] == dstBB.min[1]: # middle
            corners = [2, 3]
        else:
            corners = [1, 2]  # lower
    elif srcBB.min[0] == dstBB.min[0]: # top side or bottom side
        if srcBB.min[1] < dstBB.min[1]:  # upper
            corners = [0, 2]
        else:
            corners = [1, 3] # lower
    else:
        RuntimeWarning("Cannot find a proper way to link the original chart to "
                       "the zoomed chart! The lines between the region of "
                       "interest and the zoomed chart wiil not be plotted.")
        return
    # plot 2 lines to link the region of interest and the zoomed chart
    for k in range(2):
        srcax.annotate('', xy=srcCorners[corners[k]], xycoords="data",
            xytext=dstCorners[corners[k]], textcoords="figure fraction",
            arrowprops=arrowKwargs)
        
plt.rc('font', family='Helvetica')
#matplotlib.pyplot.switch_backend('Qt5Agg')

t0 = time.time()
u_l = -1000

# '''#1 PYFEMM Test'''
pyfemm_data = np.loadtxt('Nanomi_Lens_Default_Mesh_On_ZAxis_0.02Vacuum.txt')
u_z = pyfemm_data[:, 1]*-1
z_data = (pyfemm_data[:, 0]-50)*1e-3

plt.figure()
plt.plot(z_data , u_z, '-r', alpha = 0.5)#, label = 'FEMM')
plt.legend()

# %%
'''Sum of Norms Fit'''
n_gaussians = 201
w_best, rms, means, sigmas = sum_of_norms(z_data , u_z, n_gaussians,
                                          spacing='linear',
                                      full_output=True)

norms = w_best * norm(z_data[:, None], means, sigmas)

phi_fit = generate_n_gauss_expr(w_best, means, sigmas)
phi_lambda = sp.lambdify(sp.abc.z, phi_fit)

# plot the results
plt.plot(z_data , u_z, '-k', label='input potential')
plt.ylabel('Voltage (V)')
plt.xlabel('Z axis (m)')
ylim = plt.ylim()

# plt.plot(z, norms, ls='-', c='#FFAAAA')
plt.plot(z_data , norms.sum(1), '-r', label='sum of gaussians')

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

fig, ax = plt.subplots(2, 1, figsize = (9,9))
plt.subplots_adjust(hspace=0.5)
# fig.tight_layout(pad=8.0)
ax2 = ax[0].twinx()
# ax3=fig.add_subplot(111)
z_g_ray, G, ig = odedopri_store(model,  z_init,  np.array([1, 0]),  z_f,  1e-8,  1e-1,  1e-13, 10000, (w_best, means, sigmas, U_0, U, U_, U__))
g, g_ = G[:ig, 0], G[:ig, 1]
# plt.plot(z_g_ray[:ig], g, '-', color = 'g', label = 'Linearised ODE - g')
z_h_ray, H, ih = odedopri_store(model,  z_init,  np.array([0, 1]),  z_f,  1e-8,  1e-1,  1e-13, 10000, (w_best, means, sigmas, U_0, U, U_, U__))
h, h_ = H[:ih, 0], H[:ih, 1]
# plt.plot(z_h_ray[:ih], h, '-', color = 'b', label = 'Linearised ODE - h')

fg = interpolate.CubicSpline(z_g_ray[:ig], g)
fh = interpolate.CubicSpline(z_h_ray[:ih], h)

#Get Focal Length when ray crossed radial axis via linear interpolation. 
real_focal_length = fg.roots()[np.argmax(fg.roots()>0)]
real_gaussian_image_plane = fh.roots()[np.argmax(fh.roots()>real_focal_length)]
real_magnification = fg(real_gaussian_image_plane)

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

principal_ray_xh = np.linspace(asymptotic_gaussian_image_plane, z_f, 100)
principal_ray_yh = principal_hray_slope*principal_ray_xh+principal_hray_intercept

principal_ray_xg = np.linspace(asymptotic_focal_length, z_f, 100)
principal_ray_yg = principal_gray_slope*principal_ray_xg+principal_gray_intercept

ax[0].plot(z_g_ray[:ig]*1e3, fg(z_g_ray[:ig])*1e3, color = 'dodgerblue', label = 'g')
ax[0].plot(z_h_ray[:ih]*1e3, fh(z_h_ray[:ih])*1e3, color = 'r', label = 'h')
ax[0].plot(principal_ray_xh*1e3, principal_ray_yh*1e3, '--', color = 'indianred', label = 'h asymptote')
ax[0].plot(principal_ray_xg*1e3, principal_ray_yg*1e3, '--', color = 'skyblue', label = 'g asymptote')

ax[0].arrow(z_init*1e3, 0, 0, 1*1e3, color = 'k', width = 0.75, length_includes_head = True, head_length = 100, zorder = 10)
ax[0].arrow(asymptotic_gaussian_image_plane*1e3, 0, 0, asymptotic_magnification*1e3, color = 'gray', width = 0.01, length_includes_head = True, head_length = 10, zorder = 10)
ax[0].arrow(real_gaussian_image_plane*1e3, 0, 0, real_magnification*1e3, color = 'k', width = 0.01, length_includes_head = True, head_length = 10, zorder = 10)
ax[0].scatter([],[], c='k',marker=r'$\uparrow$',s=20, label='Real Object')
ax[0].scatter([],[], c='k',marker=r'$\downarrow$',s=20, label='Real Image')
ax[0].scatter([],[], c = 'gray',marker=r'$\downarrow$',s=20, label='Asymptotic Image')
ax[0].hlines(0, z_init*1e3, z_f*1e3, 'k', '-')
ax[0].vlines(real_gaussian_image_plane*1e3, 0, real_magnification)
ax[0].vlines(asymptotic_gaussian_image_plane*1e3, 0, asymptotic_magnification, 'r')

# axin = .inset_axes([0.55, 0.55, 0.4, 0.4])
ax[1].arrow(asymptotic_gaussian_image_plane*1e3, 0, 0, asymptotic_magnification*1e3, color = 'gray', width = 0.0025, length_includes_head = True, head_length = 10, zorder = 10)
ax[1].arrow(real_gaussian_image_plane*1e3, 0, 0, real_magnification*1e3, color = 'k', width = 0.0025, length_includes_head = True, head_length = 10, zorder = 10)
ax[1].hlines(0, z_init*1e3, z_f*1e3, 'k', '-')
ax[1].plot(z_g_ray[:ig]*1e3, fg(z_g_ray[:ig])*1e3, color = 'dodgerblue')
ax[1].plot(z_h_ray[:ih]*1e3, fh(z_h_ray[:ih])*1e3, color = 'r')
ax[1].plot(principal_ray_xg*1e3, principal_ray_yg*1e3, '--', color = 'skyblue')
ax[1].plot(principal_ray_xh*1e3, principal_ray_yh*1e3, '--', color = 'indianred')

xticks = [0.0034*1e3, 0.0035*1e3, 0.0036*1e3, 0.0037*1e3, 0.0038*1e3]
ax[1].set_xticks(xticks)
ax[1].set_xticklabels(xticks, rotation = 0)
# Plot the data on the inset axis and zoom in on the important part
ax[1].set_xlim(0.0034*1e3, 0.0038*1e3)
ax[1].set_ylim(-0.2*1e3, 0.01*1e3)

ax[0].set_xlabel('Z (mm)')
ax[0].set_ylabel('R (mm)')
ax[1].set_xlabel('Z (mm)')
ax[1].set_ylabel('R (mm)')
ax2.set_xlabel('Z (mm)')
ax2.set_ylabel('Voltage (V)')


ax[0].set_title("First Order Properties of NanoMi Symmetric Lens")
ax[1].set_title("Inset Axis - Real & Asymptotic Image")
ax2.set_ylim([-0.02*1e3, 0.02*1e3])
# ax2.yaxis.set_label_coords(1.10,0.5)

ax[0].spines.right.set_visible(False)
ax[0].spines.top.set_visible(False)
ax2.spines.top.set_visible(False)
ax[1].spines.right.set_visible(False)
ax[1].spines.top.set_visible(False)

ax2.plot(z_data*1e3, norms.sum(1), color = 'gray', linestyle = '-', label = 'Lens Potential', zorder = 0, alpha = 0.7)
fig.legend(fontsize = '9', bbox_to_anchor=(0.9, 1.0))
zoom_outside(srcax=ax[0], roi=[0.0035*1e3, -0.2*1e3, 0.0038*1e3, 0.01*1e3], dstax=ax[1], color="gray", arrowKwargs=dict([("alpha", 0.5)]))
# ax[1].set_xlim(0.0035, 0.0038)
# ax[1].set_ylim(-0.2, 0.01)

# ax[1].indicate_inset_zoom(axin)
ax[0].set_xlim([-0.055*1e3, 0.03*1e3])
ax[0].set_ylim([-1.5*1e3, 1.5*1e3])
ax2.set_ylim([-1000, 1000])
plt.savefig('Asymptotic_Figure.svg')
# magnifications.append(asymptotic_magnification)
# focal_lengths.append(asymptotic_zf)
# u_l_over_u_0.append(u_l/U_0)  

