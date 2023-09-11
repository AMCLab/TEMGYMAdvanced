
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
from scipy.constants import c, m_e, e
from scipy.optimize import newton
from da_spot_diagram import SphericalAberration
from matplotlib.ticker import FormatStrFormatter

plt.rc('font', family='Helvetica')

def spherical_aber_poly(m, C0, C1, C2, C3, C4):
    return C4*(1/(m**4))+C3*(1/m**3)+C2*(1/m**2)+C1/m+C0

def disc_of_confusion_poly(m, C0, C1, C2, C3, C4):
    return m*C0+C1+C2/m+C3/m**2+C4/m**3

eps = e/(2*m_e*c**2)
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
# fig, ax = plt.subplots(1, 2, figsize = (12, 5))
# ax[0].plot(z, u_z*-1, '-k', label='Input Potential')
# ax[0].set_ylabel('Voltage (V)')
# ax[0].set_xlabel('Z (m)')
# ylim = plt.ylim()

# ax[0].plot([],[], ls='-', c='#FFAAAA', label = 'Single Gaussian Function')
# ax[0].plot(z, norms*-1, ls='-', c='#FFAAAA')
# ax[0].plot(z, norms.sum(1)*-1, '-r', label='Sum of Gaussians')

# ax[0].text(0.97, 0.8,
#           "rms error = %.2g" % rms,
#           ha='right', va='top', transform=plt.gca().transAxes)
# ax[0].set_title('Nanomi Lens Potential - V = 1000V')

# ax[1].plot(z, abs(norms.sum(1)-(u_z)), 'gray')
# ax[1].set_ylabel('Error in Gaussian Fit (V)')
# ax[1].set_xlabel('Z (m)')
# ax[0].legend(loc=0, fontsize = 8)

# ax[0].set_ylim([0, 1000])
# ax[0].spines.right.set_visible(False)
# ax[0].spines.top.set_visible(False)
# ax[1].spines.right.set_visible(False)
# ax[1].spines.top.set_visible(False)
# ax[0].set_xlim([-0.02, 0.02])
# ax[1].set_xlim([-0.02, 0.02])
# ax[0].set_xticks(np.linspace(-0.02, 0.02, 5, endpoint = True))
# ax[1].set_xticks(np.linspace(-0.02, 0.02, 5, endpoint = True))
# plt.savefig('FitPotential.svg', dpi = 800)


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
           
ratios = np.array([0.8, 0.9, 1.0])

u_0_values = u_l/ratios
fig, ax = plt.subplots(1, 2, figsize = (14, 8))
ax = ax.ravel().ravel()

roots = []
colors = ['r', 'g', 'b']
# %%
for idx, U_0 in enumerate(u_0_values):
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
    
    a = -0.1
    b = 0.1
    dz = 1e-6
    z = np.arange(a, b, dz)
    
    z_r2_ray, R2, ir2 = odedopri_store(model,  a,  np.array([1, 0]),  b,  1e-8,  1e-1,  1e-15,  int(1e5), (w_best, means, sigmas, U_0, U, U_, U__))
    r2, r2_ = R2[:ir2, 0], R2[:ir2, 1]
    
    principal_ray_slope = r2_[-1]
    principal_ray_intercept = r2[-1]-principal_ray_slope*z[-1]
    
    zpo = (1-principal_ray_intercept)/(principal_ray_slope)
    zfi = -principal_ray_intercept/principal_ray_slope
    fi = abs(zpo) + zfi
    
    principal_ray_x = np.linspace(-0.01, z[-1], 100)
    principal_ray_y = principal_ray_slope*principal_ray_x+principal_ray_intercept
    
    fr2 = interpolate.CubicSpline(z_r2_ray[:ir2], r2)
    fr2_ = interpolate.CubicSpline(z_r2_ray[:ir2], r2_)
    
    r2, r2_ = fr2(z), fr2_(z)
    
    # z_r1_ray, R1, ir1 = odedopri_store(model,  a,  np.array([1, 0]),  b,  1e-8,  1e-1,  1e-15,  int(1e5), (np.flip(w_best), np.flip(means), np.flip(sigmas), U_0, U, U_, U__))
    # r1, r1_ = R1[:ir1, 0], R1[:ir1, 1]
    
    # principal_ray_slope = r1_[-1]
    # principal_ray_intercept = r1[-1]-principal_ray_slope*z[-1]
    
    zpo = (1-principal_ray_intercept)/(principal_ray_slope)
    zfi = -principal_ray_intercept/principal_ray_slope
    fi = abs(zpo) + zfi
    
    principal_ray_x = np.linspace(-0.01, z[-1], 100)
    principal_ray_y = principal_ray_slope*principal_ray_x+principal_ray_intercept
    
    # fr1 = interpolate.CubicSpline(z_r1_ray[:ir1], r1)
    # fr1_ = interpolate.CubicSpline(z_r1_ray[:ir1], r1_)
    
    r2, r2_ = fr2(z), fr2_(z)
    r1, r1_ = np.flip(fr2(z)), np.flip(fr2_(z))*-1
    
    # plt.figure()
    # plt.plot(z, r1, 'b')
    # plt.plot(z, r2, 'r')
    # plt.plot(z, r2_, 'r')
    # plt.plot(z, r1_, 'b')
    # plt.plot(principal_ray_x, principal_ray_y, color = 'k', linestyle = '--')
    # plt.vlines(zpo, 0, 1, color = 'k', linestyle = '--')
    
    Uz0 = U(z[0, None], means, sigmas, w_best).sum() - U_0
    U_val = U(z[:, None], means, sigmas, w_best).sum(1) - U_0
    U_val_ = U_(z[:, None], means, sigmas, w_best).sum(1)
    U_val__ = U__(z[:, None], means, sigmas, w_best).sum(1)
    U_val___ = U___(z[:, None], means, sigmas, w_best).sum(1)
    U_val____ = U____(z[:, None], means, sigmas, w_best).sum(1)
    M_star = -1
    
    def P():
        return np.sqrt(U_val/Uz0)*((5*U_val__**2)/(4*(U_val**2))+(5*U_val_**4)/(24*(U_val**4)))/16
    
    def Q():
        return np.sqrt(U_val/Uz0)*((14*U_val_**3)/(3*(U_val**3)))/16
    
    def R():
        return np.sqrt(U_val/Uz0)*(-1*(3*U_val_**2)/(2*(U_val)**2))/16
    
    def Cso0():
        return P()*(r1**4)+Q()*r1**3*r1_+R()*r1**2*r1_**2
    
    def Cso1():
        return 4*P()*r1**3*r2+Q()*(3*r1**2*r1_*r2+r1**3*r2_)+2*R()*(r1*r1_**2*r2+r1**2*r1_*r2_)
    
    def Cso2():
        return 6*P()*r1**2*r2**2 + 3*Q()*(r1*r1_*r2**2+r1**2*r2*r2_)+R()*(r1_**2*r2**2 + 4*r1*r1_*r2*r2_+r1**2*r2_**2)
    
    def Cso3():
        return 4*P()*r1*r2**3+Q()*(r1_*r2**3+3*r1*r2**2*r2_)+2*R()*(r1*r2*r2_**2+r1_*r2**2*r2_)
    
    def Cso4():
        return P()*(r2**4)+Q()*(r2**3*r2_)+R()*(r2**2*r2_**2)
    
    def Ae():
        return (1/64)*np.sqrt(U_val/Uz0)*(4*((U_val__/U_val)**2)-(U_val_*U_val___)/(U_val**2)-10*((U_val_/U_val)**2)*(U_val__/U_val)+10*(U_val_/U_val)**4)
    
    # def Ae_Pop():
    #     return (1/(192*np.sqrt(Uz0)))*(4*(3+5*eps*U_val)*(U_val__**2)/(U_val**(3/2))-(3+4*eps*U_val)*(U_val_*U_val___/(U_val**(3/2)))-30*(1+eps*U_val)*((U_val_**2)*U_val__)/(U_val**(5/2))+30*(U_val_**4)/U_val**(7/2))
                                               
    i_one = simpson(Ae()*r2**4, z)
    i_two = simpson(Ae()*r2**3*r1, z)-1/(8*fi**3)
    i_three = simpson(Ae()*r2**2*r1**2, z)-(np.sqrt(Uz0)/(24*fi**2))*simpson(U_val__/(U_val**(3/2)), z)
    i_four = 2*simpson(Ae()*r2**2*r1**2, z)-(np.sqrt(Uz0)/(24*fi**2))*simpson(U_val__/(U_val**(3/2)), z)
    i_five = simpson(Ae()*r2*r1**3, z)-1/(8*fi**3)
    i_six = simpson(Ae()*r1**4, z)
    
    Cso0_val = i_six*fi**4
    Cso1_val = -4*i_five*fi**4-fi/2
    Cso2_val = 2*(i_three+i_four)*fi**4
    Cso3_val = -4*i_two*fi**4-fi/2
    Cso4_val = i_one*fi**4
    
    print(Cso0_val, Cso1_val, Cso2_val, Cso3_val, Cso4_val)
    
    # Cso0_val = (r1_[0]**(-4))*simpson(Cso0(), z)
    # Cso1_val = -1*(r1[-1]/(r2[0]*(r1_[0])**4))*simpson(Cso1(), z)
    # Cso2_val = ((r1[-1]**2)/(r2[0]**2*(r1_[0])**4))*simpson(Cso2(), z)
    # Cso3_val = -1*((r1[-1]**3)/(r2[0]**3*(r1_[0])**4))*simpson(Cso3(), z)
    # Cso4_val = (r1[-1]/(r2[0]*r1_[0]))**4*simpson(Cso4(), z)
    
    # print(Cso0_val, Cso1_val, Cso2_val, Cso3_val, Cso4_val)
    
    m_star, Cso0, Cso1, Cso2, Cso3, Cso4 = sp.symbols('M*, Cso0, Cso1, Cso2, Cso3, Cso4')

    m_fit = np.linspace(-0.001, -10, 10000)
    
    # %%
    #ax[0].plot(m_fit, disc_of_confusion_poly(m_fit, (Cso0_val+Cso4_val)/2, (Cso1_val+Cso3_val)/2, Cso2_val, (Cso1_val+Cso3_val)/2, (Cso0_val+Cso4_val)/2), color = colors[idx], linestyle = '-.')
    #ax[0].plot(m_fit, disc_of_confusion_poly(m_fit, Cso0_val, Cso1_val, Cso2_val, Cso1_val, Cso0_val), color = colors[idx], linestyle = ':')
    ax[0].plot(m_fit, disc_of_confusion_poly(m_fit, Cso0_val, Cso1_val, Cso2_val, Cso3_val, Cso4_val)*1e3, color = colors[idx], label = 'Asymptotic ${{C_{{si}}}}$ - $U_L/U_0$ = {number:.{digits}f}'.format(number=u_l/U_0, digits=1), zorder = 0)
    ax[0].set_xlabel('Magnification (Asymptotic)')
    
    f_sym = disc_of_confusion_poly(m_star, Cso0, Cso1, Cso2, Cso3, Cso4)
    ax[0].set_ylabel('${C_{si}}$ (mm) (Asymptotic)') #Coefficient - ($Cso_{0} M* + Cso_{1} + Cso_{2}/{M*} + Cso_{3}/{M*^{2}} + Cso_{4}/{M*^{3}$)
    
    ax[0].spines.right.set_visible(False)
    ax[0].spines.top.set_visible(False)
    
    # ax[0].set_xlim([m_fit[0], m_fit[-1]])
    ax[0].set_ylim([-1.3, 0.01])
    
    f = disc_of_confusion_poly(m_star, Cso0_val, Cso1_val, Cso2_val, Cso3_val, Cso4_val)
    df = sp.diff(f, (m_star, 1))
    d_df = sp.diff(f, (m_star, 2))
    f_expr = sp.lambdify(m_star, f)
    df_expr = sp.lambdify(m_star, df)
    d_df_expr = sp.lambdify(m_star, d_df)
    
    poly_expr = sp.polys.polytools.poly_from_expr(df)
    our_root = newton(df_expr, -2, d_df_expr, maxiter = 50)
    roots.append(our_root)
    print(disc_of_confusion_poly(our_root, Cso0_val, Cso1_val, Cso2_val, Cso3_val, Cso4_val))
    print(disc_of_confusion_poly(our_root, Cso0_val, Cso1_val, Cso2_val, Cso3_val, Cso4_val)/our_root)
    ax[0].scatter(our_root, disc_of_confusion_poly(our_root, Cso0_val, Cso1_val, Cso2_val, Cso3_val, Cso4_val)*1e3, facecolor = 'w', edgecolor = colors[idx], label = 'Minimum ${{C_{{si}}}}$ Coefficient - $U_L/U_0$ = {number:.{digits}f}'.format(number=u_l/U_0, digits=1), zorder = 1)
    
    poly_expr = sp.polys.polytools.poly_from_expr(df)
    
    ax[0].set_xlim([-6.0,0])
    ax[0].set_ylim([-3*1e3, 1.0*1e3])
    ax[0].legend(fontsize = 10)
    ax[0].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # ax[0].set_yscale('symlog')
    

mag_data = np.loadtxt('RealMag_Realf_AsympMag_Asympf_zobject_0.8-1.0_closer.txt')
asymptotic_magnifications, z_object = mag_data[:, 2], mag_data[:, 6]

fmag_asymp = interpolate.CubicSpline(asymptotic_magnifications[:500], z_object[:500])

z_object_with_smallest_disc = fmag_asymp(roots[0])

#ax[1].scatter(asymptotic_magnifications[:500], z_object[:500], facecolor = 'w', edgecolor = colors[0], label = 'Asymptotic Magnification - $U_L/U_0$ = 0.8', zorder = 0)
ax[1].plot(asymptotic_magnifications[:500], z_object[:500]*1e3, color = colors[0], zorder = 0, label = 'Asymptotic Magnification vs Object Location - $U_L/U_0$ = 0.8') 
ax[1].vlines(roots[0], -0.125*1e3, z_object_with_smallest_disc*1e3, linestyle = '--', color = colors[0], zorder = 1)
ax[1].scatter(roots[0], z_object_with_smallest_disc*1e3, facecolor = 'w', edgecolor = colors[0], zorder = 1, label = 'Object Location for Minimum ${C_{si}}$ Coefficient = ' + str(np.round(z_object_with_smallest_disc*1e3, 3)) + ' (mm) - $U_L/U_0$ = {number:.{digits}f}'.format(number=u_l/u_0_values[0], digits=1))


fmag_asymp = interpolate.CubicSpline(asymptotic_magnifications[500:1000], z_object[500:1000])

z_object_with_smallest_disc = fmag_asymp(roots[1])

#ax[1].scatter(asymptotic_magnifications[500:1000], z_object[500:1000], facecolor = 'w', edgecolor = colors[1], label = 'Asymptotic Magnification - $U_L/U_0$ = 0.9', zorder = 0)
ax[1].plot(asymptotic_magnifications[500:1000], z_object[500:1000]*1e3, color = colors[1], zorder = 0, label = 'Asymptotic Magnification vs Object Location - $U_L/U_0$ = 0.9') 
ax[1].vlines(roots[1], -0.125*1e3, z_object_with_smallest_disc*1e3, linestyle = '--', color = colors[1], zorder = 1)
ax[1].scatter(roots[1], z_object_with_smallest_disc*1e3, facecolor = 'w', edgecolor = colors[1], zorder = 1, label = 'Object Location for Minimum ${C_{si}}$ Coefficient = ' + str(np.round(z_object_with_smallest_disc*1e3, 3)) + ' (mm) - $U_L/U_0$ = {number:.{digits}f}'.format(number=u_l/u_0_values[1], digits=1))

fmag_asymp = interpolate.CubicSpline(asymptotic_magnifications[1000:1500], z_object[1000:1500])

z_object_with_smallest_disc = fmag_asymp(roots[2])

#ax[1].scatter(asymptotic_magnifications[500:1000], z_object[500:1000], facecolor = 'w', edgecolor = colors[1], label = 'Asymptotic Magnification - $U_L/U_0$ = 0.9', zorder = 0)
ax[1].plot(asymptotic_magnifications[1000:1500], z_object[1000:1500]*1e3, color = colors[2], zorder = 0, label = 'Asymptotic Magnification vs Object Location - $U_L/U_0$ = 1.0') 
ax[1].vlines(roots[2], -0.125*1e3, z_object_with_smallest_disc*1e3, linestyle = '--', color = colors[2], zorder = 1)
ax[1].scatter(roots[2], z_object_with_smallest_disc*1e3, facecolor = 'w', edgecolor = colors[2], zorder = 1, label = 'Object Location for Minimum ${C_{si}}$ Coefficient = ' + str(np.round(z_object_with_smallest_disc*1e3, 3)) + ' (mm) - $U_L/U_0$ = {number:.{digits}f}'.format(number=u_l/u_0_values[2], digits=1))

ax[1].set_xlim([-4.0, 0])
ax[1].set_ylim([-0.125*1e3, 0.1*1e3])

ax[1].set_ylabel('Object Distance (mm) (Real)')
ax[1].set_xlabel('Magnification (Asymptotic)')
ax[1].spines.right.set_visible(False)
ax[1].spines.top.set_visible(False)
ax[1].legend(fontsize = 10, bbox_to_anchor=(1.05, 1))


fig.savefig('minimum_disc_radius.svg', dpi = 800)

plt.show()
