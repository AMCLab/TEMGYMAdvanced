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

def spherical_aber_reducued_poly(m, C0, C1, C2):
    return C0*(1+1/(m**4))+C1*((1/m**3)+1/m)+C2*(1/m**2)

def spherical_aber_poly(m, C0, C1, C2, C3, C4):
    return C4*(1/(m**4))+C3*(1/m**3)+C2*(1/m**2)+C1/m+C0

def disc_of_confusion_poly(m, C0, C1, C2, C3, C4):
    return m*C0+C1+C2/m+C3/m**2+C4/m**3

plt.rc('font', family='Helvetica')

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
u_0_values = u_l/np.linspace(0.5, 1.0, 6, endpoint = True)
ul_U0 = []
C0 = []
C1 = []
C2 = []
C3 = []
C4 = []

for U_0 in u_0_values:

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
    

    zfi = -principal_ray_intercept/principal_ray_slope
    fi = -1/(r2_[-1])
    zpi = zfi-fi
    
    principal_ray_x = np.linspace(-0.01, z[-1], 100)
    principal_ray_y = principal_ray_slope*principal_ray_x+principal_ray_intercept
    
    r2, r2_ = fr2(z), fr2_(z)
    r1, r1_ = np.flip(fr2(z)), np.flip(fr2_(z))*-1
    
    plt.figure()
    plt.plot(z, r1, 'b')
    plt.plot(z, r2, 'r')
    plt.plot(z, r2_, 'r')
    plt.plot(z, r1_, 'b')
    plt.plot(principal_ray_x, principal_ray_y, color = 'k', linestyle = '--')
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
        return 1/64*np.sqrt(U_val/Uz0)*(4*(U_val__/U_val)**2-(U_val_*U_val___)/(U_val**2)-10*(U_val_/U_val)**2*(U_val__/U_val)+10*(U_val_/U_val)**4)
    
    i_one = simpson(Ae()*r2**4, z)
    i_two = simpson(Ae()*r2**3*r1, z)-1/(8*fi**3)
    i_three = simpson(Ae()*r2**2*r1**2, z)-(np.sqrt(Uz0)/(24*fi**2))*simpson(U_val__/(U_val**(3/2)), z)
    i_four = 2*simpson(Ae()*r2**2*r1**2, z)-(np.sqrt(Uz0)/(24*fi**2))*simpson(U_val__/(U_val**(3/2)), z)
    i_five = simpson(Ae()*r2*r1**3, z)-1/(8*fi**3)
    i_six = simpson(Ae()*r1**4, z)
    
    Cso0_val_orloff = i_six*fi**4
    Cso1_val_orloff = -4*i_five*fi**4-fi/2
    Cso2_val_orloff = 2*(i_three+i_four)*fi**4
    Cso3_val_orloff = -4*i_two*fi**4-fi/2
    Cso4_val_orloff = i_one*fi**4
    
    print(Cso0_val_orloff, Cso1_val_orloff, Cso2_val_orloff, Cso3_val_orloff, Cso4_val_orloff)
    
    Cso0_val = (r1_[0]**(-4))*simpson(Cso0(), z)
    Cso1_val = -1*(r1[-1]/(r2[0]*(r1_[0])**4))*simpson(Cso1(), z)
    Cso2_val = ((r1[-1]**2)/(r2[0]**2*(r1_[0])**4))*simpson(Cso2(), z)
    Cso3_val = -1*((r1[-1]**3)/(r2[0]**3*(r1_[0])**4))*simpson(Cso3(), z)
    Cso4_val = (r1[-1]/(r2[0]*r1_[0]))**4*simpson(Cso4(), z)
    
    print(Cso0_val, Cso1_val, Cso2_val, Cso3_val, Cso4_val)
    ul_U0.append(U_0)
    
    C0.append(Cso0_val)
    C1.append(Cso1_val)
    C2.append(Cso2_val)
    C3.append(Cso3_val)
    C4.append(Cso4_val)

X = np.array((ul_U0, C0, C1, C2, C3, C4)).T
np.savetxt('AsymptoticCoefficients.txt', X)

#data = np.loadtxt('Measured_Cs_aberint_and_DA_ULU0.txt')
data = np.loadtxt('Measured_Cs_aberint_and_DA_ULU0_0.8-1.0.txt')

B_da = data[:, 1]
B_aber = data[:, 2]
z_object = data[:, 6]
focal_length = data[:, 5]
magnifications_da = data[:, 3]

five_val = X[0, 1:]
six_val = X[1, 1:]
seven_val = X[2, 1:]
eight_val = X[3, 1:]
nine_val = X[4, 1:]
ten_val = X[5, 1:]

# mag_data = np.loadtxt('RealMag_Realf_AsympMag_Asympf_zobject.txt')
mag_data = np.loadtxt('RealMag_Realf_AsympMag_Asympf_zobject_0.8-1.0.txt')

asymp_mag = mag_data[:, 2]
real_mag = mag_data[:, 0]

z_image_real = mag_data[:, 4]
z_image_asymp = mag_data[:, 5]
# plt.plot(real_mag[60:120], B_da[60:120], label = 'UL/U0 = 0.6', color = 'b')
# plt.plot(asymp_mag[60:120], spherical_aber_poly(asymp_mag[60:120], *six_val), linestyle = '--', alpha = 0.5, color = 'b')

# plt.plot(real_mag[120:180], B_da[120:180], label = 'UL/U0 = 0.7', color = 'g')
# plt.plot(asymp_mag[120:180], spherical_aber_poly(asymp_mag[120:180], *seven_val), linestyle = '--', alpha = 0.5, color = 'g')

plt.figure()
plt.title('$C_{so}$ Vs Magnification')
plt.xlabel('Magnification (Asymptotic & Real)')
plt.ylabel('$C_{so}$ (mm) (Asymptotic & Real)')

ax = plt.gca()
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

plt.plot(real_mag[0:60], B_aber[0:60]*1e3, label = '$U_L/U_0 = 0.8$ - Real $C_{so}$', color = 'r')
plt.plot(asymp_mag[0:60], spherical_aber_poly(asymp_mag[0:60], *eight_val)*1e3, linestyle = '--', alpha = 0.5, color = 'r', label = '$U_L/U_0 = 0.8$ - Asymptotic $C_{so}$')

plt.plot(real_mag[60:120], B_aber[60:120]*1e3, label = '$U_L/U_0 = 0.9$ - Real $C_{so}$', color = 'g')
plt.plot(asymp_mag[60:120], spherical_aber_poly(asymp_mag[60:120], *nine_val)*1e3, linestyle = '--', alpha = 0.5, color = 'g', label = '$U_L/U_0 = 0.9$ - Asymptotic $C_{so}$')

plt.plot(real_mag[120:180], B_aber[120:180]*1e3, label = '$U_L/U_0 = 1.0$ - Real $C_{so}$', color = 'b')
plt.plot(asymp_mag[120:180], spherical_aber_poly(asymp_mag[120:180], *ten_val)*1e3, linestyle = '--', alpha = 0.5, color = 'b', label = '$U_L/U_0 = 1.0$ - Asymptotic $C_{so}$')
plt.yscale('symlog')
plt.legend()
plt.savefig('asymp_aber_vs_real_aber.svg', dpi = 800)


