# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 10:41:35 2023

@author: User
"""

import sympy as sp
from sympy import simplify
from sympy.core.numbers import pi
from IPython.display import display
from latex2sympy2 import latex2sympy, latex2latex
import numpy as np
import matplotlib.pyplot as plt

tol = 1e-15

# sp.init_printing()
x, x1, y, y1, z, K, eps_0, y_0 = sp.symbols('x x_1 y y_1 z K epsilon_0 y_0')
a, b = sp.symbols('a b')

phi = (K/(4*pi*eps_0))*(sp.log((sp.sqrt(x**2+(y+a)**2 + z**2)+y+a)/(sp.sqrt(x**2+y**2+z**2)+y)))

phi_latex = r"K\frac{1}{4 \pi \epsilon_0}\left[\ln \frac{\sqrt{x^2+(y+a)^2+z^2}+(y+a)}{\sqrt{x^2+y^2+z^2}+y}\right]"
phi_janghyun = latex2sympy(phi_latex)

# print(phi.equals(phi_janghyun)) 

phi_simple = phi.subs({K:1, eps_0:1, pi:np.pi, a:1})
phi_janghyun_simple = phi_janghyun.subs({K:1, eps_0:1, pi:np.pi, a:1})
 
phi_lambda_simple = sp.lambdify([x, y, z], phi_simple, 'numpy')
phi_janghyun_lambda_simple = sp.lambdify([x, y, z], phi_janghyun_simple, 'numpy')

x_eval, y_eval, z_eval = 2, 2, 2
print(np.abs(np.array(phi_lambda_simple(x_eval, y_eval, z_eval))-np.array(phi_janghyun_lambda_simple(x_eval, y_eval, z_eval))) < tol)

Ex = -1*phi.diff(x)
Ey = -1*phi.diff(y)
Ez = -1*phi.diff(z)

Ex_latex = r"$\frac{-K}{4 \pi \epsilon_0} x\left[\frac{1}{\sqrt{x^2+(y+a)^2+z^2}+(y+a)} \cdot \frac{1}{\sqrt{x^2+(y+a)^2+z^2}}-\frac{1}{\sqrt{x^2+y^2+z^2}+y} \cdot \frac{1}{\sqrt{x^2+y^2+z^2}}\right]$"
Ey_latex = r"$\frac{-K}{4 \pi \epsilon_0}\left[\frac{1}{\sqrt{x^2+(y+a)^2+z^2}+(y+a)} \cdot\left(\frac{y+a}{\sqrt{x^2+(y+a)^2+z^2}}+1\right)-\frac{1}{\sqrt{x^2+y^2+z^2}+y} \cdot\left(\frac{y}{\sqrt{x^2+y^2+z^2}}+1\right)\right]$"
Ez_latex = r"$\frac{-K}{4 \pi \epsilon_0} z\left[\frac{1}{\sqrt{x^2+(y+a)^2+z^2}+(y+a)} \cdot \frac{1}{\sqrt{x^2+(y+a)^2+z^2}}-\frac{1}{\sqrt{x^2+y^2+z^2}+y} \cdot \frac{1}{\sqrt{x^2+y^2+z^2}}\right]$"

Ex_janghyun = latex2sympy(Ex_latex)
Ey_janghyun = latex2sympy(Ey_latex)
Ez_janghyun = latex2sympy(Ez_latex)

display(simplify(Ex))
display(simplify(Ex_janghyun))

# print(Ex.equals(Ex_janghyun))
# print(Ey.equals(Ey_janghyun))
# print(Ez.equals(Ez_janghyun))

Ex_simple = Ex.subs({K:1, eps_0:1, pi:np.pi, a:1})
Ey_simple = Ey.subs({K:1, eps_0:1, pi:np.pi, a:1})
Ez_simple = Ez.subs({K:1, eps_0:1, pi:np.pi, a:1})

E_lambda_simple = sp.lambdify([x, y, z], [Ex_simple, Ey_simple, Ez_simple], 'numpy')

Ex_janghyun_simple = Ex_janghyun.subs({K:1, eps_0:1, pi:np.pi, a:-1})
Ey_janghyun_simple = Ey_janghyun.subs({K:1, eps_0:1, pi:np.pi, a:-1})
Ez_janghyun_simple = Ez_janghyun.subs({K:1, eps_0:1, pi:np.pi, a:-1})

E_janghyun_lambda_simple = sp.lambdify([x, y, z], [Ex_janghyun_simple, Ey_janghyun_simple, Ez_janghyun_simple], 'numpy')

print(np.abs(np.array(E_lambda_simple(x_eval, y_eval, z_eval))-np.array(E_janghyun_lambda_simple(x_eval, y_eval, z_eval))) < tol)

x_line = np.linspace(-1, 1, 1000)
y_line = np.linspace(-1, 1, 1000)
x_grid, y_grid = np.meshgrid(x_line, y_line)

plt.figure()
plt.imshow(phi_lambda_simple(x_grid, y_grid, 0)-phi_lambda_simple(-(x_grid-0), -(y_grid-0.1), 0), extent = [-1, 1, -1, 1])

def make_double_wire_potential_and_efield(phi, phi_0, x, y, z, K, eps_0, a,
                                          x_displacement, y_displacement, K_val,
                                          eps_0_val, length):
    
    phi = phi.subs({K: K_val, eps_0: eps_0_val, pi: np.pi, a: length})
    
    phi_top_wire = phi#.subs({y:length})
    phi_bottom_wire = phi.subs({x:-(x-x_displacement), y:-(y-y_displacement)})
    phi_wires = phi_top_wire - phi_bottom_wire
    phi_wires_electron = phi_0 - phi_wires
    
    phi_hat = (phi_wires_electron)*(1+eps*(phi_wires_electron))
    
    dphi_hat_dx = phi_hat.diff(x)
    dphi_hat_dy = phi_hat.diff(y)
    dphi_hat_dz = phi_hat.diff(z)
    
    phi_wires_lambda = sp.lambdify([x, y, z], phi_wires, 'numpy')
    phi_hat_lambda = sp.lambdify([x, y, z], phi_hat, 'numpy')
    dphi_hat_lambda = sp.lambdify([x, y, z], [dphi_hat_dx, dphi_hat_dy, dphi_hat_dz], 'numpy')
    
    return phi_hat_lambda, dphi_hat_lambda, phi_wires_lambda
    
e = -1.60217662e-19 #unit C
m = 9.10938356e-31 #unit kg
c = 2.99792458e8 #unit m/s
eps_0_val = 8.85418782e-12 #permittivity unit F/m

phi_0 = 2e5 #unit V
v_0 = c*(1-(1-(e*phi_0)/(m*(c**2)))**(-2))**(1/2) #unit m/s
eta = (abs(e)/(2*m))**(1/2) 
gamma = 1/(1-(v_0**2/c**2))**(1/2)
eps = abs(e)/(2*m*c**2)

phi_hat = (phi_0)*(1+eps*(phi_0))
v_0_hawkes = 2*eta*(phi_hat/(1+4*eps*phi_hat))**(1/2)

phi_lambda, dphi_lambda, phi_wires_lambda = make_double_wire_potential_and_efield(phi, phi_0, x, y, z, K, eps_0, a, 0, 0.1, 1, 1, 1)

plt.figure()
plt.imshow(phi_wires_lambda(x_grid, y_grid, 0), extent = [-1, 1, -1, 1])

# plt.figure()
# plt.imshow(phi_lambda(x_grid, y_grid, 0)-(phi_lambda(x_grid, y_grid, 0)+phi_lambda(x_grid, -(y_grid-0.1), 0)))


K_val = 3.31 #unit e/nm - convert to coulombs per m? 
K_val_SI = (3.31*abs(e))/1e-9 #C/m
y_displacement = 177e-9 #unit m
x_displacement = 10e-9 #unit m
a_val = 1e-3 #unit m

phi_hat_lambda, dphi_hat_lambda, phi_wires_lambda = make_double_wire_potential_and_efield(phi, phi_0, x, y, z, K, eps_0, a, x_displacement, y_displacement, 
                                                                        K_val_SI, eps_0_val, a_val)

plan_figure, plan_ax = plt.subplots()
x_line, xstep = np.linspace(-400e-9, 400e-9, 300, retstep = True)
y_line, ystep = np.linspace(-400e-9, 400e-9, 300, retstep = True)
x_grid, y_grid = np.meshgrid(x_line, y_line)
z_grid = 0*np.ones(x_grid.shape)

potential_image = phi_wires_lambda(x_grid, y_grid, 0)
plan_ax.imshow(potential_image, extent=[-400e-9, 400e-9, -400e-9, 400e-9])
plan_ax.contour(potential_image, 25, linewidths = 0.25, colors = 'k', extent = [-400e-9, 400e-9, -400e-9, 400e-9], origin = 'upper')

zy_figure, zy_ax = plt.subplots()
z_line, zstep = np.linspace(-400e-9, 400e-9, 300, retstep = True)
y_line, ystep = np.linspace(-400e-9, 400e-9, 300, retstep = True)
y_grid, z_grid = np.meshgrid(y_line, z_line)
x_grid = 0*np.ones(y_grid.shape)

potential_image = phi_wires_lambda(x_grid, z_grid, y_grid)
zy_ax.imshow(potential_image, extent=[-400e-9, 400e-9, -400e-9, 400e-9], origin = 'upper')
# plan_ax.contour(potential_image, 25, linewidths = 0.25, colors = 'k', extent=[-400e-9, 400e-9, -1000e-9, 1000e-9], origin = 'upper')


# plt.figure()
# plt.plot(phi_hat_lambda(0, 0, np.linspace(-1e-6, 1e6, 1000)))

#%%
#dormand prince adaptive step size solver
def odedopri(f,  x0,  y0,  x1,  tol,  hmax,  hmin,  maxiter, args=()):
    a21 = (1.0/5.0)
    a31 = (3.0/40.0)
    a32 = (9.0/40.0)
    a41 = (44.0/45.0)
    a42 = (-56.0/15.0)
    a43 = (32.0/9.0)
    a51 = (19372.0/6561.0)
    a52 = (-25360.0/2187.0)
    a53 = (64448.0/6561.0)
    a54 = (-212.0/729.0)
    a61 = (9017.0/3168.0)
    a62 = (-355.0/33.0)
    a63 = (46732.0/5247.0)
    a64 = (49.0/176.0)
    a65 = (-5103.0/18656.0)
    a71 = (35.0/384.0)
    a72 = (0.0)
    a73 = (500.0/1113.0)
    a74 = (125.0/192.0)
    a75 = (-2187.0/6784.0)
    a76 = (11.0/84.0)

    c2 = (1.0 / 5.0)
    c3 = (3.0 / 10.0)
    c4 = (4.0 / 5.0)
    c5 = (8.0 / 9.0)
    c6 = (1.0)
    c7 = (1.0)

    b1 = (35.0/384.0)
    b2 = (0.0)
    b3 = (500.0/1113.0)
    b4 = (125.0/192.0)
    b5 = (-2187.0/6784.0)
    b6 = (11.0/84.0)
    b7 = (0.0)

    b1p = (5179.0/57600.0)
    b2p = (0.0)
    b3p = (7571.0/16695.0)
    b4p = (393.0/640.0)
    b5p = (-92097.0/339200.0)
    b6p = (187.0/2100.0)
    b7p = (1.0/40.0)

    x = x0
    y = y0
    h = hmax
    X = np.zeros(maxiter)
    X[0] = x0
    Y = np.zeros((maxiter, len(y0)))
    Y[0] = y0
    
    #We need a special step counter because the ODE dormand prince 
    #method will do a step again with a smaller step size if the error is too large
    step = 1
    for i in range(maxiter):
       # /* Compute the function values */
       K1 = f(x,       y, *args)
       K2 = f(x + c2*h, y+h*(a21*K1), *args)
       K3 = f(x + c3*h, y+h*(a31*K1+a32*K2), *args)
       K4 = f(x + c4*h, y+h*(a41*K1+a42*K2+a43*K3), *args)
       K5 = f(x + c5*h, y+h*(a51*K1+a52*K2+a53*K3+a54*K4), *args)
       K6 = f(x + h, y+h*(a61*K1+a62*K2+a63*K3+a64*K4+a65*K5), *args)
       K7 = f(x + h, y+h*(a71*K1+a72*K2+a73*K3+a74*K4+a75*K5+a76*K6), *args)

       error = abs((b1-b1p)*K1+(b3-b3p)*K3+(b4-b4p)*K4+(b5-b5p)*K5 +
                   (b6-b6p)*K6+(b7-b7p)*K7)

       #Error in X controls tolerance
       error = max(error)

       # error control
       if error != 0.0:
           delta = 0.84 * pow(tol / error, (1.0/5.0))
       else:
           delta = np.inf

       if (error < tol):
          x = x + h
          X[step] = x
          y = y + h * (b1*K1+b3*K3+b4*K4+b5*K5+b6*K6)
          Y[step, :] = y
          step+=1

       if (delta <= 0.1):
          h = h * 0.1
       elif (delta >= 4.0):
          h = h * 4.0
       else:
          h = delta * h

       if (h > hmax):
          h = hmax

       if (x >= x1):
          print('reached end')
          break

       elif (x + h > x1):
          h = x1 - x

       elif (h < hmin):
          print('Below hmin')
          break

    return X, Y, step

#%%
# def euler_dz(z0,  x,  z1, dz, steps, phi_lambda, dphi_hat_lambda):
    
#     z = z0
#     for i in range(1, steps):
#         p = np.sqrt(1+x[i-1, 1]**2+x[i-1, 3]**2)
#         phi_hat = phi_lambda(x[i-1, 0], x[i-1, 2], z[i-1])
#         dphi_hat_x, dphi_hat_y, dphi_hat_z = dphi_hat_lambda(x[i-1, 0], x[i-1, 2], z[i-1])
        
#         x[i, 1] = x[i-1, 1] + (p**2)/(2*phi_hat)*(dphi_hat_x-x[i-1, 1]*dphi_hat_z)*dz
#         x[i, 3] = x[i-1, 3] + (p**2)/(2*phi_hat)*(dphi_hat_y-x[i-1, 3]*dphi_hat_z)*dz
        
#         x[i, 0] = x[i-1, 0] + x[i, 1]*dz
#         x[i, 2] = x[i-1, 2] + x[i, 3]*dz
        
#         z[i] = z[i-1] + dz
        
#     return  x, z
        
    
#%%
def trajectory_equation_of_motion(z, x, phi_hat_lambda, dphi_hat_lambda):
    
    p = np.sqrt(1+x[1]**2+x[3]**2)
    phi_hat = phi_hat_lambda(x[0], x[2], z)
    print(phi_hat)
    dphi_hat_x, dphi_hat_y, dphi_hat_z = dphi_hat_lambda(x[0], x[2], z)
    
    return np.array([x[1], (p**2)/(2*phi_hat)*(dphi_hat_x-x[1]*dphi_hat_z), x[3], (p**2)/(2*phi_hat)*(dphi_hat_y-x[3]*dphi_hat_z)])


#%%
#define initial x y position and slope
x0 = np.array([5e-9, 0, -y_displacement/2, 0]) #x, x', y, y'
z0 = -1e-6 
zF = 1e-6

z_out, x_out, steps = odedopri(trajectory_equation_of_motion,  z0,  x0,  zF,  1e-3,  1e-4,  1e-15,  10000, args=(phi_hat_lambda, dphi_hat_lambda))

plt.figure()
plt.plot(z_out[:steps], x_out[:steps, 0])
plt.plot(z_out[:steps], x_out[:steps, 2])

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.invert_zaxis()

ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# make the grid lines transparent
ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.plot(x_out[:steps, 0], x_out[:steps, 2], z_out[:steps], color =  'g')

plan_ax.plot(x_out[0, 0], x_out[0, 2], 'xb', label = 'Electron Start')
plan_ax.plot(x_out[steps, 0], x_out[steps, 2], 'xr', label = 'Electron End')

zy_ax.plot(z_out[:steps], x_out[:steps, 2], '-r', label = 'Electron Path')
zy_ax.set_xlim(-1000e-9, 1000e-9)
zy_ax.set_ylim(-200e-9, 200e-9)


