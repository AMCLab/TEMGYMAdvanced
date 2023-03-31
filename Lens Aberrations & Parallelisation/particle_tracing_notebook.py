# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
# %%
import psutil
import numpy as np
import sympy as sp
from sympy.functions.combinatorial.factorials import factorial as fac
import numba
from numba import jit, prange, float64
import time
import matplotlib.pyplot as plt

# %% [markdown]
# B-Field calculation functions using sympy

# %%
#Round Lens Axial Field Function
def GlaserBellField(a = 1, zpos = 0):
    z = sp.symbols('z')
    return 1/(1+(((z+zpos)/a)**2))

def RoundLensFieldCartesian(B, order):
    
    #define sympy symbols
    x, y, z, r, alpha = sp.symbols('x y z r alpha')
    k, m, i, j, a = sp.symbols('k, m, i, j, a', integer=True)
    
    #Generic Cartesian Axial Potential from szilagyi
    U_m = sp.Function('U')(z, m)
    
    rm_cos_malp  = sp.Sum((((-1)**(k))*fac(m))/(fac(2**(k))*fac(m-2*k))*(x**(m-2*k)*y**(2*k)), (k, 0, order))
    A_m = sp.Sum((((-1)**(k))*fac(m))/((4**(k))*fac(k)*fac(m+k))*(x**(2)+y**(2))**(k)*(U_m.diff((z, 2*k))), (k, i, j))
    u_cart = (A_m)*rm_cos_malp
        
    #Set up summation term limits
    z_terms_start = 0
    xy_terms_start = 1
    z_terms_end = order
    xy_terms_end = order + 1
    
    #Obtain axial potential in x, y and z components
    u_cart_x = u_cart.subs([[i, xy_terms_start],[j, xy_terms_end]]).doit()
    u_cart_y = u_cart.subs([[i, xy_terms_start],[j, xy_terms_end]]).doit()
    u_cart_z = u_cart.subs([[i, z_terms_start],[j, z_terms_end]]).doit()

    #Obtain the axial magnetic field components
    B_x = u_cart_x.subs(m, 0).diff(x).subs(sp.diff(sp.Function('U')(z, 0), z), B)
    B_y = u_cart_y.subs(m, 0).diff(y).subs(sp.diff(sp.Function('U')(z, 0), z), B)
    B_z = u_cart_z.subs(m, 0).diff(z).subs(sp.diff(sp.Function('U')(z, 0), z), B)
    
    #Lambdify and jit the output Bfield function in cartesian coordinates
    B_lambda = numba.jit(sp.lambdify((x, y, z), (B_x.doit(), B_y.doit(), B_z.doit()), 'math'))
    
    return B_lambda

@jit(nopython = True, inline = 'always', fastmath = True)
def BFieldExpansionFourTerm(x, y, z):
    Bx = x*z*(x**2 + y**2)**3*(-0.64512*z**6/(z**2 + 1.0e-6)**3 + 0.96768*z**4/(z**2 + 1.0e-6)**2 - 0.4032*z**2/(z**2 + 1.0e-6) + 0.04032)/(18432*(z**2 + 1.0e-6)**5) - x*z*(x**2 + y**2)**2*(-0.00384*z**4 /
                                                                                                                                                                                              (z**2 + 1.0e-6)**2 + 0.00384*z**2/(z**2 + 1.0e-6) - 0.00072)/(384*(z**2 + 1.0e-6)**4) + x*z*(x**2 + y**2)*(-4.8e-5*z**2/(z**2 + 1.0e-6) + 2.4e-5)/(16*(z**2 + 1.0e-6)**3) + 1.0e-6*x*z/(z**2 + 1.0e-6)**2
    By = y*z*(x**2 + y**2)**3*(-0.64512*z**6/(z**2 + 1.0e-6)**3 + 0.96768*z**4/(z**2 + 1.0e-6)**2 - 0.4032*z**2/(z**2 + 1.0e-6) + 0.04032)/(18432*(z**2 + 1.0e-6)**5) - y*z*(x**2 + y**2)**2*(-0.00384*z**4 /
                                                                                                                                                                                              (z**2 + 1.0e-6)**2 + 0.00384*z**2/(z**2 + 1.0e-6) - 0.00072)/(384*(z**2 + 1.0e-6)**4) + y*z*(x**2 + y**2)*(-4.8e-5*z**2/(z**2 + 1.0e-6) + 2.4e-5)/(16*(z**2 + 1.0e-6)**3) + 1.0e-6*y*z/(z**2 + 1.0e-6)**2
    Bz = (-x**2/4 - y**2/4)*(8.0e-6*z**2/(z**2 + 1.0e-6) - 2.0e-6)/(z**2 + 1.0e-6)**2 - (x**2 + y**2)**3*(0.04608*z**6/(z**2 + 1.0e-6)**3 - 0.0576*z**4/(z**2 + 1.0e-6)**2 + 0.01728*z**2/(z**2 +
                                                                                                                                                                                           1.0e-6) - 0.00072)/(2304*(z**2 + 1.0e-6)**4) + (x**2 + y**2)**2*(0.000384*z**4/(z**2 + 1.0e-6)**2 - 0.000288*z**2/(z**2 + 1.0e-6) + 2.4e-5)/(64*(z**2 + 1.0e-6)**3) + 1/(1000000.0*z**2 + 1)
    return Bx, By, Bz

@jit(nopython = True, inline = 'always', fastmath = True)
def BFieldExpansionOneTerm(x, y, z):
    Bx = 1.0e-6*x*z/(z**2 + 1.0e-6)**2
    
    By = 1.0e-6*y*z/(z**2 + 1.0e-6)**2
    
    Bz = 1/(1000000.0*z**2 + 1)
    
    return Bx, By, Bz
    
#@jit(nopython=True, fastmath=True)
def RoundLensFieldInterpolate(xmin, xmax, ymin, ymax, zmin, zmax, n):
    
    xx = np.linspace(xmin, xmax, n)
    yy = np.linspace(ymin, ymax, n)
    zz = np.linspace(zmin, zmax, n)
    X, Y, Z = np.meshgrid(xx, yy, zz, indexing="ij")
    
    xyz_grid = np.stack((xx, yy, zz), axis=0)
    
    x = X.ravel()
    y = Y.ravel()
    z = Z.ravel()

    b_points = np.array([x, y, z]).T

    B = np.zeros(b_points.shape)

    B += np.array(BFieldExpansionOneTerm(b_points[:, 0], b_points[:, 1], b_points[:, 2])).T

    B_matrix = B.reshape((n, n, n, 3))
    
    return B_matrix, xyz_grid

@jit(float64[:](float64,float64,float64,float64[:],float64[:],float64[:],float64[:, :, :, :]), nopython=True)
def interpolate(x, y, z, xx, yy, zz, B):
    
    #initialise output array
    c = np.array([0, 0, 0], dtype = np.float64)
    
    #find the index of the point before the value
    xi0 = np.searchsorted(xx, x)-1
    yi0 = np.searchsorted(yy, y)-1
    zi0 = np.searchsorted(zz, z)-1
    xi1 = xi0 + 1
    yi1 = yi0 + 1
    zi1 = zi0 + 1
    
    #Find the index value in the grid array
    x0 = xx[xi0]
    y0 = yy[yi0]
    z0 = zz[zi0]
    x1 = xx[xi1]
    y1 = yy[yi1]
    z1 = zz[zi1]
    
    #Calculate how far in the grid our value is (normalised)
    xd = (x-x0)/(x1-x0)
    yd = (y-y0)/(y1-y0)
    zd = (z-z0)/(z1-z0)
    
    #Loop through each coordinate and find the linear interpolated value. 
    for B_i in range(B.shape[-1]):
        B000 = B[xi0, yi0, zi0, B_i]
        B100 = B[xi1, yi0, zi0, B_i]
        B010 = B[xi0, yi1, zi0, B_i]
        B110 = B[xi1, yi1, zi0, B_i]
        B001 = B[xi0, yi0, zi1, B_i]
        B101 = B[xi1, yi0, zi1, B_i]
        B011 = B[xi0, yi1, zi1, B_i]
        B111 = B[xi1, yi1, zi1, B_i]
        
        c00 = B000*(1-xd) + B100*xd
        c01 = B001*(1-xd) + B101*xd
        c10 = B010*(1-xd) + B110*xd
        c11 = B011*(1-xd) + B111*xd
    
        c0 = c00*(1-yd) + c10*yd
        c1 = c01*(1-yd) + c11*yd
    
        c[B_i] = c0*(1-zd) + c1*zd
    
    return c

# %% [markdown]
# Two Particle tracing functions to integrate the electron equations of motion from Szilagyi's textbook.
# euler_dz_simultaneous_rays is designed to increment multiple rays at once, while euler_dz_single_ray can only trace a single ray. 

# %%
@jit(nopython = True, fastmath=True, parallel = True)
def euler_dz_simultaneous_rays(r, v, z, v_acc, q, m, gamma, dz, n, B):
    
    num_rays = r.shape[1]
    
    n_threads = 16 #number of threads on 8 core cpu
    thread_block_size = int(num_rays/n_threads) #number of rays per thread
    macro_block_size = 128
    macro_blocks = int(thread_block_size/macro_block_size)
    
    # #Initialise paralellisation of code execution
    for thread in prange(n_threads):
        
        #get starting index for the block of rays going to each thread.
        thread_idx = thread*thread_block_size
        
        for macro_block in range(macro_blocks):
            #get starting index for the block of rays of each macroblock in a thread.
            macro_block_idx = macro_block*macro_block_size
            
            #combine the starting index of the thread and macroblock
            idx_start = thread_idx + macro_block_idx
            
            #Reset the step counter 
            z0 = z
            
            #Step each ray through it's number of steps
            for i in range(1, n):
                
                #step through each macroblock
                for j in range(macro_block_size):
                    
                    #get b field and velocity term (See Szilagyi Chapter 2)
                    B_ = B(r[i-1, idx_start+j, 0], r[i-1, idx_start+j, 1], z0)
                    v_ = np.sqrt(np.sum((v[i-1, idx_start+j, :]**2))+1)
                
                    #Increment velocity.
                    v[i, idx_start+j, 0] = v[i-1, idx_start+j, 0] + (v_*q)/(m*v_acc*gamma)*(-(1+v[i-1, idx_start+j, 0]**2)*B_[1]+v[i-1, idx_start+j, 1]*(v[i-1, idx_start+j, 0]*B_[0]+B_[2]))*dz
                    v[i, idx_start+j, 1] = v[i-1, idx_start+j, 1] + (v_*q)/(m*v_acc*gamma)*(+(1+v[i-1, idx_start+j, 1]**2)*B_[0]-v[i-1, idx_start+j, 0]*(v[i-1, idx_start+j, 1]*B_[1]+B_[2]))*dz    
                    
                    #Obtain new position       
                    r[i, idx_start+j, :]  = r[i-1, idx_start+j, :]  + v[i, idx_start+j, :]*dz
                    
                #Find next z for Bfield
                z0 = z0 + dz
        
    return r, v

@jit(nopython=True)
def euler_dz_single_ray(x0, v0, v_mag, q, m, gamma, dz, n, B):
    
    #Create 2D array of zeros to store single ray positions and velocities
    X = np.zeros((len(x0), n), np.float64)
    X[:, 0] = x0.copy()
    V = np.zeros((len(v0), n), np.float64)
    V[:, 0] = v0.copy()
    V[2, :] = 1
    
    for i in range(1, n):
        
        #Obtain B
        B_= B(X[0, i-1], X[1, i-1], X[2, i-1])
        
        #Increment velocity and position
        v_ = np.sqrt(np.sum((V[:, i-1]**2)))
        
        V[0, i] = V[0, i-1] + q/(gamma*m*v_mag)*v_*(-(1+V[0, i-1]**2)*B_[1]+V[1, i-1]*(V[0, i-1]*B_[0]+B_[2]))*dz
        V[1, i] = V[1, i-1] + q/(gamma*m*v_mag)*v_*(+(1+V[1, i-1]**2)*B_[0]-V[0, i-1]*(V[1, i-1]*B_[1]+B_[2]))*dz      
        X[:, i] = X[:, i-1] + V[:, i]*dz
        X[2, i] = X[2, i-1] + dz
        
        
    return X, V

@jit(nopython=True)
def euler_dz_single_ray_interp(x0, v0, v_mag, q, m, gamma, dz, n, grid, B_grid):
    
    #Create 2D array of zeros to store single ray positions and velocities
    X = np.zeros((len(x0), n), np.float64)
    X[:, 0] = x0
    V = np.zeros((len(v0), n), np.float64)
    V[:, 0] = v0
    
    x = x0
    v = v0
    
    for i in range(1, n):
        
        #Obtain B
        B_= interpolate(x[0], x[1], x[2], grid[0, :], grid[1, :], grid[2, :], B_grid)
        
        #Increment velocity and position
        v_ = np.sqrt(np.sum((v**2))+1)
        
        v[0] = v[0] + q/(gamma*m*v_mag)*v_*(-(1+v[0]**2)*B_[1]+v[1]*(v[0]*B_[0]+B_[2]))*dz
        v[1] = v[1] + q/(gamma*m*v_mag)*v_*(+(1+v[1]**2)*B_[0]-v[0]*(v[1]*B_[1]+B_[2]))*dz                       
        x = x + v*dz
        
        X[:, i] = x
        V[:, i] = v
        
    return X, V

# %% [markdown]
# Run the script to trace a single ray. Time it and plot it. 

# %%
#Define constants
q  = -1.60217662e-19
m = 9.10938356e-31
c = 2.99792458e8

#Create the Round Lens field and obtain a lambda function to obtain the Bfield at any position. 
B = GlaserBellField(a = 1e-3, zpos = 0)
B_lambda = RoundLensFieldCartesian(B, order = 0)

#Choose initial angle of particle in polar direction (radians)
polar_angle = 0
alpha_angle = 0

#Get relativistic acceleration velocity for a 300kV electron
v_acc = 1*c*(1-(1-(q*3e5)/(m*(c**2)))**(-2))**(1/2)
gamma = 1/(1-(v_acc**2/c**2))**(1/2)

#Distance to run simulation in metres
l_max = 10e-2
z_init = -5e-2

#Step size of particle integrator
dz = 1e-5

#Integer steps to run the integration scheme for and number of rays
steps = int(l_max/dz)

x0 = np.array([1e-4, 0, -5e-2], dtype = np.float64)
v0 = np.array([np.sin(polar_angle) * np.cos(alpha_angle), np.sin(polar_angle) * np.sin(alpha_angle), np.cos(polar_angle)], dtype = np.float64)

B, grid = RoundLensFieldInterpolate(-1e-3, 1e-3, -1e-3, 1e-3, -5e-2, 5e-2, 200)
b_test = interpolate(0, 0, 0, grid[0, :], grid[1, :], grid[2, :], B)

fig, ax = plt.subplots()

pts, _ = euler_dz_single_ray(x0.copy(), v0.copy(), v_acc, q, m, gamma, dz, 2, BFieldExpansionFourTerm)
start = time.time()
pts, _ = euler_dz_single_ray(x0.copy(), v0.copy(), v_acc, q, m, gamma, dz, steps, BFieldExpansionFourTerm)
end = time.time()

factor = q/(gamma*m*v_acc)
ax.plot(pts[2, :], np.sqrt(pts[0, :]**2+pts[1, :]**2), 'b', linewidth = 1, label = 'Euler-Cromer Single Ray')
print('Single Ray Time = ', end - start)


# pts, _ = euler_dz_single_ray_interp(x0.copy(), v0.copy(), v_acc, q, m, gamma, dz, 2, grid, B)
# start = time.time()
# pts, _ = euler_dz_single_ray_interp(x0.copy(), v0.copy(), v_acc, q, m, gamma, dz, steps, grid, B)
# end = time.time()
# ax.plot(pts[2, :], np.sqrt(pts[0, :]**2+pts[1, :]**2)*np.sign(pts[0, :]), 'r', linewidth = 2, alpha = 0.5, label = 'Euler-Cromer Interp Single Ray')
# #print('Single Ray Time Interpolation = ', end - start)

# ax.set_title('Euler-Cromer - Single Ray ')
# ax.set_xlabel('Z (m)')
# ax.set_ylabel('R (m)')
# plt.legend()

# fig, ax = plt.subplots()
# ax.plot(np.linspace(z_init, z_init+l_max, 10000), B_lambda(0, 0, np.linspace(z_init, z_init+l_max, 10000))[2], 'g', linewidth = 3, alpha = 0.5, label = 'Analytical')
# # ax.plot(grid[2, :], B[250, 250, :, 2], 'b', label = 'Interpolated')
# ax.set_title('BField Interpolation vs Analytical')
# ax.set_xlabel('B (T)')
# ax.set_ylabel('R (m)')
# plt.legend()
'''
# %% [markdown]
# Run code to trace many rays simultaneously

# %%
num_rays = 2**13

#Make a 2 x steps x number of rays size array to store the particle locations at each step
r = np.zeros((steps, num_rays, 2), dtype = np.float64)

#Create an array of initial particle locations from 0 to 100 micrometres away from the centre in the x direction
x_init = np.linspace(0, 8e-4, num_rays, dtype = np.float64)
y_init = np.linspace(0, 0, num_rays, dtype = np.float64)

#Update the initial conditions for the particle location
r[0, :, 0] = x_init
r[0, :, 1] = y_init

#We need to make a 2 x steps x number of rays size array to store the particle velocities at each step
v = np.zeros((steps, num_rays, 2), dtype = np.float64)

polar_init = np.linspace(0, 0, num_rays, dtype = np.float64)
alpha_init = np.linspace(0, 0, num_rays, dtype = np.float64)
vx_init = np.sin(polar_angle) * np.cos(alpha_angle)
vy_init = np.sin(polar_angle) * np.sin(alpha_angle)
v[0, :, 0] = vx_init
v[0, :, 1] = vy_init

#Create array of initial z positions

#initialise function in numba
r_test = np.zeros((steps, num_rays, 2), dtype = np.float64)
v_test = np.zeros((steps, num_rays, 2), dtype = np.float64)
z_test = np.array(np.ones((num_rays))*z_init) 

pts, _ = euler_dz_simultaneous_rays(r_test, v_test, z_init, v_acc, q, m, gamma, dz, 1, BFieldExpansionFourTerm)

z = np.array(np.ones((num_rays))*z_init) 
start = time.time()
pts, _ = euler_dz_simultaneous_rays(r, v, z_init, v_acc, q, m, gamma, dz, steps, BFieldExpansionFourTerm)
end = time.time()

print(num_rays,' Simultaneous Rays = ', end - start)

fig, ax = plt.subplots()
z_coords = np.ones((num_rays, steps))*np.linspace(z_init, z_init+steps*dz-dz, steps)

ax.plot(z_coords.T, np.sqrt(pts[:, :, 0]**2+pts[:, :, 1]**2), 'b', linewidth = 1, label = 'Euler-Cromer Single Ray')
ax.set_title('Euler-Cromer - Multiple Rays ')
ax.set_xlabel('Z (m)')
ax.set_ylabel('R (m)')

# %%
import psutil
psutil.cpu_count()


# %%
# get_ipython().system('cat /proc/cpuinfo')
# get_ipython().system('cat /gpu/gpuinfo')


# %%

'''

