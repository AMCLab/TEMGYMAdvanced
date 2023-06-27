#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:42:06 2022

@author: andy
"""
import numpy as np
from numba import jit, prange
import math

@jit(nopython = True, fastmath=True, parallel = True)
def euler_dz_simultaneous_rays_astig(r, v, z, alpha, dz, z_prop, B, Bmag, steps, step):
    
    num_rays = r.shape[1]
    
    n_threads = 16 #number of threads on 8 core cpu
    thread_block_size = int(num_rays/n_threads) #number of rays per thread
    macro_block_size = 1
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
            for i in range(step, steps + step):
                
                #step through each macroblock
                for j in range(macro_block_size):
                    
                    r[i, idx_start+j, 2] = z0
                    theta = np.arctan2(r[i-1, idx_start+j, 1], r[i-1, idx_start+j, 0])
                    
                    #get b field and velocity term (See Szilagyi Chapter 2)
                    Bx, By, Bz = B(r[i-1, idx_start+j, 0], r[i-1, idx_start+j, 1], z0, theta)
                    Bx, By, Bz = Bmag*Bx, Bmag*By, Bmag*Bz
                    v_ = math.sqrt((v[i-1, idx_start + j, 0]**2+v[i-1, idx_start + j, 1]**2+1))
                
                    #Increment velocity.
                    v[i, idx_start+j, 0] = v[i-1, idx_start+j, 0] + v_*alpha*(-(1+v[i-1, idx_start+j, 0]**2)*By+v[i-1, idx_start+j, 1]*(v[i-1, idx_start+j, 0]*Bx+Bz))*dz
                    v[i, idx_start+j, 1] = v[i-1, idx_start+j, 1] + v_*alpha*(+(1+v[i-1, idx_start+j, 1]**2)*Bx-v[i-1, idx_start+j, 0]*(v[i-1, idx_start+j, 1]*By+Bz))*dz    
                    
                    r[i, idx_start+j, 0]  = r[i-1, idx_start+j, 0]  + v[i, idx_start+j, 0]*dz
                    r[i, idx_start+j, 1]  = r[i-1, idx_start+j, 1]  + v[i, idx_start+j, 1]*dz   
                    
                #Find next z for Bfield
                z0 = z0 + dz
            
    return r, v

@jit(nopython = True, fastmath=True, parallel = True)
def euler_dz_simultaneous_rays(r, v, z, alpha, dz, z_prop, B, Bmag, steps, step):
    
    num_rays = r.shape[1]
    
    n_threads = 16 #number of threads on 8 core cpu
    thread_block_size = int(num_rays/n_threads) #number of rays per thread
    macro_block_size = 1
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
            for i in range(step, steps + step):
                
                #step through each macroblock
                for j in range(macro_block_size):
                    
                    r[i, idx_start+j, 2] = z0
                    
                    #get b field and velocity term (See Szilagyi Chapter 2)
                    Bx, By, Bz = B(r[i-1, idx_start+j, 0], r[i-1, idx_start+j, 1], z0)
                    Bx, By, Bz = Bmag*Bx, Bmag*By, Bmag*Bz
                    v_ = math.sqrt((v[i-1, idx_start + j, 0]**2+v[i-1, idx_start + j, 1]**2+1))
                
                    #Increment velocity.
                    v[i, idx_start+j, 0] = v[i-1, idx_start+j, 0] + v_*alpha*(-(1+v[i-1, idx_start+j, 0]**2)*By+v[i-1, idx_start+j, 1]*(v[i-1, idx_start+j, 0]*Bx+Bz))*dz
                    v[i, idx_start+j, 1] = v[i-1, idx_start+j, 1] + v_*alpha*(+(1+v[i-1, idx_start+j, 1]**2)*Bx-v[i-1, idx_start+j, 0]*(v[i-1, idx_start+j, 1]*By+Bz))*dz    
                    
                    r[i, idx_start+j, 0]  = r[i-1, idx_start+j, 0]  + v[i, idx_start+j, 0]*dz
                    r[i, idx_start+j, 1]  = r[i-1, idx_start+j, 1]  + v[i, idx_start+j, 1]*dz   
                    
                #Find next z for Bfield
                z0 = z0 + dz
            
    return r, v
@jit(nopython = True, fastmath=True, parallel = True)
def euler_dz_simultaneous_rays_interp(r, v, z, alpha, dz, z_prop, B_grid, grid, grid_step, Bmag, steps, step):
    num_rays = r.shape[1]

    n_threads = 16 #number of threads on 8 core cpu
    thread_block_size = int(num_rays/n_threads) #number of rays per thread
    macro_block_size = 1
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
            for i in range(step, steps + step + 1):
                
                #step through each macroblock
                for j in range(macro_block_size):
                    
                    r[i, idx_start+j, 2] = z0
                    
                    #get b field and velocity term (See Szilagyi Chapter 2)
                    Bx, By, Bz = b_interp(r[i-1, idx_start+j, 0], r[i-1, idx_start+j, 1], z0, grid[0, :], grid[1, :], grid[2, :], grid_step[0], grid_step[1], grid_step[2], B_grid)
                    Bx, By, Bz = Bmag*Bx, Bmag*By, Bmag*Bz
                    v_ = math.sqrt((v[i-1, idx_start + j, 0]**2+v[i-1, idx_start + j, 1]**2+1))
                
                    #Increment velocity.
                    v[i, idx_start+j, 0] = v[i-1, idx_start+j, 0] + v_*alpha*(-(1+v[i-1, idx_start+j, 0]**2)*By+v[i-1, idx_start+j, 1]*(v[i-1, idx_start+j, 0]*Bx+Bz))*dz
                    v[i, idx_start+j, 1] = v[i-1, idx_start+j, 1] + v_*alpha*(+(1+v[i-1, idx_start+j, 1]**2)*Bx-v[i-1, idx_start+j, 0]*(v[i-1, idx_start+j, 1]*By+Bz))*dz    
                    
                    r[i, idx_start+j, 0]  = r[i-1, idx_start+j, 0] + v[i, idx_start+j, 0]*dz
                    r[i, idx_start+j, 1]  = r[i-1, idx_start+j, 1] + v[i, idx_start+j, 1]*dz   
                    
                #Find next z for Bfield
                z0 = z0 + dz
            
            #propagate the rays to the next component (or the end)
            for j in range(macro_block_size):
                #propagate the rays to the next component
                #use the fact that 1/arccos(tan(x)) = x/(sqrt(1+x**2)) to write this equation in a better way. 
                v[step+steps+1, idx_start+j, 0] = v[step+steps, idx_start+j, 0]
                v[step+steps+1, idx_start+j, 1] = v[step+steps, idx_start+j, 1]
                r[step+steps+1, idx_start+j, 0] = r[step+steps, idx_start+j, 0] - z_prop*v[step+steps, idx_start+j, 0] #z_prop/(np.tan(np.arccos(v[step+steps, idx_start+j, 0])))
                r[step+steps+1, idx_start+j, 1] = r[step+steps, idx_start+j, 1] - z_prop*v[step+steps, idx_start+j, 1] #z_prop/(np.tan(np.arccos(v[step+steps, idx_start+j, 1])))
                r[step+steps+1, idx_start+j, 2] = r[step+steps, idx_start+j, 2] - z_prop
    
    return r, v

@jit(nopython=True, fastmath = True, inline = 'always')
def b_interp(x, y, z, xx, yy, zz, x_step, y_step, z_step, B):
    
    c = [0., 0., 0.]
    xi0 = int((x+xx[-1])//x_step)
    yi0 = int((y+yy[-1])//y_step)
    zi0 = int((zz[0]-z)//z_step)
    
    xi1 = xi0 + 1
    yi1 = yi0 + 1
    zi1 = zi0 + 1
    
    xd = (x-xx[xi0])/(xx[xi1]-xx[xi0])
    yd = (y-yy[yi0])/(yy[yi1]-yy[yi0])
    zd = (z-zz[zi0])/(zz[zi1]-zz[zi0])

    B000 = B[xi0, yi0, zi0, :]
    B100 = B[xi1, yi0, zi0, :]
    B010 = B[xi0, yi1, zi0, :]
    B110 = B[xi1, yi1, zi0, :]
    B001 = B[xi0, yi0, zi1, :]
    B101 = B[xi1, yi0, zi1, :]
    B011 = B[xi0, yi1, zi1, :]
    B111 = B[xi1, yi1, zi1, :]
    
    for B_i in range(3):
        c00 = B000[B_i]*(1-xd) + B100[B_i]*xd
        c01 = B001[B_i]*(1-xd) + B101[B_i]*xd
        c10 = B010[B_i]*(1-xd) + B110[B_i]*xd
        c11 = B011[B_i]*(1-xd) + B111[B_i]*xd
    
        c0 = c00*(1-yd) + c10*yd
        c1 = c01*(1-yd) + c11*yd
    
        c[B_i] = c0*(1-zd) + c1*zd
    
    return c  

def make_filled_circle(r0, num_rays, outer_radius):
    
    #Use the equation from stack overflow about ukrainina graves 
    #to calculate the number of rings including decimal remainder
    num_circles_dec = (-1+np.sqrt(1+4*(num_rays)/(np.pi)))/2
    
    #Get the number of integer rings
    num_circles_int = int(np.floor(num_circles_dec))
    
    #Calculate the number of points per ring with the integer amoung of rings
    num_points_kth_ring = np.round(2*np.pi*(np.arange(0, num_circles_int+1))).astype(int)
    
    #get the remainding amount of rays
    remainder_rays = num_rays - np.sum(num_points_kth_ring)
    
    #Get the proportion of points in each rung
    proportion = num_points_kth_ring/np.sum(num_points_kth_ring)
    
    #resolve this proportion to an integer value, and reverse it
    num_rays_to_each_ring = np.ceil(proportion*remainder_rays)[::-1]
    
    #We need to decide on where to stop adding the remainder of rays to the rest of the rings. 
    #We find this point by summing the rays in each ring from outside to inside, and then getting the index where it is greater than or equal to the remainder
    index_to_stop_adding_rays = np.where(np.cumsum(num_rays_to_each_ring) >= remainder_rays)[0][0]
    
    #We then get the total number of rays to add
    rays_to_add = np.cumsum(num_rays_to_each_ring)[index_to_stop_adding_rays].astype(np.int32)
    
    #The number of rays to add isn't always matching the remainder, so we collect them here with this line
    final_sub = rays_to_add - remainder_rays 
    
    #Here we take them away so we get the number of rays we want
    num_rays_to_each_ring[index_to_stop_adding_rays] = num_rays_to_each_ring[index_to_stop_adding_rays] - final_sub
    
    #Then we add all of these rays to the correct ring
    num_points_kth_ring[::-1][:index_to_stop_adding_rays+1] = num_points_kth_ring[::-1][:index_to_stop_adding_rays+1] + num_rays_to_each_ring[:index_to_stop_adding_rays+1]
    
    #Add one point for the centre, and take one away from the end
    num_points_kth_ring[0] = 1
    num_points_kth_ring[-1] = num_points_kth_ring[-1] - 1
    
    #Make get the radii for the number of circles of rays we need
    radii = np.linspace(0, outer_radius, num_circles_int+1)
    
    idx = 0
    for i in range(len(radii)):
       for j in range(num_points_kth_ring[i]):
           radius = radii[i]
           t = j*(2 * np.pi / num_points_kth_ring[i])
           r0[0, idx, 0] = radius*np.cos(t)
           r0[0, idx, 1] = radius*np.sin(t)
           idx+=1
           
    return r0, num_points_kth_ring