# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 14:32:26 2023

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

biprism_z = 0.0
wavelength = 0.5 * 1e-6

a = 0
b = 130e-6
biprism_pos = 5e-6
num_rays = 1000001

n1 = 1.0
n2 = 1.5

# Define the vertices of the triangle (A, B, C)
biprism_height = 5e-6
biprism_length = 25e-6
top_pt_tri = (biprism_pos + 0, biprism_length)
mid_pt_tri = (biprism_pos + biprism_height, 0)
bot_pt_tri = (biprism_pos + 0, -biprism_length)

ray_fig, ray_ax = plt.subplots()
ray_ax.plot([top_pt_tri[0], mid_pt_tri[0]], [top_pt_tri[1], mid_pt_tri[1]], 'b-')
ray_ax.plot([mid_pt_tri[0], bot_pt_tri[0]], [mid_pt_tri[1], bot_pt_tri[1]], 'b-')
ray_ax.plot([bot_pt_tri[0], top_pt_tri[0]], [bot_pt_tri[1], top_pt_tri[1]], 'b-')
ray_ax.set_xlim([0, b])
ray_ax.axis('equal')

source_up = 0
source_low = 0
steps = 4

z_pos = np.linspace(a, a, num_rays)
x_pos = np.linspace(-biprism_length*0.8, biprism_length*0.8, num_rays)
x_slope = np.concatenate((np.linspace(-source_up, -source_low, num_rays//2, endpoint = True), 
                         np.linspace(source_low, source_up, num_rays//2+1, endpoint = True))) 

x_pos_out = np.zeros((num_rays, steps), np.float64)
z_pos_out = np.zeros((num_rays, steps), np.float64)

x_slope_out = np.zeros((num_rays, steps), np.float64)
opl_out = np.zeros((num_rays, steps), np.float64)

x_pos_out[:, 0] = x_pos
x_slope_out[:, 0] = x_slope
z_pos_out[:, 0] = z_pos

for step in range(1, steps):
    for x_idx, x0 in enumerate(x_pos_out[:, step-1]):
        x_0 = x_slope_out[x_idx, step-1]
        
        if step == 1:
            n1 = 1.0
            n2 = 1.0
            
            z0 = z_pos_out[x_idx, step-1]
            z1 = biprism_pos
            z_dist = z1-z0
            
            x1 = x0 + x_0*z_dist
            
            z_pos_out[x_idx, step] = z1
            x_pos_out[x_idx, step] = x1
            x_slope_out[x_idx, step] = x_0

            dist = np.sqrt((x_pos_out[x_idx, step]-x_pos_out[x_idx, step])**2 + z_dist**2)
            opl_out[x_idx, step] = opl_out[x_idx, step-1] + n2*dist
            
        elif step == 2:
            
            n1 = 1.0
            n2 = 1.5
            z0 = z_pos_out[x_idx, step-1]
            z1 = b
            
            theta1 = np.tan(x_0)
            theta2 = np.arcsin(n1/n2*np.sin(theta1))
            
            x_1 = np.tan(theta2)
            
            x1 = x0 + x_1*z_dist
            
            if x0 > 0:
                zint, xint = line_intersection(((z0, x0), (z1, x1)), ((mid_pt_tri[0], mid_pt_tri[1]), (top_pt_tri[0], top_pt_tri[1])))
            elif x0 < 0:  
                zint, xint = line_intersection(((z0, x0), (z1, x1)), ((bot_pt_tri[0], bot_pt_tri[1]), (mid_pt_tri[0], mid_pt_tri[1])))
            
            z_pos_out[x_idx, step] = zint
            x_pos_out[x_idx, step] = xint
            x_slope_out[x_idx, step] = x_1
            
            dist = np.sqrt((xint-x0)**2 + (zint-z0)**2)
            opl_out[x_idx, step] = opl_out[x_idx, step-1] + n2*dist
        
        if step == 3:
            
            n1 = 1.5
            n2 = 1.0
            x0 = x_pos_out[x_idx, step-1]
            z0 = z_pos_out[x_idx, step-1]
            z1 = b
            z_dist = z1 - z0
            
           
            ray_vector = np.array([z0 - z_pos_out[x_idx, step-2], x0 - x_pos_out[x_idx, step-2]])
            ray_unit_vector = ray_vector/np.sqrt(np.sum(ray_vector**2))
            
            
            if x0 > 0:
                surface_vector = np.array([top_pt_tri[0] - z0, top_pt_tri[1] - x0])
                norm_vector = np.array([surface_vector[1], -surface_vector[0]])
                norm_unit_vector = norm_vector/np.sqrt(np.sum(norm_vector**2))
                
            elif x0 <0:
                surface_vector = np.array([bot_pt_tri[0] - z0, bot_pt_tri[1] - x0])
                norm_vector = np.array([-surface_vector[1], surface_vector[0]])
                norm_unit_vector = norm_vector/np.sqrt(np.sum(norm_vector**2))
                
            alpha = np.dot(ray_unit_vector, norm_unit_vector)
            
            k1 = (-2*alpha+np.sqrt((2*alpha)**2-4*alpha*(1-(n2/n1)**2)))/(2)
            k2 = (-2*alpha-np.sqrt((2*alpha)**2-4*alpha*(1-(n2/n1)**2)))/(2)
            
            F = ray_unit_vector + k1*norm_unit_vector
            
            slope_out = F[1]/F[0]
            x_slope_out[x_idx, step] = slope_out
            
            x1 = x0 + z_dist*slope_out
            
            x_pos_out[x_idx, step] = x1
            z_pos_out[x_idx, step] = z1
            
            dist = np.sqrt((x1-x0)**2 + z_dist**2)
            opl_out[x_idx, step] = opl_out[x_idx, step-1] + n2*dist
            
fractional_wavelength = opl_out[:, -1]/wavelength
phase_out = np.mod(fractional_wavelength*2*np.pi, 2*np.pi)

ray_ax.plot(z_pos_out[::10000, :].T, x_pos_out[::10000, :].T, '-r')

x_pixels = 1001
y_pixels = 1

image = np.zeros((x_pixels, y_pixels), np.float64)

scale = biprism_length*2
pixel_size = (2*scale)/x_pixels

bins = np.arange(-scale, scale+pixel_size, pixel_size)

hist, _ = np.histogram(x_pos_out[:, -1], bins=(bins))

x_bin_indices = np.digitize(x_pos_out[:, -1], bins, right = True) - 1

bin_values = defaultdict(list)

for i, (x_idx) in enumerate(x_bin_indices):
    if x_idx >= x_pixels:
        continue
    if x_idx < 0:
        continue
    
    bin_values[(x_idx, 0)].append(phase_out[i])  

#Loop through the coordinates, find the phases of all the rays in each pixel, and calculate the interference. 
for x, y in bin_values:
    phases = np.array(bin_values[x, y])
    image[x, y] = np.abs(np.sum(np.exp(1j*(phases))))**2
    

image_intensity_fig, image_intensity_ax = plt.subplots()

image_intensity = image.T
image_intensity_tiled = np.tile(image_intensity.T, 1000).T

#Plot the intensity image
image_intensity_ax.imshow(image_intensity_tiled/np.max(image_intensity), origin = 'lower', extent = (-scale, scale, -scale, scale))

x_intensity = bins[:-1] + pixel_size/2

intenisty_fig, intenisty_ax = plt.subplots()
intenisty_ax.plot(x_intensity, image_intensity[0, :]/np.max(image_intensity), label = 'Intensity')
intenisty_ax.set_xlabel('x (m)')
intenisty_ax.set_ylabel('I')
intenisty_ax.legend()

from diffractio import degrees, mm, plt, sp, um, np
from diffractio.scalar_masks_XZ import Scalar_mask_XZ
from diffractio.scalar_sources_X import Scalar_source_X

b = b / 1e-6
biprism_pos = biprism_pos/1e-6

biprism_height = biprism_height/1e-6
biprism_length = biprism_length/1e-6

num_x = 1024 * 4
num_z = 1024 * 8
x0 = np.linspace(-biprism_length * um, biprism_length * um, num_x)
z0_near = np.linspace(0 * um, b * um, num_z)
z0_all = np.linspace(0 * um, b * um, num_z)

wavelength = wavelength / 1e-6 * um

u0 = Scalar_source_X(x=x0, wavelength=wavelength)
u0.plane_wave(A = 1)

t0 = Scalar_mask_XZ(x=x0, z=z0_near, wavelength=wavelength)
t0.biprism(r0=(0 * um, biprism_pos * um),
            length=2*biprism_length * um,
            height= biprism_height * um,
            refraction_index=1.5,
            angle=0 * degrees)
t0.incident_field(u0)

t0.WPM(verbose=True, has_edges = False)

t0.draw(kind='intensity', draw_borders=True)
wave_xz_ax = plt.gca()
wave_xz_ax.axis('square')
wave_xz_ax.plot(z_pos_out[::50000, :].T/1e-6, x_pos_out[::50000, :].T/1e-6, '-b')
wave_xz_ax.set_xlim(0, b)
wave_xz_ax.set_ylim(-biprism_length, biprism_length)

t0.draw(kind='phase', draw_borders=True)
wave_xz_ax = plt.gca()
wave_xz_ax.axis('square')
wave_xz_ax.plot(z_pos_out[::50000, :].T/1e-6, x_pos_out[::50000, :].T/1e-6, '-b')
wave_xz_ax.set_xlim(0, b)
wave_xz_ax.set_ylim(-biprism_length, biprism_length)

u_field_last_plane = t0.profile_transversal(z0= b * um, normalize = 'maximum')
wave_x_ax = plt.gca()
wave_x_ax.plot(x_intensity/1e-6, image_intensity[0, :]/np.max(image_intensity), label = 'Far Field Intensity Numeric')
plt.ylim(-10 * um, 10 * um)
plt.legend()


