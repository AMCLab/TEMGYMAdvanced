
   
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:31:50 2022
@author: andy
"""
import matplotlib.pyplot as plt

import numpy as np
from b_field_functions import (
    SaddleGeometry,
    SaddleBField,
    LineGeometry,
    LensGeometry,
)

from particle_tracing_cpu import (
    make_filled_circle,
    euler_dz_simultaneous_rays,
    euler_dz_simultaneous_rays_astig,
    euler_dz_simultaneous_rays_interp,
)

import pyqtgraph.opengl as gl
from Laplace import AnalyticalLaplace

class TemGym():
    
    def __init__(self, U, num_rays, detector_size, detector_pixels):
        
        #Define constants
        q = -1.60217662e-19
        m = 9.10938356e-31
        c = 2.99792458e8
        
        #Detector parameters
        self.detector_pixels = detector_pixels
        self.detector_size = detector_size
        self.detector_diagonal = np.sqrt((detector_size/2)**2+(detector_size/2)**2)
        self.num_rays = num_rays
        
        #Calculate acceleration voltage at 300kV
        self.v_acc = 1*c*(1-(1-(q*U)/(m*(c**2)))**(-2))**(1/2)
        self.gamma = 1/(1-(self.v_acc**2/c**2))**(1/2)
        self.alpha = q/(self.gamma*m*self.v_acc)
        
        #Choose initial angle of particle in polar direction (radians)
        self.polar_angle = 0
        self.alpha_angle = 0

        self.start_z = 2e-1

        #Generate initial ray conditions
        self.outer_radius = 1e-4
        
        self.z0 = 2e-1
        
        #setup step counter
        self.steps = 2
        
        self.component_dict = self.component_setup()
        
        #Calculate the number of steps in the position array
        for component in self.component_dict:
            self.steps+= self.component_dict[component]['Steps']

        self.steps = self.steps + len(self.component_dict) + 1

        #Make a 2 x steps x number of rays size array to store the particle locations at each step
        self.r = np.zeros((self.steps, self.num_rays, 3), dtype = np.float64)
        self.r, self.ring_idcs = make_filled_circle(self.r.copy(), self.num_rays, self.outer_radius)

        self.polar_init = np.ones(self.num_rays, dtype = np.float64)
        self.alpha_init = np.ones(self.num_rays, dtype = np.float64)

        self.polar_angle = np.pi
        self.polar = self.polar_init * self.polar_angle

        self.v = np.zeros((self.steps, self.num_rays, 2), dtype = np.float64)

        #set initial velocity vectors
        self.v[:2, :, 0] = np.sin(self.polar) * np.cos(self.alpha_init)
        self.v[:2, :, 1] = np.sin(self.polar) * np.sin(self.alpha_init)

        self.polar_angles = []
        self.circularities = []
        self.def_ratios = []

        #define initial model parameters
        self.esource_polar = 0
        self.lens_bmag = 3
        
        self.time = 0
        
        self.current_spot_radial_pos = 0
        
        self.average_positions = []
        self.upper_quadrupole_bmag = 0
        self.lower_quadrupole_bmag = 0
        self.reset()
        
    def step(self, action):
        
        self.r[0, :, 2] = self.z0
        
        #Initialise image arrays
        image_pixel_coords = np.array([])
        self.detector_image = np.zeros((self.detector_pixels, self.detector_pixels), dtype = np.uint8)
        
        self.polar_angle = np.sin(np.pi + self.esource_polar)
        
        self.v[:2, :, 0] = self.polar_init * self.polar_angle
        
        #first_object_distance
        component_top = self.component_dict[list(self.component_dict.keys())[0]]['Top']
        
        z_to_first = abs(component_top - self.z0)
        
        #propagate the beam to the start of the first object
        self.r[1, :, 0] = self.r[0, :, 0] + (z_to_first * self.v[0, :, 0])/np.sqrt(1 + self.v[0, :, 0]**2)
        self.r[1, :, 1] = self.r[0, :, 1] + (z_to_first * self.v[0, :, 1])/np.sqrt(1 + self.v[0, :, 1]**2)
        self.r[1, :, 2] = component_top
        
        step = 1
        
        #trace rays through each component
        for component in self.component_dict:
            component_top = self.component_dict[component]['Top']
            z_prop = self.component_dict[component]['Len_To_Next']
            
            if component == "Upper_Saddle_Quad":
                steps = self.component_dict[component]['Steps']+1
                dz = self.component_dict[component]['dz']
                
                Bmag = self.upper_quadrupole_bmag 
                print(Bmag)
                
                B_grid = self.component_dict[component]['B_Field']
                grid = self.component_dict[component]['Grid']
                grid_step = self.component_dict[component]['GridStep']
                
                self.r, self.v = euler_dz_simultaneous_rays_interp(self.r, self.v, component_top, self.alpha, -dz, z_prop, B_grid, grid, grid_step, Bmag, steps, step)
                print(self.r[steps, 0, 2])
                step = step + steps + 1
            
            elif component == "Lower_Saddle_Quad":
                steps = self.component_dict[component]['Steps']+1
                dz = self.component_dict[component]['dz']
                
                Bmag = self.lower_quadrupole_bmag 
                print(Bmag)
                
                B_grid = self.component_dict[component]['B_Field']
                grid = self.component_dict[component]['Grid']
                grid_step = self.component_dict[component]['GridStep']
                
                self.r, self.v = euler_dz_simultaneous_rays_interp(self.r, self.v, component_top, self.alpha, -dz, z_prop, B_grid, grid, grid_step, Bmag, steps, step)
                print(self.r[steps, 0, 2])
                step = step + steps + 1
                
            elif component == "Lens_One":
                B = self.component_dict[component]['B_Field']
                dz = self.component_dict[component]['dz']
                print(Bmag)
                z_prop = 0
                steps = self.component_dict[component]['Steps']+1
                Bmag = self.lens_bmag
                self.r, self.v = euler_dz_simultaneous_rays_astig(self.r, self.v, component_top, self.alpha, -dz, z_prop, B, Bmag, steps, step)
    
        # set final image pixel coordinates
        image_pixel_coords = (
            np.round(self.r[-1, :, :2] / (self.detector_size) * self.detector_pixels) + self.detector_pixels//2-1
        ).astype(np.int32)
        
        ave_pos = np.average(image_pixel_coords, axis = 0)

        ave_radial_pos = np.sqrt(
            (ave_pos[0]-self.detector_pixels//2)**2 +
            (ave_pos[1]-self.detector_pixels//2)**2
        )
        
        self.average_positions.append(ave_radial_pos)
    
        #Check if we have satisfied the failure mode which is for any pixel to have left the screen
        if  np.any(image_pixel_coords >= self.detector_pixels) or np.any(image_pixel_coords < 0):
            
            image_pixel_coords = np.delete(image_pixel_coords, np.where(
                (image_pixel_coords < 0) | (image_pixel_coords >= self.detector_pixels)), axis = 0)
            
            self.detector_image[
                image_pixel_coords[:, 0],
                image_pixel_coords[:, 1],
            ] = 1
            
            self.reward = -1.0
            self.done = 1
            
            return self.detector_image, self.reward, self.done
        
        #Check if we have satisfied the done condition, which is that the last 3 average radial spot positions, are on average less than 2 pixels from the centre
        elif len(self.average_positions) >= 3 and np.average(self.average_positions[-4:]) < 2:
            self.done = 1
            self.reward = 1.0
            
            self.detector_image[
                image_pixel_coords[:, 0],
                image_pixel_coords[:, 1],
            ] = 1
            
            return self.detector_image, self.reward, self.done
        
        #return the reward for making a good or bad move. 
        elif len(self.average_positions) > 1:
            self.done = 0
            p = 1.0
            tau = 20
            c = 0.4
            
            if ave_radial_pos > p:
                self.reward = c*2**(-(ave_radial_pos-p)/tau)
            else:
                self.reward = 1
                
            self.detector_image[
                image_pixel_coords[:, 0],
                image_pixel_coords[:, 1],
            ] = 1
            
            return self.detector_image, self.reward, self.done

        #final return statement if no conditions are met, which happens when we reset
        else:
            self.detector_image[
                image_pixel_coords[:, 0],
                image_pixel_coords[:, 1],
            ] = 1
            
            return self.detector_image, 0, 0
        
    def reset(self):
        
        # self.quadrupole_bmag = np.random.uniform(0, 15)*np.random.choice([-1, 1])
        # print(self.quadrupole_bmag)
        self.average_positions = []
        self.detector_image, _, _ = self.step(0) 

        return self.detector_image
        
    def component_setup(self):
        
        ###TEM Component Specification
        component_dict = {
            "Upper_Saddle_Quad": {
                "Type": "Quadrupole",
                "Current_Sign": 1,
                "Position": [0, 0, 0.08],
                "Top": 0.09,
                "Bottom":0.07,
                "Length": 0.02,
                "Len_To_Next":0.01,
                "dz":1e-4,
                "Steps":int(round(0.02/1e-4)),
                "Radius": 0.01,
                "Angle": 0,
                "Offset": 0,
                "n_arc": 50,
                "n_lengths": 50,
                "MultiPole_Type": "Quadrupole",
            },
            "Lower_Saddle_Quad": {
                "Type": "Quadrupole",
                "Current_Sign": 1,
                "Position": [0, 0, 0.05],
                "Top": 0.06,
                "Bottom":0.04,
                "Length": 0.02,
                "Len_To_Next":0,
                "dz":1e-4,
                "Steps":int(round(0.02/1e-4)),
                "Radius": 0.01,
                "Angle": np.pi/4,
                "Offset": 0,
                "n_arc": 50,
                "n_lengths": 50,
                "MultiPole_Type": "Quadrupole",
            },
            "Lens_One": {
                "Type": "Lens",
                "Current_Sign": 1,
                "Position": [0, 0, 0.02],
                "Top": 0.04,
                "Bottom": 0.00,
                "Len_To_Next":0.00,
                "Length": 0.04,
                "dz":4e-5,
                "Steps":int(round(0.04/4e-5)),
                "Radius": 0.003,
                "a": 1e-3,
            },
        }
        
        # Draw components and calculate Their BField
        for component in component_dict:
            if component_dict[component]["Type"] == "Quadrupole":
                Position = component_dict[component]["Position"]
                R = component_dict[component]["Radius"]
                L = component_dict[component]["Length"]
                PHI = component_dict[component]["Angle"]
                TOP = component_dict[component]["Top"]
                BOTTOM = component_dict[component]["Bottom"]
                offset = component_dict[component]["Offset"]
                n_arc = component_dict[component]["n_arc"]
                n_lengths = component_dict[component]["n_lengths"]
                n = 50
                
                points = LineGeometry(R, L, Position, 0+PHI, n_lengths)
                component_dict[component]["Saddle_Points_One"] = points
                bfield_extent = np.array([[R, -R], [R, -R], [TOP, BOTTOM]])
                saddle = gl.GLLinePlotItem(pos=points, color="w", width = 5)
                component_dict[component]["3D_Mesh_One"] = saddle
                BField1, _, _ = SaddleBField(points, Position, n, -1, bfield_extent)
                
                points = LineGeometry(R, L, Position, -np.pi/2+PHI, n_lengths)
                component_dict[component]["Saddle_Points_Two"] = points
                bfield_extent = np.array([[R, -R], [R, -R], [TOP, BOTTOM]])
                saddle = gl.GLLinePlotItem(pos=points, color="w", width = 5)
                component_dict[component]["3D_Mesh_Two"] = saddle
                BField2, _, _ = SaddleBField(points, Position, n, 1, bfield_extent)
                
                points = LineGeometry(R, L, Position, np.pi/2+PHI, n_lengths)
                component_dict[component]["Saddle_Points_Three"] = points
                bfield_extent = np.array([[R, -R], [R, -R], [TOP, BOTTOM]])
                saddle = gl.GLLinePlotItem(pos=points, color="w", width = 5)
                component_dict[component]["3D_Mesh_Three"] = saddle
                BField3, _, _ = SaddleBField(points, Position, n, 1, bfield_extent)

                points = LineGeometry(R, L, Position, np.pi+PHI, n_lengths)
                component_dict[component]["Saddle_Points_Four"] = points
                bfield_extent = np.array([[R, -R], [R, -R], [TOP, BOTTOM]])
                saddle = gl.GLLinePlotItem(pos=points, color="w", width = 5)
                component_dict[component]["3D_Mesh_Four"] = saddle
                BField4, xyz_grid, xyz_grid_step = SaddleBField(points, Position, n, -1, bfield_extent)
                BField = BField1 + BField2 + BField3 + BField4
                
                component_dict[component]["B_Field"] = BField
                component_dict[component]["Grid"] = xyz_grid
                component_dict[component]["GridStep"] = xyz_grid_step
                
                '''Special Plotting Function
                x, y, z = np.meshgrid(np.linspace(-0.01, 0.01, 50), np.linspace(-0.01, 0.01, 50), np.linspace(0.07, 0.09, 50))
                fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
                points1 = component_dict[component]["Saddle_Points_One"]
                points2 = component_dict[component]["Saddle_Points_Two"]
                points3 = component_dict[component]["Saddle_Points_Three"]
                points4 = component_dict[component]["Saddle_Points_Four"]
                plt.plot(points1[:, 0], points1[:, 1], points1[:, 2], 'r')
                plt.plot(points2[:, 0], points2[:, 1], points2[:, 2], 'r')
                plt.plot(points3[:, 0], points3[:, 1], points3[:, 2], 'r')
                plt.plot(points4[:, 0], points4[:, 1], points4[:, 2], 'r')
                plt.quiver(x[:,:,25], y[:,:,25], z[:,:,25], BField[:,:,25, 0], BField[:,:,25, 1], BField[:,:,25, 2], length = 0.0005, normalize = True)
                '''
            
            if component_dict[component]["Type"] == "Lens":
                Position = component_dict[component]["Position"]
                R = component_dict[component]["Radius"]
                L = component_dict[component]["Length"]
                n = 101

                lens = LensGeometry(R, L, Position, n)
                a = component_dict[component]["a"]

                Lap = AnalyticalLaplace(0)
                BField = Lap.RoundLensFieldCartStigmatic(
                    Lap.GlaserBellFieldStigmatic(a = a, zpos = -Position[2]))
                
                component_dict[component]["3D_Mesh"] = lens
                component_dict[component]["B_Field"] = BField
                
        return component_dict
    