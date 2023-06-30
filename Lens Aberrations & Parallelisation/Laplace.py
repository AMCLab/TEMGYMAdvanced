# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 16:05:34 2021

@author: DAVY
"""
import numpy as np
import sympy as sp
import numba
from sympy.functions.combinatorial.factorials import factorial as fac

class AnalyticalLaplace():
    
    def __init__(self, order = 0):
        #sp.init_printing()
        
        #Coordinate Variables
        self.x, self.y, self.z, self.r, self.alpha, self.theta, self.V = sp.symbols('x y z r alpha theta V')
        
        #Index Variables
        self.k, self.m, self.i, self.j, self.a = sp.symbols('k, m, i, j, a', integer=True)
    
        #Generic Axial Potential 
        self.U_m = sp.Function('U')(self.z, self.m)
        
        #B field for Round Lens
        self.B = sp.Function('B')(self.z)
        
        #Parameter Variables for Dipole
        self.Z, self.R, self.phi, self.NI, self.pi, self.L, self.R_1, self.R_2, self.R_ = sp.symbols('Z R phi NI pi L R_1 R_2, R_')
        self.sin = sp.Function('sin')(self.phi)
        
        #Defining Functions for Dipole Deflector as used by Lencova
        self.gamma = sp.Function('gamma')(self.z, self.Z, self.R)
        self.s1 = sp.Function('s_1')(self.z, self.Z, self.R)
        self.t1 = sp.Function('t_1')(self.z, self.Z, self.R)
        self.t3 = sp.Function('t_3')(self.z, self.Z, self.R)
        self.t5 = sp.Function('t_5')(self.z, self.Z, self.R)
        self.d1 = sp.Function('d_1')(self.z)
        self.d3 = sp.Function('d_3')(self.z)
        self.d5 = sp.Function('d_5')(self.z)

        self.order = order
        
        #create the generic potential function
        self.u_cart = self.MagPotentialCart()
        self.u_cyl = self.MagPotentialCyl()
        
        self.mu = 1.256637e-6 #Vacuum Permeability constant
        
    def MagPotentialCyl(self):
        
        # Explicity copy self variables to make equation more readable and avoid self hell
        z, r, alpha = self.z, self.r, self.alpha
        k, m, i, j = self.k, self.m, self.i, self.j
        U_m = self.U_m
        
        A_m = sp.Sum((((-1)**(k))*fac(m))/((4**(k))*fac(k)*fac(m+k))*(r**(2*k))*(U_m.diff((z, 2*k))), (k, i, j))
        u = (A_m)*(r**(m))*sp.cos(m*alpha)
        
        return u
    
    def MagPotentialCart(self):
        
        # Explicity copy self variables to make equation more readable and avoid self hell
        x, y, z = self.x, self.y, self.z
        k, m, i, j = self.k, self.m, self.i, self.j
        U_m = self.U_m
        
        #convert r^m*cos(m*alp) to x y coordinates see eq.3.61 szilagyi
        rm_cos_malp  = sp.Sum((((-1)**(k))*fac(m))/(fac(2**(k))*fac(m-2*k))*(x**(m-2*k)*y**(2*k)), (k, 0, m))
        A_m = sp.Sum((((-1)**(k))*fac(m))/((4**(k))*fac(k)*fac(m+k))*(x**(2)+y**(2))**(k)*(U_m.diff((z, 2*k))), (k, i, j))
        u = (A_m)*rm_cos_malp
        
        return u
    
    def GlaserBellField(self, a = 1, zpos = 0):
        
        z = self.z
        B = 1/(1+(((z+zpos)/a)**2))
        
        return B
    
    def UnipotentialField(self, a, zpos= 0):
        z = self.z
        B = sp.exp(-(z/a)**2)
        
        return B
    
    def SchiskeField(self, a = 0.025, phi_0 = 5, k = 0.5**(1/2)):
        z = self.z
        phi = phi_0*(1-((k**2)/(1+(z/a)**2)))
        
        return phi
    
    # def SchiskeKormskaField(self, a = 0.025, phi_0 = 30e3, k = 0.5**(1/2)):
    #     z = self.z
    #     x = self.x
    #     Vf = self.V
    #     phi = phi_0*(1-((k**2)/(1+(z/a)**2)))+Vf*(sp.log((x*x+z*z)/(**2)))/(2*sp.log(r/R))
        
    #     return phi
    
    def SchiskeField_(self, a = 0.025, phi_0 = 5, k = 0.5**(1/2)):
        z = self.z
        phi = phi_0*((k**2)/(1+(z/a)**2))
        
        return phi
    
    def KormskaBiprism(self, Vf = 90, r = 0.25e-6, R = 1e-3):
        z = self.z
        x = self.x
        Vf = self.V
        phi = Vf*(sp.log((x*x+z*z)/(R**2)))/(2*sp.log(r/R))
        
        return phi
    
    def KormskaBiprismZ(self, Vf = 90, r = 0.25e-6, R = 1e-3):
        z = self.z
        phi = Vf*(sp.log((z*z)/(R*R)))/(2*sp.log(r/R))
        
        return phi
    
    def KormskaBiprismX(self, Vf = 90, r = 0.25e-6, R = 1e-3):                                                                                      
        x = self.x
        Vf = self.V
        phi = Vf*(sp.log((x*x)/(R**2)))/(2*sp.log(r/R))
        
        return phi
        
    def SeptierBiprism(self, Vf = 90, r = 1e-5, R = 1e-2):
        z = self.z
        x = self.x
        # Vf = self.V
        phi = Vf/(2*sp.log(((np.pi*r)/(4*R))))*(
                    sp.log((sp.cosh((np.pi*z)/(2*R))-sp.cos((np.pi*x)/(2*R)))/
                            (sp.cosh((np.pi*z)/(2*R))+sp.cos((np.pi*x)/(2*R)))))
                    

        return phi
    
    def SeptierBiprismZ(self, phi0 = 3e4, Vf = 90, phi_0 = -3e4, r = 1e-5, R = 1e-2):
        z = self.z
        phi = Vf/(2*sp.log(((np.pi*r)/(4*R))))*(
                    sp.log((sp.cosh((np.pi*z)/(2*R))-1)/
                            (sp.cosh((np.pi*z)/(2*R))+1)))
                    

        return phi
    
    def SeptierBiprismX(self, phi0 = 3e4, Vf = 90, phi_0 = -3e4, r = 1e-5, R = 1e-2):
        x = self.x
        # Vf = self.V
        phi = Vf/(2*sp.log(((np.pi*r)/(4*R))))*(
                    sp.log(1-sp.cos((np.pi*x)/(2*R)))/
                            (1+sp.cos((np.pi*x)/(2*R))))
        
        
        return phi
    
    def GlaserBellFieldStigmatic(self, a = 1, zpos = 0, b = 0.85):
        
        z = self.z
        theta = self.theta
        e = (1-(b/1)**2)**(1/2)
        B = 1/(1+(((z+zpos)/a)**2))*(b/(1-(e*sp.cos(theta))**2)**(1/2))
        
        return B
    
    def RoundLensFieldCyl(self, B):
        #A round lens is composed of simply the 0th Harmonic in Laplace's Equation. Therefore
        #we set m = 0
        
        z, r, alpha,  m, i, j, a = self.z, self.r, self.alpha, self.m, self.i, self.j, self.a
        
        u_cyl = self.u_cyl
        
        order = self.order
        
        #Differentiate once w.r.t z to get z component, and differentiate once w.r.t r to get r component. 
        #B_alpha = 0 as round lens
        #See Szilagyi Page 60
        z_terms_start = 0
        r_terms_start = 1
        z_terms_end = order
        r_terms_end = order + 1
        
        u_cyl_z = u_cyl.subs([[i, z_terms_start],[j, z_terms_end]]).doit()
        u_cyl_r = u_cyl.subs([[i, r_terms_start],[j, r_terms_end]]).doit()

        B_z = u_cyl_z.subs(m, 0).diff(z).subs(sp.diff(sp.Function('U')(z, 0), z), B)
        B_r = u_cyl_r.subs(m, 0).diff(r).subs(sp.diff(sp.Function('U')(z, 0), z), B)
        
        # self.B_z_round_lens = B_z.doit()
        # self.B_r_round_lens = B_r.doit()
        
        # B_z_lambda = sp.lambdify([r, z], B_z.doit(), 'numpy')
        # B_r_lambda = sp.lambdify([r, z], B_r.doit(), 'numpy')
        
        B_lambda = sp.lambdify([r, alpha, z], [B_r.doit(), 0, B_z.doit()], 'numpy')
        
        return B_lambda

    def RoundLensFieldCart(self, B):
        #A round lens is composed of simply the 0th Harmonic in Laplace's Equation. Therefore
        #we set m = 0
        
        x, y, z, m, i, j, a = self.x, self.y, self.z, self.m, self.i, self.j, self.a
        
        u_cart = self.u_cart
        
        order = self.order
        
        #Differentiate once w.r.t z to get z component, and differentiate once w.r.t r to get r component. 
        #B_alpha = 0 as round lens
        #See Szilagyi Page 60
        z_terms_start = 0
        xy_terms_start = 1
        z_terms_end = order
        xy_terms_end = order + 1
        
        u_cart_x = u_cart.subs([[i, xy_terms_start],[j, xy_terms_end]]).doit()
        u_cart_y = u_cart.subs([[i, xy_terms_start],[j, xy_terms_end]]).doit()
        u_cart_z = u_cart.subs([[i, z_terms_start],[j, z_terms_end]]).doit()
        
        # display(u_cart_x.subs(m, 0).diff(x).doit().subs(sp.diff(sp.Function('U')(z, 0), z), 'B_z'))
        # display(u_cart_y.subs(m, 0).diff(y).doit().subs(sp.diff(sp.Function('U')(z, 0), z), 'B_z'))
        # display(u_cart_z.subs(m, 0).diff(z).doit().subs(sp.diff(sp.Function('U')(z, 0), z), 'B_z'))
        # print(sp.printing.latex(u_cart_x.subs(m, 0).diff(x).doit().subs(sp.diff(sp.Function('U')(z, 0), z), 'B_z')), '\n')
        # print(sp.printing.latex(u_cart_y.subs(m, 0).diff(y).doit().subs(sp.diff(sp.Function('U')(z, 0), z), 'B_z')), '\n')
        # print(sp.printing.latex(u_cart_z.subs(m, 0).diff(z).doit().subs(sp.diff(sp.Function('U')(z, 0), z), 'B_z')), '\n')
        
        B_x = u_cart_x.subs(m, 0).diff(x).subs(sp.diff(sp.Function('U')(z, 0), z), B)
        B_y = u_cart_y.subs(m, 0).diff(y).subs(sp.diff(sp.Function('U')(z, 0), z), B)
        B_z = u_cart_z.subs(m, 0).diff(z).subs(sp.diff(sp.Function('U')(z, 0), z), B)

        B_lambda = sp.lambdify((x, y, z), (B_x.doit(), B_y.doit(), B_z.doit()), 'math')
        
        return numba.jit(B_lambda), B_lambda, B_z
    
    def BiprismFieldCartE(self, U):
        
        x, y, z, V = self.x, self.y, self.z, self.V
        
        #E is the -gradient of the potential
        E_x = -U.diff(x)
        E_y = -U.diff(y)
        E_z = -U.diff(z)
        
        U_lambda = sp.lambdify((x, y, z, V), U, 'numpy')
        E_lambda = sp.lambdify((x, y, z, V), (E_x.doit(), E_y.doit(), E_z.doit()), 'numpy')
        
        return E_lambda, U_lambda
    
    def RoundLensFieldCartE(self, U):
        #A round lens is composed of simply the 0th Harmonic in Laplace's Equation. Therefore
        #we set m = 0
        
        x, y, z, m, i, j = self.x, self.y, self.z, self.m, self.i, self.j
        
        u_cart = self.u_cart
        
        order = self.order
        
        #See Szilagyi Page 60
        z_terms_start = 0
        xy_terms_start = 1
        z_terms_end = order
        xy_terms_end = order + 1
        
        u_cart_x = u_cart.subs([[i, xy_terms_start],[j, xy_terms_end]]).doit()
        u_cart_y = u_cart.subs([[i, xy_terms_start],[j, xy_terms_end]]).doit()
        u_cart_z = u_cart.subs([[i, z_terms_start],[j, z_terms_end]]).doit()
        potential = u_cart_z.subs(m, 0).subs(sp.Function('U')(z, 0), U).doit()
        
        #E is the -gradient of the potential
        E_x = -1*u_cart_x.subs(m, 0).diff(x).subs(sp.Function('U')(z, 0), U)
        E_y = -1*u_cart_y.subs(m, 0).diff(y).subs(sp.Function('U')(z, 0), U)
        E_z = -1*u_cart_z.subs(m, 0).diff(z).subs(sp.Function('U')(z, 0), U)
        
        U_lambda = sp.lambdify((x, y, z), (potential), 'numpy')
        E_lambda = sp.lambdify((x, y, z), [E_x.doit(), E_y.doit(), E_z.doit()], 'numpy')
        
        return numba.jit(E_lambda), numba.jit(U_lambda), E_lambda, U_lambda
    
    def RoundLensFieldCartStigmatic(self, B):
        #A round lens is composed of simply the 0th Harmonic in Laplace's Equation. Therefore
        #we set m = 0
        
        x, y, z, m, i, j, a, theta = self.x, self.y, self.z, self.m, self.i, self.j, self.a, self.theta
        
        u_cart = self.u_cart
        
        order = self.order
        
        #Differentiate once w.r.t z to get z component, and differentiate once w.r.t r to get r component. 
        #B_alpha = 0 as round lens
        #See Szilagyi Page 60
        z_terms_start = 0
        xy_terms_start = 1
        z_terms_end = order
        xy_terms_end = order + 1
        
        u_cart_x = u_cart.subs([[i, xy_terms_start],[j, xy_terms_end]]).doit()
        u_cart_y = u_cart.subs([[i, xy_terms_start],[j, xy_terms_end]]).doit()
        u_cart_z = u_cart.subs([[i, z_terms_start],[j, z_terms_end]]).doit()
        
        B_x = u_cart_x.subs(m, 0).diff(x).subs(sp.diff(sp.Function('U')(z, 0), z), B)
        B_y = u_cart_y.subs(m, 0).diff(y).subs(sp.diff(sp.Function('U')(z, 0), z), B)
        B_z = u_cart_z.subs(m, 0).diff(z).subs(sp.diff(sp.Function('U')(z, 0), z), B)

        B_lambda = numba.jit(sp.lambdify((x, y, z, theta), (B_x.doit(), B_y.doit(), B_z.doit()), 'math'))
        
        return B_lambda
    
    def DipoleFieldCyl(self, d1, d3, d5):
        #A dipole field is composed of the 1th, 3rd & 5th Harmonic in Laplace's Equation. Therefore
        #we set m = 1, 3, 5 and also calculate the U1, U3 & U5 axial fields as analytically calculated by Lencova
        #for toroidal and saddle defletors (straight & angled). Can only go up to 5th order as that is all that is published
        
        z, r, alpha, m, mu = self.z, self.r, self.alpha, self.m, self.mu
    
        u_cyl = self.u_cyl
        
        u1 = u_cyl.subs(m, 1)
        u3 = u_cyl.subs(m, 3)
        u5 = u_cyl.subs(m, 5)
        
        B_z_1 = -mu*u1.diff(z, 1)
        B_z_1 = B_z_1.subs(sp.Function('U')(z, 1), d1).doit()
    
        B_r_1 = -mu*u1.diff(r, 1)
        B_r_1 = B_r_1.subs(sp.Function('U')(z, 1), d1).doit()
        
        B_alp_1 = -mu*u1.diff(alpha, 1)
        B_alp_1  = B_alp_1.subs(sp.Function('U')(z, 1), d1).doit()
    
        B_z_3 = -mu*u3.diff(z, 1)
        B_z_3 = B_z_3.subs(sp.Function('U')(z, 3), d3).doit()
        
        B_r_3 = -mu*u3.diff(r, 1)
        B_r_3 = B_r_3.subs(sp.Function('U')(z, 3), d3).doit()
        
        B_alp_3 = -mu*u3.diff(alpha, 1)
        B_alp_3  = B_alp_3.subs(sp.Function('U')(z, 3), d3).doit()
        
        B_z_5 = -mu*u5.diff(z, 1)
        B_z_5 = B_z_5.subs(sp.Function('U')(z, 5), d5).doit()
        
        B_r_5 = -mu*u5.diff(r, 1)
        B_r_5 = B_r_5.subs(sp.Function('U')(z, 5), d5).doit()
        
        B_alp_5 = -mu*u5.diff(alpha, 1)
        B_alp_5  = B_alp_5.subs(sp.Function('U')(z, 5), d5).doit()
    
        B_z = B_z_1 #+ B_z_3 + B_z_5
        B_r = B_r_1 #+ B_r_3 + B_r_5
        B_alp = B_alp_1 #+ B_alp_3 + B_alp_5
        
        self.B_z_lambda = sp.lambdify([r, alpha, z], B_z)
        self.B_r_lambda = sp.lambdify([r, alpha, z], B_r)
        self.B_alp_lambda = sp.lambdify([r, alpha, z], B_alp)
        
        return [self.B_r_lambda, self.B_alp_lambda, self.B_z_lambda]
    
    def DipoleFieldCyl1st(self, d1):
        #A dipole field is composed of the 1th, 3rd & 5th Harmonic in Laplace's Equation. Therefore
        #we set m = 1, 3, 5 and also calculate the U1, U3 & U5 axial fields as analytically calculated by Lencova
        #for toroidal and saddle defletors (straight & angled). Can only go up to 5th order as that is all that is published
        
        z, r, alpha, i, j, m, mu = self.z, self.r, self.alpha, self.i, self.j, self.m, self.mu
        order = self.order
        
        u_cyl = self.u_cyl
        
        u_cyl = u_cyl.subs([[i, 0],[j, order]]).doit()
        u1 = u_cyl.subs(m, 1)
        
        B_z_1 = -mu*u1.diff(z, 1)
        B_z_1 = B_z_1.subs(sp.Function('U')(z, 1), d1).doit()
    
        B_r_1 = -mu*u1.diff(r, 1)
        B_r_1 = B_r_1.subs(sp.Function('U')(z, 1), d1).doit()
        
        B_alp_1 = -mu*u1.diff(alpha, 1)
        B_alp_1  = B_alp_1.subs(sp.Function('U')(z, 1), d1).doit()
    
        B_z = B_z_1 
        B_r = B_r_1 
        B_alp = B_alp_1
        
        self.B_z_lambda = sp.lambdify([r, alpha, z], B_z)
        self.B_r_lambda = sp.lambdify([r, alpha, z], B_r)
        self.B_alp_lambda = sp.lambdify([r, alpha, z], B_alp)
        
        return [self.B_r_lambda, self.B_alp_lambda, self.B_z_lambda]
    
    def DipoleFieldCart(self, d1, d3, d5):
        
        #A dipole field is composed of the 1th, 3rd & 5th Harmonic in Laplace's Equation. Therefore
        #we set m = 1, 3, 5 and also calculate the U1, U3 & U5 axial fields as analytically calculated by Lencova
        #for toroidal and saddle defletors (straight & angled). Can only go up to 5th order as that is all that is published
        
        x, y, z, m, i, j, mu = self.x, self.y, self.z, self.m, self.i, self.j, self.mu
        order = self.order
        
        mu = 1
        
        u_cart = self.u_cart
        u_cart = u_cart.subs([[i, 0],[j, order]]).doit()
        
        u1 = u_cart.subs(m, 1).doit()
        u3 = u_cart.subs(m, 3).doit()
        u5 = u_cart.subs(m, 5).doit()
        
        B_x_1 = -mu*u1.diff(x, 1)
        B_x_1 = B_x_1.subs(sp.Function('U')(z, 1), d1).doit()
        
        B_y_1 = -mu*u1.diff(y, 1)
        B_y_1  = B_y_1.subs(sp.Function('U')(z, 1), d1).doit()
        
        B_z_1 = -mu*u1.diff(z, 1)
        B_z_1 = B_z_1.subs(sp.Function('U')(z, 1), d1).doit()
    
        B_x_3 = -mu*u3.diff(x, 1)
        B_x_3 = B_x_3.subs(sp.Function('U')(z, 3), d3).doit()
        
        B_y_3 = -mu*u3.diff(y, 1)
        B_y_3  = B_y_3.subs(sp.Function('U')(z, 3), d3).doit()
        
        B_z_3 = -mu*u3.diff(z, 1)
        B_z_3 = B_z_3.subs(sp.Function('U')(z, 3), d3).doit()
        
        B_x_5 = -mu*u5.diff(x, 1)
        B_x_5 = B_x_5.subs(sp.Function('U')(z, 5), d5).doit()
        
        B_y_5 = -mu*u5.diff(y, 1)
        B_y_5  = B_y_5.subs(sp.Function('U')(z, 5), d5).doit()
    
        B_z_5 = -mu*u5.diff(z, 1)
        B_z_5 = B_z_5.subs(sp.Function('U')(z, 5), d5).doit()
        

        B_x = B_x_1 + B_x_3 + B_x_5
        B_y = B_y_1 + B_y_3 + B_y_5
        B_z = B_z_1 + B_z_3 + B_z_5
        
        self.B_x_lambda = sp.lambdify([x, y, z], B_x)
        self.B_y_lambda = sp.lambdify([x, y, z], B_y)
        self.B_z_lambda = sp.lambdify([x, y, z], B_z)
        
        return [self.B_x_lambda, self.B_y_lambda, self.B_z_lambda]
    
    def DipoleFieldCart1st(self, d1):
        
        #A dipole field is composed of the 1th, 3rd & 5th Harmonic in Laplace's Equation. Therefore
        #we set m = 1, 3, 5 and also calculate the U1, U3 & U5 axial fields as analytically calculated by Lencova
        #for toroidal and saddle defletors (straight & angled). Can only go up to 5th order as that is all that is published
        
        x, y, z, m, i, j, mu = self.x, self.y, self.z, self.m, self.i, self.j, self.mu
        order = self.order
        
        mu = 1.25663706212e-6
        
        u_cart = self.u_cart
        u_cart = u_cart.subs([[i, 0],[j, order]]).doit()
        
        u1 = u_cart.subs(m, 1).doit()
        
        B_x_1 = -mu*u1.diff(x, 1)
        B_x_1 = B_x_1.subs(sp.Function('U')(z, 1), d1).doit()
        
        B_y_1 = -mu*u1.diff(y, 1)
        B_y_1  = B_y_1.subs(sp.Function('U')(z, 1), d1).doit()
        
        B_z_1 = -mu*u1.diff(z, 1)
        B_z_1 = B_z_1.subs(sp.Function('U')(z, 1), d1).doit()

        B_x = B_x_1
        B_y = B_y_1
        B_z = B_z_1
        
        B_lambda = numba.jit(sp.lambdify([x, y, z], [B_x, B_y, B_z]))

        
        return B_lambda
    
    def QuadrupoleFieldCart(self, d1):
        
        #A quadrupole field is composed of the 2nd, 4th & 6th Harmonic in Laplace's Equation. Therefore
        #we set m = 2, 4, 6.
        
        x, y, z, m, i, j, a = self.x, self.y, self.z, self.m, self.i, self.j, self.a
        mu = 1
        u_cart = self.u_cart
        order = self.order
        
        z_terms_start = 0
        xy_terms_start = 1
        z_terms_end = order
        xy_terms_end = order + 1
        
        u_cart_x = u_cart.subs([[i, xy_terms_start],[j, xy_terms_end]]).doit()
        u_cart_y = u_cart.subs([[i, xy_terms_start],[j, xy_terms_end]]).doit()
        u_cart_z = u_cart.subs([[i, z_terms_start],[j, z_terms_end]]).doit()

        B_x_1 = u_cart_x.subs(m, 2).diff(x).subs(sp.diff(sp.Function('U')(z, 2), z), d1)
        B_y_1 = u_cart_y.subs(m, 2).diff(y).subs(sp.diff(sp.Function('U')(z, 2), z), d1)
        B_z_1 = u_cart_z.subs(m, 2).diff(z).subs(sp.diff(sp.Function('U')(z, 2), z), d1)
    
        B_lambda = numba.jit(sp.lambdify((x, y, z), (B_x_1.doit(), B_y_1.doit(), B_z_1.doit()), 'math'))
        
        return B_lambda

    def StraightToroidalDeflector(self, NI_val, phi_val, L_val, R_1_val, R_2_val):
        
        z = self.z
        
        #Create all variables for equation printing
        gamma, t1, t3, t5, d1, d3, d5 = self.gamma, self.t1, self.t3, self.t5, self.d1, self.d3, self.d5
        
        Z, R, phi, NI, pi, sin, R_1, R_2, L = self.Z, self.R, self.phi, self.NI, self.pi, self.sin, self.R_1, self.R_2, self.L
        
        d1 = (NI/pi)*sin*((t1.subs([[Z, L/2], [R, R_2]])) + t1.subs([[Z, -L/2], [R, R_1]]) - t1.subs([[Z, L/2], [R, R_1]]) - t1.subs([[Z, -L/2], [R, R_2]]))
        d3 = (NI/24*pi)*sin*((t3.subs([[Z, L/2], [R, R_2]])) + t3.subs([[Z, -L/2], [R, R_1]]) - t3.subs([[Z, L/2], [R, R_1]]) - t3.subs([[Z, -L/2], [R, R_2]]))
        d5 = (NI/640*pi)*sin*((t5.subs([[Z, L/2], [R, R_2]])) + t5.subs([[Z, -L/2], [R, R_1]]) - t5.subs([[Z, L/2], [R, R_1]]) - t5.subs([[Z, -L/2], [R, R_2]]))
        
        t1_expr = 1/(R*gamma)
        t3_expr = -1*(3*(gamma**6)-12*(gamma**4)+3*(gamma**2)-2)/((R**3)*(gamma**3))
        t5_expr = -1*(25*(gamma**12)-120*(gamma**10)+230*(gamma**8)-260*(gamma**6)
                               -15*(gamma**4)+20*(gamma**2)-8)/((R**5)*(gamma**5))
        
        gamma_expr = (z-Z)/(sp.sqrt((z-Z)**2+(R**2)))

        #Now Substitute Expressions into equations for evaluation
        t1_expr = t1_expr.subs(gamma, gamma_expr)
        t3_expr = t3_expr.subs(gamma, gamma_expr)
        t5_expr = t5_expr.subs(gamma, gamma_expr)
        
        d1_expr = d1.subs([[t1, t1_expr], [NI, NI_val], [phi, phi_val]])
        d3_expr = d3.subs([[t3, t3_expr], [NI, NI_val], [phi, phi_val]])
        d5_expr = d5.subs([[t5, t5_expr], [NI, NI_val], [phi, phi_val]])
        
        d1_expr = (NI/pi)*sin*((t1_expr.subs([[Z, L_val/2], [R, R_2_val]])) + (t1_expr.subs([[Z, -L_val/2], [R, R_1_val]])) 
                               - (t1_expr.subs([[Z, L_val/2], [R, R_1_val]])) - (t1_expr.subs([[Z, -L_val/2], [R, R_2_val]])))
        
        d1_expr = d1_expr.subs([[NI, NI_val], [phi, phi_val], [pi, sp.pi]])
        
        d3_expr = (NI/24*pi)*sin*((t3_expr.subs([[Z, L_val/2], [R, R_2_val]])) + (t3_expr.subs([[Z, -L_val/2], [R, R_1_val]])) 
                               - (t3_expr.subs([[Z, L_val/2], [R, R_1_val]])) - (t3_expr.subs([[Z, -L_val/2], [R, R_2_val]])))
        
        d3_expr = d3_expr.subs([[NI, NI_val], [phi, phi_val],[pi, sp.pi]])
        
        d5_expr = (NI/640*pi)*sin*((t5_expr.subs([[Z, L_val/2], [R, R_2_val]])) + (t5_expr.subs([[Z, -L_val/2], [R, R_1_val]])) 
                               - (t5_expr.subs([[Z, L_val/2], [R, R_1_val]])) - (t5_expr.subs([[Z, -L_val/2], [R, R_2_val]])))
        
        d5_expr = d5_expr.subs([[NI, NI_val], [phi, phi_val],[pi, sp.pi]])

        return d1_expr, d3_expr, d5_expr
    
    def SaddleDeflector(self, NI_val, phi_val, L_val, R_val):
        z = self.z
        
        #Create all variables for equation printing
        gamma, s1, d1 = self.gamma, self.s1, self.d1
        
        Z, R, phi, NI, pi, sin, L, R_ = self.Z, self.R, self.phi, self.NI, self.pi, self.sin, self.L, self.R_
        
        d1 = (NI/pi)*sin*((s1.subs([[Z, L/2], [R, R_]])) - s1.subs([[Z, -L/2], [R, R_]]))

        s1_expr = gamma*(gamma**2-2)/R
        gamma_expr = (z-Z)/(sp.sqrt((z-Z)**2+(R**2)))

        #Now Substitute Expressions into equations for evaluation
        s1_expr = s1_expr.subs(gamma, gamma_expr)
        
        #Line to check if function looks right
        # d1_expr = d1.subs([[s1, s1_expr], [NI, NI_val], [phi, phi_val]])
        
        d1_expr = (NI/pi)*sin*((s1_expr.subs([[Z, L_val/2], [R, R_val]])) + (s1_expr.subs([[Z, -L_val/2], [R, R_val]])))
        
        d1_expr = d1_expr.subs([[NI, NI_val], [phi, phi_val], [pi, sp.pi]])
        
        return d1_expr
    
    def ResolveBcyl(self, pos, Bcyl):
        x, y, z = pos
        # x = x.ravel().T
        # y = y.ravel().T
        # z = z.ravel().T
        r = np.sqrt(x**2 + y**2)
        angle = np.arctan2(y, x)
        
        Br, Balp, Bz = Bcyl(r, angle, z)
        Bx = Br * np.cos(angle)
        By = Br * np.sin(angle)
        
        return np.array([Bx, By, Bz])

# Laplace = AnalyticalLaplace(0)

# d1, d3, d5 = Laplace.StraightToroidalDeflector(1, np.pi/2, 1, 1, 2)
# Bx, By, Bz = Laplace.DipoleFieldCart(d1, d3, d5)
# Br, Balp, Bzz = Laplace.DipoleFieldCyl(d1, d3, d5)

# A = AnalyticalLaplace(1)
# Bz = A.GlaserBellField(a = 1, B0 = 1)
# B_z, B_r = A.RoundLensField(Bz)

# Bz, Br = A.MagneticLensField()
# u = A.MagPotentialCylindrical()
# u_dif = u.subs(A.m, 0).diff(A.z)
# u_dif = u_dif.subs(sp.diff(sp.Function('U')(A.z, 0), A.z), A.B)
# display(u_dif)
# display(u_dif.doit())

# for f in sp.preorder_traversal(u_dif):
#     if len(f.args)>0:
#         if f.args[0] == (A.z):
#             u_dif.subs(f, A.B)
# display(u_dif)
# display(u_dif.subs(A.U_m, A.B))

# display(Bz)
# display(Br)