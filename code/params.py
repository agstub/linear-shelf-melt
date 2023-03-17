# this file sets the physical and numerical parameters in the problem

import numpy as np
from scipy.fft import fftfreq

#---------------------- physical parameters ------------------------------------

# dimensional parameters
H = 1000                    # ice thickness (m)
eta = 1e13                  # Newtonian ice viscosity (Pa s)
rho_i = 917                 # ice density (kg/m^3)
rho_w = 1000                # water density (kg/m^3)
g = 9.81                    # gravitational acceleration
delta = rho_w/rho_i-1       # flotation factor
t_r = 2*eta/(rho_i*g*H)     # viscous relaxation time

eps_shelf = 2.529e-14       # regularization corresponding to the relaxation("R") and buoyancy ("B")
                            # response time in the long-wavelength limit k-->0.
                            #
                            # ^Notes: this is chosen to be this value
                            # so that the long-wavelength limiting behavior
                            # R-B --> 1/4 is attained at k=1e-3, which means that
                            # the other limiting behaviors (i.e. of the larger eigenvalue)
                            # are also respected for small k. Otherwise, the solutions
                            # tend to be bad at small k because the R and B functions
                            # blow as k-->0.

#---------------------- numerical parameters------------------------------------
# discretization parameters
Nx = 201                    # number of grid points in x-direction
Ny = 201                    # number of grid points in y-direction
Nt = 200                    # number of time steps

t_final = 200               # final time (in multiples of t_r)

t0 = np.linspace(0,t_final,num=Nt) # time array

dt = t0[1]

L = 50                         # horizontal x-y domain is a square of length 2L (default L=40)
x0 = np.linspace(-L,L,num=Nx)  # x coordinate array
y0 = np.linspace(-L,L,num=Ny)  # y coordinate array
dx = np.abs(x0[1]-x0[0])       # grid size in x direction'
dy = np.abs(y0[1]-y0[0])       # grid size in y direction

# frequency
kx0 =  fftfreq(Nx,dx)
ky0 =  fftfreq(Ny,dy)

# set zero frequency to small number because some functions in the integral kernels
# have singularities at the zero frequency
kx0[0] = 1e-15
ky0[0] = 1e-15

# mesh grids for physical space domain
t,x,y = np.meshgrid(t0,x0,y0,indexing='ij')

# mesh grids for frequency domain
t,kx,ky = np.meshgrid(t0,kx0,ky0,indexing='ij')

# magnitude of the wavevector
k = np.sqrt(kx**2+ky**2)
