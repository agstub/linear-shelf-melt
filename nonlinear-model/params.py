# All model/numerical parameters are set here.

# Model parameters
A0 = 1e-32                         # Glen's law coefficient (ice softness, Pa^{-n}/s)
n = 4.0                            # Glen's law exponent


B0 = A0**(-1/n)                    # Ice hardness (Pa s^{1/n})
B = (2**((n-1.0)/(2*n)))*B0        # "2*Viscosity" constant in weak form (Pa s^{1/n})
rm2 = 1 + 1.0/n - 2.0              # Exponent in weak form: r-2

rho_i = 917.0                      # Density of ice
rho_w = 1020.0                     # Density of water
g = 9.81                           # Gravitational acceleration
eta0 = 1e14                        # viscosity at zero deviatoric stress 

L = 40*1000.0                      # Length of the domain
H = 500.0                          # Height of the domain
sea_level = H*(rho_i/rho_w)        # Sea level elevation.
z_max = 0.9*H                      # Maximum channel height
t_r = 2*eta0/(rho_i*g*H)           # viscous relaxation time scale

# Numerical parameters
eps_v = (2*eta0/B)**(2.0/rm2)      # Flow law regularization parameter

# Mesh parameters
Nx = int(L/100)                    # Number of elements in x direction
Nz = int(H/100)                    # Number of elements in z direction

# Time-stepping parameters
t_e = (4*eta0/((rho_w-rho_i)*g*H))*(rho_w/rho_i)
t_f = 5*t_e                        # Final time (in terms of relaxation timescale)
nt = 10*int(t_f/t_r)               # Number of time steps
dt = t_f/nt                        # Timestep size

save_vtk = False                   # Flag for saving solutions in VTK format
