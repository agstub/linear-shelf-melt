# this program use SymPy to symbolically derive several quantities that are described
# in the notes: the (1) relaxation and buoyancy transfer functions, (2) the velocity
# solutions, and (3) a limiting value of one of the eigenvalues of the problem.
#
# all of the printing is commented out; this file is best read/used in conjunction
# with the derivation in the appendix.

#---------------------- 1. ELEVATION SOLUTIONS----------------------------------
import sympy as sp
nu = sp.Symbol('n')
mu = sp.exp(nu)
# use this matrix for floating ice:
M = sp.Matrix(( [mu, -1/mu, nu*mu,-nu/mu], [mu, 1/mu, mu*(nu+1),(nu-1)/mu], [1, 1, 1,-1],[1,-1,0,0] ))

b1 = sp.Symbol('b1')                # proportional to h
b2 = sp.Symbol('b2')                # proportional to s

# solution vector
A,B,C,D = sp.symbols('A,B,C,D')

# rhs vector:
b = sp.Matrix(4,1,[b1,0,0,b2])

sol, = sp.linsolve((M,b),[A,B,C,D])

# vertical velocity at upper surface of ice sheet
w_h = mu*sol[0] + (1/mu)*sol[1] + nu*mu*sol[2] + (nu/mu)*sol[3]

# # print the result (modulo a 1/k factor) for floating ice:
# sp.pprint(sp.collect(sp.collect(sp.collect(sp.simplify(w_h),b1),b2),mu) )

# Also need to print w_b for floating ice, since it is part of the solution
# (modulo a 1/k factor)
w_b = sol[0]+sol[1]

# # print this:
# sp.pprint(sp.collect(sp.collect(sp.collect(sp.simplify(w_b),b1),b2),mu) )

#---------------------- 2. VELOCITY SOLUTIONS------------------------------------

# # we also need the vertical velocity at an arbitrary depth z, which we can
# # compute as follows:
z = sp.Symbol('z')

# w(z) modulo a 1/k factor
w = sp.exp(nu*z)*sol[0] + (1/sp.exp(nu*z))*sol[1] + nu*z*sp.exp(nu*z)*sol[2] + nu*z*sol[3]/sp.exp(nu*z)
# # print this:
# sp.pprint(sp.simplify(sp.collect(sp.collect(sp.simplify(w.subs(b1,0)),b1),b2)) )

# coefficients from the elevation solution problem
A,B,C,D = sol

k = sp.Symbol('k')

wb = (A+B)/k                                    # w at z=0
wh = (A*mu + B/mu + nu*mu*C + nu*D/mu)/k        # w at z=H

wz0 = A-B+C+D                                   # dw/dz at z=0

wzh = A*mu -B/mu +C*mu +C*nu*mu -D*nu/mu + D/mu # dw/dz at z=H

# second derivative at z=H
wzzh = A*k*mu + B*k/mu + 2*C*k*mu + C*k*nu*mu + D*k*nu/mu  -2*D*k/mu

# second derivative at z=0
wzz0 = k*(A+B+2*C-2*D)

Ph = wzh - wz0*(mu+1/mu)/2- wzz0*(1/k)*(mu-1/mu)/2   # P(H)

Pzh = wzzh - wz0*k*(mu-1/mu)/2 - wzz0*(mu+1/mu)/2    # P_z(H)

b3 = -(k*wh + Pzh/k)                                 # first rhs vector entry

b4 = -(k*wb)                                         # 2nd rhs vector entry

# Matrix for horizontal surface velocity solutions
M2 = sp.Matrix(( [mu, -1/mu],[1, -1]))

# solution vector
E,F = sp.symbols('E,F')

# RHS vector:
d = sp.Matrix(2,1,[b3,b4])

sol2, = sp.linsolve((M2,d),[E,F])


uh = Ph + sol2[0]*mu + sol2[1]/mu

wz = A*sp.exp(nu*z) - B*sp.exp(-nu*z) + C*(1+nu*z)*sp.exp(nu*z)+ D*(1-nu*z)*sp.exp(-nu*z)

coshkz = (sp.exp(nu*z) + sp.exp(-nu*z))/2
sinhkz = (sp.exp(nu*z) - sp.exp(-nu*z))/2


P = wz -wz0*coshkz- wzz0*sinhkz/k

u = P + sol2[0]*sp.exp(nu*z) + sol2[1]*sp.exp(-nu*z)

## print velocity response functions
# sp.pprint(sp.simplify((sp.collect(sp.collect(u.subs(b2,0),b1),b2))))

#----------------------- 3. EIGENVALUE LIMIT------------------------------------
k = sp.Symbol('k',positive=True)
d = sp.Symbol('delta',positive=True)
mu = sp.exp(k)

R0 = (mu**4)+4*k*(mu**2)-1
D = k*(mu**4-2*(1+2*k**2)*(mu**2)+1)
B0 = 2*(k+1)*(mu**3) + 2*(k-1)*mu

R = R0/D
B = B0/D

# symbolic expression for the larger eigenvalue in the problem:
lamda = -(d+1)*R*(1- sp.sqrt((4*d/(d+1)**2)*((B0/R0)**2  -1) + 1))/2

# take the limit as k --> 0:
L = sp.limit(lamda,k,0)

# # print the limit:
# sp.pprint(sp.factor(sp.simplify(L)))
