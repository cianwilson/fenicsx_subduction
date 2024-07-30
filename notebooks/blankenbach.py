#!/usr/bin/env python
# coding: utf-8

# # Blankenbach Thermal Convection Example
# 
# Authors: Cameron Seebeck, Cian Wilson

# ## Implementation

# ### Preamble

# In[ ]:


from mpi4py import MPI
import dolfinx as df
import dolfinx.fem.petsc
import numpy as np
import ufl
import matplotlib.pyplot as pl
import basix
import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
sys.path.append(os.path.join(basedir, os.path.pardir, 'python'))
import utils
import pyvista as pv
if __name__ == "__main__" and "__file__" in globals():
    pv.OFF_SCREEN = True
import pathlib
if __name__ == "__main__":
    output_folder = pathlib.Path(os.path.join(basedir, "output"))
    output_folder.mkdir(exist_ok=True, parents=True)


# In[ ]:


def T0(x):
    return 1.-x[1] + 0.2*np.cos(x[0]*np.pi)*np.sin(x[1]*np.pi)

def eta_T(b=1000):
    return ufl.exp(-np.log(b)*T)


# In[ ]:


def solve_blankenbach(Ra, ne, p=1, b=None, alpha=0.8, rtol=5.e-6, atol=5.e-9, maxits=50):
    """
    A python function to solve two-dimensional thermal convection 
    in a unit square domain.  By default this assumes an isoviscous rheology 
    but a UFL expression for the viscosity can be passed in using the kwarg eta.
    Parameters:
    * Ra  - the Rayleigh number
    * ne  - number of elements in each dimension
    * p   - polynomial order of the pressure and temperature solutions (defaults to 1)
    * eta - convergence speed of lower boundary (defaults to 1)
    """
    # Describe the domain (a unit square)
    # and also the tessellation of that domain into ne
    # equally spaced squared in each dimension, which are
    # subduvided into two triangular elements each
    mesh = df.mesh.create_unit_square(MPI.COMM_WORLD, ne, ne)

    # Define velocity, pressure and temperature elements
    v_e = basix.ufl.element("Lagrange", mesh.basix_cell(), p+1, shape=(mesh.geometry.dim,))
    p_e = basix.ufl.element("Lagrange", mesh.basix_cell(), p)
    T_e = basix.ufl.element("Lagrange", mesh.basix_cell(), p)

    # Define the mixed element of the coupled velocity and pressure
    vp_e = basix.ufl.mixed_element([v_e, p_e])

    # Define the mixed velocity-pressure function space
    V_vp = df.fem.functionspace(mesh, vp_e)
    # Define the temperature function space
    V_T  = df.fem.functionspace(mesh, T_e)

    # Define velocity and pressure sub function spaces
    V_v,  _ = V_vp.sub(0).collapse()
    V_vx, _ = V_v.sub(0).collapse()
    V_vy, _ = V_v.sub(1).collapse()
    V_p,  _ = V_vp.sub(1).collapse()

    # Define the finite element function for the temperature and initialize it
    # with the initial guess
    T = df.fem.Function(V_T)
    T.interpolate(T0)
    # Also define a finite element function for the temporary solution at each iteration
    T_i = df.fem.Function(V_T)

    # Define the finite element functions for the velocity and pressure functions
    vp = df.fem.Function(V_vp)
    v = vp.sub(0)
    p = vp.sub(1)
    # Also define a finite element function for the temporary solution at each iteration
    vp_i = df.fem.Function(V_vp)

    # Define the velocity and pressure test functions
    v_t, p_t = ufl.TestFunctions(V_vp)
    # Define the temperature test function
    T_t = ufl.TestFunction(V_T)

    # Define the velocity and pressure trial functions
    v_a, p_a = ufl.TrialFunctions(V_vp)
    # Define the temperature trial function
    T_a = ufl.TrialFunction(V_T)

    # Declare a list of boundary conditions for the Stokes problem
    bcs_s = []

    # Define the location of the left and right boundary and find the x-velocity DOFs
    def boundary_leftandright(x):
        return np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1))
    dofs_vx_leftright = df.fem.locate_dofs_geometrical((V_vp.sub(0).sub(0), V_vx), boundary_leftandright)
    # Specify the velocity value and define a Dirichlet boundary condition
    zero_vx = df.fem.Function(V_vx)
    zero_vx.x.array[:] = 0.0
    bcs_s.append(df.fem.dirichletbc(zero_vx, dofs_vx_leftright, V_vp.sub(0).sub(0)))

    # Define the location of the top and bottom boundary and find the y-velocity DOFs
    def boundary_topandbase(x):
        return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1))
    dofs_vy_topbase = df.fem.locate_dofs_geometrical((V_vp.sub(0).sub(1), V_vy), boundary_topandbase)
    zero_vy = df.fem.Function(V_vy)
    zero_vy.x.array[:] = 0.0
    bcs_s.append(df.fem.dirichletbc(zero_vy, dofs_vy_topbase, V_vp.sub(0).sub(1)))

    # Define the location of the lower left corner of the domain and find the pressure DOF there
    def corner_lowerleft(x):
        return np.logical_and(np.isclose(x[0], 0), np.isclose(x[1], 0))
    dofs_p_lowerleft = df.fem.locate_dofs_geometrical((V_vp.sub(1), V_p), corner_lowerleft)
    # Specify the arbitrary pressure value and define a Dirichlet boundary condition
    zero_p = df.fem.Function(V_p)
    zero_p.x.array[:] = 0.0
    bcs_s.append(df.fem.dirichletbc(zero_p, dofs_p_lowerleft, V_vp.sub(1)))

    # Define extra constants
    Ra_c = df.fem.Constant(mesh, df.default_scalar_type(Ra))
    gravity = df.fem.Constant(mesh, df.default_scalar_type((0.0,-1.0)))
    eta = 1
    if b is not None: eta = ufl.exp(-np.log(b)*T)

    # Define the integrals to be assembled into the stiffness matrix for the Stokes system
    Ks = ufl.inner(ufl.sym(ufl.grad(v_t)), 2*eta*ufl.sym(ufl.grad(v_a)))*ufl.dx
    Gs = -ufl.div(v_t)*p_a*ufl.dx
    Ds = -p_t*ufl.div(v_a)*ufl.dx
    Ss = Ks + Gs + Ds

    # Define the integral to the assembled into the forcing vector for the Stokes system
    fs = -ufl.inner(v_t, gravity)*Ra_c*T*ufl.dx

    # Set up the Stokes problem (given the boundary conditions, bcs)
    problem_s = df.fem.petsc.LinearProblem(Ss, fs, bcs=bcs_s, u=vp_i, \
                                           petsc_options={"ksp_type": "preonly", \
                                                          "pc_type": "lu", \
                                                          "pc_factor_mat_solver_type": "mumps"})

    # Declare a list of boundary conditions for the temperature problem
    bcs_T = []

    # Define the location of the top boundary and find the temperature DOFs
    def boundary_top(x):
        return np.isclose(x[1], 1)
    dofs_T_top = df.fem.locate_dofs_geometrical(V_T, boundary_top)
    zero_T = df.fem.Function(V_T)
    zero_T.x.array[:] = 0.0
    bcs_T.append(df.fem.dirichletbc(zero_T, dofs_T_top))
    
    # Define the location of the base boundary and find the temperature DOFs
    def boundary_base(x):
        return np.isclose(x[1], 0)
    dofs_T_base = df.fem.locate_dofs_geometrical(V_T, boundary_base)
    one_T = df.fem.Function(V_T)
    one_T.x.array[:] = 1.0
    bcs_T.append(df.fem.dirichletbc(one_T, dofs_T_base))

    # Define the integrals to be assembled into the stiffness matrix for the temperature system
    ST = (T_t*ufl.inner(v, ufl.grad(T_a)) + ufl.inner(ufl.grad(T_t), ufl.grad(T_a)))*ufl.dx

    # Define the integral to the assembled into the forcing vector for the temperature system
    # which in this case is just zero
    fT = zero_T*T_t*ufl.dx
    
    problem_T = df.fem.petsc.LinearProblem(ST, fT, bcs=bcs_T, u=T_i, \
                                           petsc_options={"ksp_type": "preonly", \
                                                          "pc_type": "lu", \
                                                          "pc_factor_mat_solver_type": "mumps"})

    # Define the non-linear residual for the Stokes problem
    rs = ufl.action(Ss, vp) - fs
    # Define the non-linear residual for the temperature problem
    rT = ufl.action(ST, T) - fT

    def calculate_residual():
        """
        Return the total residual of the problem
        """
        rs_vec = df.fem.assemble_vector(df.fem.form(rs))
        df.fem.set_bc(rs_vec.array, bcs_s, scale=0.0)
        rT_vec = df.fem.assemble_vector(df.fem.form(rT))
        df.fem.set_bc(rT_vec.array, bcs_T, scale=0.0)
        r = np.sqrt(rs_vec.petsc_vec.norm()**2 + \
                    rT_vec.petsc_vec.norm()**2)
        return r

    # calculate the initial residual
    r = calculate_residual()
    r0 = r
    rrel = r/r0 # 1
    if mesh.comm.rank == 0:
        print("{:<11} {:<12} {:<17}".format('Iteration','Residual','Relative Residual'))
        print("-"*42)

    # Iterate until the residual converges (hopefully)
    it = 0
    if mesh.comm.rank == 0: print("{:<11} {:<12.6g} {:<12.6g}".format(it, r, rrel,))
    while rrel > rtol and r > atol:
        if it > maxits: break
        vp_i = problem_s.solve()
        vp.x.array[:] = (1-alpha)*vp.x.array + alpha*vp_i.x.array
        T_i = problem_T.solve()
        T.x.array[:] = (1-alpha)*T.x.array + alpha*T_i.x.array
        # calculate a new residual
        r = calculate_residual()
        rrel = r/r0
        it += 1
        if mesh.comm.rank == 0: print("{:<11} {:<12.6g} {:<12.6g}".format(it, r, rrel,))

    # Check for convergence failures
    if it > maxits:
        raise Exception("Nonlinear iteration failed to converge after {} iterations (maxits = {}), r = {} (atol = {}), rrel = {} (rtol = {}).".format(it, \
                                                                                                                                                      maxits, \
                                                                                                                                                      r, \
                                                                                                                                                      rtol, \
                                                                                                                                                      rrel, \
                                                                                                                                                      rtol,))

    # Return the subfunctions for velocity and pressure and the function for temperature
    return v.collapse(), p.collapse(), T
    


# In[ ]:


def blankenbach_diagnostics(v, T):
    mesh = T.function_space.mesh
    
    fdim = mesh.topology.dim - 1
    top_facets = df.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[1], 1))
    facet_tags = df.mesh.meshtags(mesh, fdim, np.sort(top_facets), np.full_like(top_facets, 1))
    ds = ufl.Measure('ds', domain=mesh, subdomain_data=facet_tags)

    Nu = -df.fem.assemble_scalar(df.fem.form(T.dx(1)*ds(1)))
    Nu = mesh.comm.allreduce(Nu, op=MPI.SUM)

    vrms = df.fem.assemble_scalar(df.fem.form((ufl.inner(v, v)*ufl.dx)))
    vrms = mesh.comm.allreduce(vrms, op=MPI.SUM)**0.5

    return Nu, vrms


# In[ ]:


# code for Stokes Equation
ne = 40
p = 1
# Case 1a
Ra = 1.e4
v_1a, p_1a, T_1a = solve_blankenbach(Ra, ne, p=p)
T_1a.name = 'Temperature'
print('Nu = {}, vrms = {}'.format(*blankenbach_diagnostics(v_1a, T_1a)))


# In[ ]:


# visualize
plotter_1a = utils.plot_scalar(T_1a, cmap='coolwarm')
utils.plot_vector_glyphs(v_1a, plotter=plotter_1a, color='k', factor=0.0005)
utils.plot_show(plotter_1a)


# In[ ]:


# code for Stokes Equation
ne = 40
p = 1
# Case 1b
Ra = 1.e5
v_1b, p_1b, T_1b = solve_blankenbach(Ra, ne, p=p)
T_1b.name = 'Temperature'
print('Nu = {}, vrms = {}'.format(*blankenbach_diagnostics(v_1b, T_1b)))


# In[ ]:


# visualize
plotter_1b = utils.plot_scalar(T_1b, cmap='coolwarm')
utils.plot_vector_glyphs(v_1b, plotter=plotter_1b, color='k', factor=0.00005)
utils.plot_show(plotter_1b)


# In[ ]:


# code for Stokes Equation
ne = 60
p = 1
# Case 1c
Ra = 1.e6
v_1c, p_1c, T_1c = solve_blankenbach(Ra, ne, p=p)
T_1c.name = 'Temperature'
print('Nu = {}, vrms = {}'.format(*blankenbach_diagnostics(v_1c, T_1c)))


# In[ ]:


# visualize
plotter_1c = utils.plot_scalar(T_1c, cmap='coolwarm')
utils.plot_vector_glyphs(v_1c, plotter=plotter_1c, color='k', factor=0.00001)
utils.plot_show(plotter_1c)


# In[ ]:


# code for Stokes Equation
ne = 60
p = 1
# Case 2a
Ra = 1.e4
v_2a, p_2a, T_2a = solve_blankenbach(Ra, ne, p=p, b=1.e3)
T_2a.name = 'Temperature'
print('Nu = {}, vrms = {}'.format(*blankenbach_diagnostics(v_2a, T_2a)))


# In[ ]:


# visualize
plotter_2a = utils.plot_scalar(T_2a, cmap='coolwarm')
utils.plot_vector_glyphs(v_2a, plotter=plotter_2a, color='k', factor=0.00002)
utils.plot_show(plotter_2a)


# ## Finish up

# Convert this notebook to a python script (making sure to save first)

# In[ ]:


if __name__ == "__main__" and "__file__" not in globals():
    from ipylab import JupyterFrontEnd
    app = JupyterFrontEnd()
    app.commands.execute('docmanager:save')
    get_ipython().system('jupyter nbconvert --NbConvertApp.export_format=script --ClearOutputPreprocessor.enabled=True blankenbach.ipynb')


# In[ ]:




