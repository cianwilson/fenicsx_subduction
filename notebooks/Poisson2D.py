#!/usr/bin/env python
# coding: utf-8

# # Poisson Example 2D

# ## Description

# We can generalize (and formalize) the description of the Poisson equation 
# using the steady-state heat diffusion equation in multiple dimensions
# \begin{align}
# -\nabla \cdot\left( k \nabla T \right) &= H && \text{in }\Omega
# \end{align}
# $T$ is the temperature solution we are seeking, $k$ is the thermal conductivity and 
# $H$ is a heat source, and $\Omega$ is the domain with boundary $\partial\Omega$.  If $k$ is constant in space we can simplify to
# \begin{align}
# -\nabla^2 T &= h && \text{in }\Omega
# \end{align}
# where $h = \frac{H}{k}$.  

# ### Boundary conditions

# We supplement the Poisson equation with some combination of the boundary conditions 
# \begin{align}
# T &= g_D && \text{on } \partial\Omega_D \subset \partial\Omega \\
# \nabla T\cdot\hat{\vec{n}} &= g_N && \text{on } \partial\Omega_N \subset \partial\Omega \\
# aT + \nabla T\cdot\hat{\vec{n}} &= g_R && \text{on } \partial\Omega_R \subset \partial\Omega 
# \end{align}
# where $\partial\Omega_D$, $\partial\Omega_N$ and $\partial\Omega_R$ are
# segments of the domain boundary that do not overlap ($\partial\Omega_D \bigcap \partial\Omega_N =\emptyset$, $\partial\Omega_D \bigcap \partial\Omega_R =\emptyset$, $\partial\Omega_N \bigcap \partial\Omega_R =\emptyset$) and that together span the entire boundary ($\partial\Omega_D \bigcup \partial\Omega_N \bigcup \partial\Omega_R = \partial\Omega$).  The unit outward-pointing normal to the boundary $\partial\Omega$ is denoted by $\hat{\vec{n}}$ and $g_D = g_D(\vec{x}, t)$, $g_N = g_N(\vec{x}, t)$ and $g_R = g_R(\vec{x}, t)$ are known functions of space and time.  
# 
# The first boundary condition is known as a Dirichlet boundary condition and specifies the value of the solution on $\partial\Omega_D$. The second is a Neumann boundary condition and specifies the value of the flux through $\partial\Omega_N$. Finally, the third is a Robin boundary condition, which describes a linear combination of the flux and the solution on $\partial\Omega_R$.

# ### Weak form

# The first step in the finite element discretization is to transform the equation into its **weak form**.  This requires multiplying the equation by a test function,  $T_t$,  and integrating over the domain $\Omega$
# \begin{equation}
# -\int_\Omega T_t \nabla^2 T ~dx = \int_\Omega T_t h ~dx
# \end{equation}
# After integrating the left-hand side by parts
# \begin{equation}
# \int_\Omega \nabla T_t \cdot \nabla T ~dx - \int_{\partial\Omega} T_t \nabla T\cdot\hat{\vec{n}}~ds = \int_\Omega T_t h ~dx
# \end{equation}
# we can see that we have reduced the continuity requirements on $T$ by only requiring its first derivative to be bounded across $\Omega$. Integrating by parts also allows Neumann and Robin boundary conditions to be imposed "naturally" through the second integral on the left-hand side since this directly incorporates the flux components across the boundary.  In this formulation, Dirichlet conditions cannot be imposed weakly and are referred to as essential boundary conditions,  that are required of the solution but do not arise naturally in the weak form.  The weak form therefore becomes: find $T$ such that $T$=$g_D$ on $\partial\Omega_D$ and
# \begin{equation}
# \int_\Omega \nabla T_t \cdot \nabla T ~dx - \int_{\partial\Omega_N} T_t g_N ~ds - \int_{\partial\Omega_R} T_t \left(g_R - aT\right)~ds = \int_\Omega T_t h ~dx
# \end{equation}
# for all $T_t$ such that $T_t = 0$ on $\partial\Omega_D$.  

# ### Discretization

# The weak and strong forms of the problem are equivalent so long as the solution is sufficiently smooth.  We make our first approximation by, instead of seeking $T$ such that $T = g_D$ on $\partial\Omega_D$, seeking the discrete trial function $\tilde{T}$ such that $\tilde{T} = g_D$ on $\partial\Omega_D$ where
# \begin{equation}
# T \approx \tilde{T} = \sum_j \phi_j T_j
# \end{equation}
# for all test functions $\tilde{T}_t$ where
# \begin{equation}
# T_t \approx \tilde{T}_t = \sum_i \phi_i T_{ti}
# \end{equation}
# noting again that $\tilde{T}_t = 0$ on $\partial\Omega_D$.  
# $\phi_j$ are the finite element shape functions. Assuming these are continuous across elements of the mesh, $\tilde{T}$ and $\tilde{T}_t$ can be substituted into the weak form to yield
# \begin{multline}
# \sum_i\sum_j T_{ti}T_j\sum_k  \int_{e_k} \nabla \phi_i \cdot \nabla \phi_j ~dx 
#  + \sum_i\sum_j T_{ti}T_j \sum_k \int_{\partial e_k \cap {\partial\Omega_R}} \phi_i a\phi_j ~ds
# \\- \sum_i T_{ti} \sum_k \int_{\partial e_k \cap {\partial\Omega_N}} \phi_i g_N ~ds 
# - \sum_i T_{ti} \sum_k \int_{\partial e_k \cap {\partial\Omega_R}} \phi_i g_R 
# \\= \sum_i T_{ti} \sum_k \int_{e_k} \phi_i h ~dx
# \end{multline}
# where we are integrating over the whole domain by summing the integrals over all the elements  $e_k$ ($\int_\Omega dx$=$\sum_k\int_{e_k} dx$).  Note that in practice, because the shape functions are zero over most of the domain, only element integrals with non-zero values need be included in the summation.  The element boundaries, $\partial e_k$, are only of interest (due to the assumed continuity of the shape functions between the elements) if they either intersect with $\partial\Omega_N$, $\partial e_k \cap {\partial\Omega_N}$, or $\partial\Omega_R$, $\partial e_k \cap {\partial\Omega_R}$.  Since the solution of the now discretized weak form should be valid for all $\tilde{T}_t$ we can drop $T_{ti}$
# \begin{multline}
# \sum_jT_j\sum_k  \int_{e_k} \nabla \phi_i \cdot \nabla \phi_j ~dx 
#  + \sum_jT_j\sum_k \int_{\partial e_k \cap {\partial\Omega_R}} \phi_i a \phi_j ~ds
# \\- \sum_k \int_{\partial e_k \cap {\partial\Omega_N}} \phi_i g_N ~ds 
# - \sum_k \int_{\partial e_k \cap {\partial\Omega_R}} \phi_i g_R~ds 
# = \sum_k \int_{e_k} \phi_i h ~dx
# \end{multline}
# This represents a matrix-vector system of the form
# \begin{equation}
# {\bf S} {\bf u} = {\bf f}
# \end{equation}
# with
# \begin{align}
# {\bf S} &= S_{ij} = \sum_k\int_{e_k} \nabla \phi_i \cdot \nabla \phi_j ~dx + \sum_k \int_{\partial e_k \cap {\partial\Omega_R}} \phi_i a\phi_j ~ds  \\
# {\bf f} &= f_i = \sum_k \int_{e_k} \phi_i h ~dx + \sum_k \int_{\partial e_k \cap {\partial\Omega_N}} \phi_i g_N ~ds 
# + \sum_k \int_{\partial e_k \cap {\partial\Omega_R}} \phi_i g_R~ds \\
# {\bf u} &= {\bf T} = T_j 
# \end{align}
# 
# The compact support of the shape functions $\phi_{(i,j)}$, which limits their nonzero values to the elements immediately neighboring DOF $i$ or $j$, means that the integrals in can be evaluated efficiently by only considering shape functions associated with an element $e_k$.  It also means that the resulting matrix ${\bf S}$ is sparse, with most entries being zero.

# ### A specific example

# In this case we use a manufactured solution (that is, one that is not necessarily an example of a solution to a PDE representing a naturally occurring physical problem) where we take a known analytical solution $T(x,y)$ and substitute this into the original equation to find $h$, then use this as the right-hand side in our numerical test. We choose $T(x,y)$=$\exp\left(x+\tfrac{y}{2}\right)$, which is the solution to
# \begin{equation}
# - \nabla^2 T = -\tfrac{5}{4} \exp \left( x+\tfrac{y}{2} \right)
# \end{equation}
# Solving the Poisson equation numerically in a unit square, $\Omega=[0,1]\times[0,1]$, for the approximate solution $\tilde{T} \approx T$, we impose the boundary conditions
# \begin{align}
#   \tilde{T} &= \exp\left(x+\tfrac{y}{2}\right) && \text{on } \partial\Omega \text{ where } x=0 \text{ or } y=0 \\
#   \nabla \tilde{T}\cdot \hat{\vec{n}} &= \exp\left(x + \tfrac{y}{2}\right) && \text{on } \partial\Omega \text{ where } x=1  \\
#   \nabla \tilde{T}\cdot \hat{\vec{n}} &= \tfrac{1}{2}\exp\left(x + \tfrac{y}{2}\right) && \text{on } \partial\Omega \text{ where } y=1
#  \end{align}
# representing an essential Dirichlet condition on the value of $\tilde{T}$ and natural Neumann conditions on $\nabla\tilde{T}$.

# ## Implementation

# This example was presented by [Wilson & van Keken, 2023](http://dx.doi.org/10.1186/s40645-023-00588-6) using FEniCS v2019.1.0 and [TerraFERMA](terraferma.github.io), a GUI-based model building framework that also uses FEniCS v2019.1.0.  Here we reproduce these results using the latest version of FEniCS, FEniCSx.

# ### Preamble

# We start by loading all the modules we will require.

# In[ ]:


from mpi4py import MPI
import dolfinx as df
import dolfinx.fem.petsc
import numpy as np
import ufl
import matplotlib.pyplot as pl
import pathlib
import sys, os
sys.path.append(os.path.join(os.path.pardir, 'python'))
import utils
import pyvista as pv
if __name__ == "__main__" and "__file__" in globals():
    pv.OFF_SCREEN = True


# ### Solution

# We then declare a python function `solve_poisson_2d` that contains a complete description of the discrete Poisson equation problem.
# 
# This function follows much the same flow as described above:
# 1. we describe the unit square domain $\Omega = [0,1]\times[0,1]$ and discretize it into $2\times$`ne`$\times$`ne` triangular elements or cells to make a `mesh`
# 2. we declare the **function space**, `V`, to use Lagrange polynomials of degree `p`
# 3. using this function space we declare trial, `T_a`, and test, `T_t`, functions
# 4. we define the Dirichlet boundary condition, `bc` at $x=0$ and $y=0$, setting the desired value there to the known exact solution
# 5. we define a finite element `Function`, `gN`, containing the values of $\nabla \tilde{T}$ on the Neumann boundaries $x=1$ and $y=1$ (note that this will be used in the weak form rather than as a boundary condition object)
# 6. we define the right hand side forcing function $h$, `h`
# 7. we describe the **discrete weak forms**, `S` and `f`, that will be used to assemble the matrix $\mathbf{S}$ and vector $\mathbf{f}$
# 8. we solve the matrix problem using a linear algebra back-end and return the solution
# 
# For a more detailed description of solving the Poisson equation using FEniCSx, see [the FEniCSx tutorial](https://jsdokken.com/dolfinx-tutorial/chapter1/fundamentals.html).

# In[ ]:


def solve_poisson_2d(ne, p=1):
    """
    A python function to solve a two-dimensional Poisson problem
    on a unit square domain.
    Parameters:
    * ne - number of elements in each dimension
    * p  - polynomial order of the solution function space
    """
    # Describe the domain (a unit square)
    # and also the tessellation of that domain into ne 
    # equally spaced squares in each dimension which are
    # subdivided into two triangular elements each
    mesh = df.mesh.create_unit_square(MPI.COMM_WORLD, ne, ne)

    # Define the solution function space using Lagrange polynomials
    # of order p
    V = df.fem.functionspace(mesh, ("Lagrange", p))

    # Define the trial and test functions on the same function space (V)
    T_a = ufl.TrialFunction(V)
    T_t = ufl.TestFunction(V)

    # Define the location of the boundary condition, x=0 and y=0
    def boundary(x):
        return np.logical_or(np.isclose(x[0], 0), np.isclose(x[1], 0))
    boundary_dofs = df.fem.locate_dofs_geometrical(V, boundary)
    # Specify the value and define a Dirichlet boundary condition (bc)
    gD = df.fem.Function(V)
    gD.interpolate(lambda x: np.exp(x[0] + x[1]/2.))
    bc = df.fem.dirichletbc(gD, boundary_dofs)

    # Get the coordinates
    x = ufl.SpatialCoordinate(mesh)
    # Define the Neumann boundary condition function
    gN = ufl.as_vector((ufl.exp(x[0] + x[1]/2.), 0.5*ufl.exp(x[0] + x[1]/2.)))
    # Define the right hand side function, h
    h = -5./4.*ufl.exp(x[0] + x[1]/2.)

    # Get the unit vector normal to the facets
    n = ufl.FacetNormal(mesh)
    # Define the integral to be assembled into the stiffness matrix
    S = ufl.inner(ufl.grad(T_t), ufl.grad(T_a))*ufl.dx
    # Define the integral to be assembled into the forcing vector,
    # incorporating the Neumann boundary condition weakly
    f = T_t*h*ufl.dx + T_t*ufl.inner(gN, n)*ufl.ds

    # Compute the solution (given the boundary condition, bc)
    problem = df.fem.petsc.LinearProblem(S, f, bcs=[bc], \
                                         petsc_options={"ksp_type": "preonly", \
                                                        "pc_type": "lu"})
    T_i = problem.solve()

    return T_i


# We can now numerically solve the equations using, e.g. 4 elements and piecewise linear polynomials.

# In[ ]:


if __name__ == "__main__":
    T_P1 = solve_poisson_2d(4)
    T_P1.name = "T (P1)"


# And use a utility function to plot it.

# In[ ]:


if __name__ == "__main__":
    plotter_P1 = utils.plot_scalar(T_P1, gather=True)
    utils.plot_mesh(T_P1.function_space.mesh, plotter=plotter_P1, gather=True, show_edges=True, style="wireframe", color='k', line_width=2)
    utils.plot_scalar_values(T_P1, plotter=plotter_P1, gather=True, point_size=15, font_size=22, shape_color='w', text_color='k', bold=False)
    utils.plot_show(plotter_P1)
    utils.plot_save(plotter_P1, "2d_poisson_P1_solution.png")
    comm = T_P1.function_space.mesh.comm
    if comm.size > 1:
        # if we're running in parallel then save an image per process as well
        plotter_P1_p = utils.plot_scalar(T_P1)
        utils.plot_mesh(T_P1.function_space.mesh, plotter=plotter_P1_p, show_edges=True, style="wireframe", color='k', line_width=2)
        utils.plot_scalar_values(T_P1, plotter=plotter_P1_p, point_size=15, font_size=22, shape_color='w', text_color='k', bold=False)
        utils.plot_save(plotter_P1_p, "2d_poisson_P1_solution_p{:d}.png".format(comm.rank,))


# Similarly, we can solve the equation using quadratic elements (`p=2`).

# In[ ]:


if __name__ == "__main__":
    T_P2 = solve_poisson_2d(4, p=2)
    T_P2.name = "T (P2)"


# In[ ]:


if __name__ == "__main__":
    plotter_P2 = utils.plot_scalar(T_P2, gather=True)
    utils.plot_mesh(T_P2.function_space.mesh, plotter=plotter_P2, gather=True, show_edges=True, style="wireframe", color='k', line_width=2)
    utils.plot_scalar_values(T_P2, plotter=plotter_P2, gather=True, point_size=15, font_size=12, shape_color='w', text_color='k', bold=False)
    utils.plot_show(plotter_P2)
    utils.plot_save(plotter_P2, "2d_poisson_P2_solution.png")
    comm = T_P2.function_space.mesh.comm
    if comm.size > 1:
        # if we're running in parallel then save an image per process as well
        plotter_P2_p = utils.plot_scalar(T_P2)
        utils.plot_mesh(T_P2.function_space.mesh, plotter=plotter_P2_p, show_edges=True, style="wireframe", color='k', line_width=2)
        utils.plot_scalar_values(T_P2, plotter=plotter_P2_p, point_size=15, font_size=12, shape_color='w', text_color='k', bold=False)
        utils.plot_save(plotter_P2_p, "2d_poisson_P2_solution_p{:d}.png".format(comm.rank,))


# ## Themes and variations

# Given that we know the exact solution to this problem is $T(x,y)$=$\exp\left(x+\tfrac{y}{2}\right)$ write a python function to evaluate the error in our numerical solution.
# 
# Then loop over a variety of `ne`s and `p`s and check that the numerical solution converges with an increasing number of degrees of freedom.
# 
# Note that, aside from the analytic solution being different, this is very similar to the 1D case in `Poisson1D.ipynb`.

# ## Finish up

# Convert this notebook to a python script (making sure to save first)

# In[ ]:


if __name__ == "__main__" and "__file__" not in globals():
    from ipylab import JupyterFrontEnd
    app = JupyterFrontEnd()
    app.commands.execute('docmanager:save')
    get_ipython().system('jupyter nbconvert --NbConvertApp.export_format=script --ClearOutputPreprocessor.enabled=True Poisson2D.ipynb')


# In[ ]:




