#!/usr/bin/env python
# coding: utf-8

# # Subduction Zone Setup

# Author: Cian Wilson

# ## Implementation

# Our implementation will follow a similar workflow to that seen repeatedly in the background examples section.
# 
# 1. we will describe the subduction zone geometry and tesselate it intro non-overlapping triangles to create a **mesh**
# 2. we will declare **function spaces** for the temperature, wedge velocity and pressure, and slab velocity and pressure
# 3. using these function space we will declare **trial** and **test functions**
# 4. we will define Dirichlet boundary conditions at the boundaries as described in the introduction
# 5. we will use the crustal heat sources in the right hand side forcing function for the temperature
# 6. we will describe **discrete weak forms** for temperature and each of the coupled velocity-pressure systems that will be used to assemble the matrices (and vectors) to be solved
# 7. we will solve the matrix problems using a linear algebra back-end repeatedly in a Picard iteration to find the root of the residuals and return the non-linear solution
# 
# The only difference in a subduction zone problem is that each of these steps is more complicated than in the earlier examples.  Here we split steps 1-7 up across three notebooks.  In the first we implement a function to describe the slab surface using a spline.  The remaining details of the geometry are constructed in a function defined in the next notebook.  Finally we implement a python class to describe the remaining steps, 2-7, of the problem.

# ### Geometry
# 
# Throughout our implementation, in the following notebooks, we will demonstrate its functionality using the simplified geometry previously laid out and repeated below in Figure 1. However our implementation will be applicable to a broader range of geometries and setups.
# 
# ![Figure 8a of Wilson & van Keken, 2023](images/benchmarkgeometry.png)
# *Figure 1: Geometry and coefficients for a simplified 2D subduction zone model. All coefficients and parameters are nondimensional. The decoupling point is indicated by the star.*

# ### Parameters
# 
# We also recall the default parameters repeated below in Table 1.
# 
# \begin{array}{llll}
# \hline
# \text{Quantity} & \text{Symbol} & \text{Nominal value} & \text{Nondimensional value} \\
# \hline
# \text{Reference temperature scale} &  T_0 & 1~\text{K}=1^\circ \text{C} & - \\
# \text{Surface temperature} & T^*_s & 273~\text{K}=0^\circ \text{C} &  T_s=0\\
# \text{Mantle temperature} & T^*_m & 1623~\text{K}=1350^\circ \text{C}  & T_m=1350\\
# \text{Surface heat flow}^\text{c} & q^*_s & ^{\S} \text{W/m}^2  & q_s^{\S}\\
# \text{Reference density} & \rho_0 & 3300~\text{kg/m}^3 & - \\
# \text{Crustal density}^\text{c} & \rho^*_c & 2750~\text{kg/m}^3 & \rho_c=0.833333\\
# \text{Mantle density} & \rho^*_m & 3300~\text{kg/m}^3 & \rho_m=1 \\
# \text{Reference thermal conductivity} & k_0 & 3.1 ~\text{W/(m~K)} & - \\
# \text{Crustal thermal conductivity}^\text{c} & k^*_c & 2.5 ~\text{W/(m~K)} & k_c=0.8064516\\
# \text{Mantle thermal conductivity} & k^*_m & 3.1 ~\text{W/(m~K)} & k_m=1 \\
# \text{Volumetric heat production (upper crust)}^\text{c} & H^*_1 & 1.3 \mu \text{W/m}^3 & H_1=0.419354\\
# \text{Volumetric heat production (lower crust)}^\text{c} & H_2^* & 0.27 \mu \text{W/m}^3 & H_2=0.087097\\
# \text{Age of overriding crust}^\text{o} & A_c^* & ^{\S}~\text{Myr} & A_c^{\S} \\
# \text{Age of subduction}^\text{t} & A_s^* & ^{\S}~\text{Myr }& A_s^{\S} \\
# \text{Age of subducting slab} & A^* & ^{\S}~\text{Myr} & A^{\S} \\
# \text{Reference length scale} & h_0 & 1~\text{km} & - \\
# \text{Depth of base of upper crust}^\text{c} & z_1^* & 15~\text{km} & z_1=15 \\
# \text{Depth of base of lower crust (Moho)} & z_2^* & ^{\S}~\text{km} & z_2^{\S} \\
# \text{Trench depth} & z_\text{trench}^* & ^{\S}~\text{km} & z_\text{trench}^{\S} \\
# \text{Position of the coast line} & x_\text{coast}^* & ^{\S}~\text{km} & x_\text{coast}^{\S} \\
# \text{Wedge inflow/outflow transition depth} & z_\text{io}^* & ^{\S}~\text{km} & z_\text{io}^{\S} \\
# \text{Depth of domain} & D^* & ^{\S}~\text{km} & D^{\S} \\
# \text{Width of domain} & L^* & ^{\S}~\text{km} & L^{\S} \\
# \text{Depth of change from decoupling to coupling} & d_c^* & 80~\text{km} & d_c=80 \\
# \text{Reference heat capacity} & {c_p}_0 & 1250~\text{J/(kg~K)} & - \\
# \text{Reference thermal diffusivity} & \kappa_0 & 0.7515\times10^{\text{-6}}~\text{m}^2\text{/s} & - \\
# \text{Activation energy} & E & 540~\text{kJ/mol} & -\\
# \text{Powerlaw exponent} & n & 3.5 & -\\
# \text{Pre-exponential constant} & A^*_\eta & 28968.6~\text{Pa~s}^{1/n} & - \\
# \text{Reference viscosity scale} & \eta_0 & 10^{\textrm{21}}~\text{Pa~s} & -\\
# \text{Viscosity cap} & \eta^*_\text{max} & 10^{\textrm{25}}~\text{Pa~s} & - \\
# \text{Gas constant} & R^* & 8.3145~\text{J/(mol~K)} & - \\
# \text{Derived velocity scale} & {v}_0& 23.716014~\text{mm/yr} & - \\
# \text{Convergence velocity} & V_s^* & ^{\S}~\text{mm/yr} & V_s^{\S} \\
# \end{array}
# \begin{array}{rl}
# ^\text{c} & \text{ocean-continent subduction only} \\
# ^\text{o} & \text{ocean-ocean subduction only} \\
# ^\text{t} & \text{time-dependent simulations only} \\
# ^\S       & \text{varies between models} \\
# \end{array}
# *Table 1: Nomenclature and reference values*
# 
# Most of these are available for us to use through a file in `../data/default_params.json`

# In[ ]:


import os
params_filename = os.path.join(os.pardir, "data", "default_params.json")


# Loading this file

# In[ ]:


import json
with open(params_filename, "r") as fp:
    default_params = json.load(fp)


# This contains default parameters required to define the geometry. Keys ending in `_sid` and `_rid` are surface and region IDs respectively that we use to identify boundaries and regions of the mesh (these are unlikely to need to be changed). *_res_fact are resolution factors scaled by a factor to set the resolution at various points in the mesh. Finally, those ending in _depth are depths (in km) of various important points along the slab surface or boundaries (as defined in Table 1).

# In[ ]:


from mpi4py import MPI
if __name__ == "__main__":
    if MPI.COMM_WORLD.rank == 0:
        print("{:<35} {:<10}".format('Key','Value'))
        print("-"*45)
        for k, v in default_params.items():
            print("{:<35} {:<10}".format(k, v))


# We will additionally use parameters from the benchmark proposed in [Wilson & van Keken, PEPS, 2023](http://dx.doi.org/10.1186/s40645-023-00588-6) as defined in Table 2 below.
# 
# \begin{array}{llllllllllll}
# \text{case} & \text{type} & \eta & q_s^* & A^* & z_2 & z_\text{io} & z_\text{trench} & x_\text{coast} & D & L & V_s^* \\
#  & & & \text{(W/m}^2\text{)} & \text{(Myr)} &       &         & & & &    & (mm/yr)      \\   
# \hline
# 1 & \text{c} & 1               & 0.065 & 100 & 40 & 139 & 0 & 0 & 200 & 400 & 100  \\
# 2 & \text{c} & \eta^*/\eta_0 & 0.065 &  100 & 40 & 154  & 0 & 0& 200 & 400 & 100 \\
# \end{array}
# \begin{array}{rl}
# \text{c} & \text{ocean-continent subduction} \\
# \end{array}
# *Table 2: Benchmark parameter values*

# Since these benchmark parameters are so few we will simply enter them as needed.  For the global suite all parameters marked as varying between models in Table 1 will change between cases.  An additional database of these parameters is provided in `../data/all_sz.json`, which we also load here

# In[ ]:


allsz_filename = os.path.join(os.pardir, "data", "all_sz.json")
with open(allsz_filename, "r") as fp:
    allsz_params = json.load(fp)


# The `allsz_params` dictionary contains parameters for all 56 subduction zones organized by name.

# In[ ]:


if __name__ == "__main__":
    if MPI.COMM_WORLD.rank == 0:
        print("{}".format('Name'))
        print("-"*30)
        for k in allsz_params.keys():
            print("{}".format(k,))


# Taking two examples (one continental-oceanic, "01_Alaska_Peninsula", and one oceanic-oceanic, "19_N_Antilles") we can examine the contents of `allsz_params`.

# In[ ]:


if __name__ == "__main__":
    names = ['01_Alaska_Peninsula', '19_N_Antilles']
    if MPI.COMM_WORLD.rank == 0:
        for name in names:
            print("{}:".format(name))
            print("{:<35} {:<10}".format('Key','Value'))
            print("-"*100)
            for k, v in allsz_params[name].items():
                if v is not None: print("{:<35} {}".format(k, v))
            print("="*100)


# ## Finish up

# Convert this notebook to a python script (making sure to save first) so that we can access it in subsequent notebooks.

# In[ ]:


if __name__ == "__main__" and "__file__" not in globals():
    from ipylab import JupyterFrontEnd
    app = JupyterFrontEnd()
    app.commands.execute('docmanager:save')
    get_ipython().system('jupyter nbconvert --NbConvertApp.export_format=script --ClearOutputPreprocessor.enabled=True sz_base.ipynb')


# In[ ]:




