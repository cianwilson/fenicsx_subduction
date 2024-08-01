#!/usr/bin/env python
# coding: utf-8

# # 39 N Philippines
# 
# Authors: Kidus Teshome, Cian Wilson

# ## Steady state implementation

# ### Preamble

# Set some path information.

# In[ ]:


import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
sys.path.append(os.path.join(basedir, os.path.pardir))
sys.path.append(os.path.join(basedir, os.path.pardir, os.path.pardir, 'python'))


# Loading everything we need from `sz_problem` and also set our default plotting and output preferences.

# In[ ]:


import utils
from sz_base import allsz_params
from sz_slab import create_slab, plot_slab
from sz_geometry import create_sz_geometry
from sz_problem import SubductionProblem
import dolfinx as df
import pyvista as pv
if __name__ == "__main__" and "__file__" in globals():
    pv.OFF_SCREEN = True
import pathlib
output_folder = pathlib.Path(os.path.join(basedir, "output"))
output_folder.mkdir(exist_ok=True, parents=True)


# ### Parameters

# We first select the name and resolution scale, `resscale` of the model.
# 
# ```{admonition} Resolution
# By default the resolution is low to allow for a quick runtime and smaller website size.  If sufficient computational resources are available set a lower `resscale` to get higher resolutions and results with sufficient accuracy.
# ```
# 

# In[ ]:


name = "39_N_Philippines"
resscale = 3.0


# Then load the remaining parameters from the global suite.

# In[ ]:


szdict = allsz_params[name]
print("{}:".format(name))
print("{:<20} {:<10}".format('Key','Value'))
print("-"*85)
for k, v in allsz_params[name].items():
    if v is not None: print("{:<20} {}".format(k, v))


# Any of these can be modified in the dictionary.
# 
# Several additional parameters can be modified, for details see the documentation for the `SubductionProblem` class.

# In[ ]:


if __name__ == "__main__" and "__file__" not in globals():
    get_ipython().run_line_magic('pinfo', 'SubductionProblem')


# The `if __name__ == "__main__" and "__file__" not in globals():` logic above is only necessary to make sure that this only runs in the Jupyter notebook version of this code and not the python version.  It is not generally necessary when getting the docstring of a function or class in Jupyter.

# ### Setup

# Setup a slab.

# In[ ]:


slab = create_slab(szdict['xs'], szdict['ys'], resscale, szdict['lc_depth'])
_ = plot_slab(slab)


# Create the subduction zome geometry around the slab.

# In[ ]:


geom = create_sz_geometry(slab, resscale, szdict['sztype'], szdict['io_depth'], szdict['extra_width'], 
                             szdict['coast_distance'], szdict['lc_depth'], szdict['uc_depth'])
_ = geom.plot()


# Finally, declare the `SubductionZone` problem class using the dictionary of parameters.

# In[ ]:


sz = SubductionProblem(geom, **szdict)


# ### Solve

# Solve using a dislocation creep rheology and assuming a steady state.

# In[ ]:


sz.solve_steadystate_dislocationcreep()


# ### Save

# Plot the solution.

# In[ ]:


plotter = utils.plot_scalar(sz.T_i, scale=sz.T0, gather=True, cmap='coolwarm')
utils.plot_vector_glyphs(sz.vw_i, plotter=plotter, gather=True, factor=0.1, color='k', scale=utils.mps_to_mmpyr(sz.v0))
utils.plot_vector_glyphs(sz.vs_i, plotter=plotter, gather=True, factor=0.1, color='k', scale=utils.mps_to_mmpyr(sz.v0))
utils.plot_show(plotter)
utils.plot_save(plotter, output_folder / "{}_ss_solution_resscale_{:.2f}.png".format(name, resscale))


# Save it to disk so that it can be examined with other visualization software (e.g. [Paraview](https://www.paraview.org/)).

# In[ ]:


filename = output_folder / "{}_ss_solution_resscale_{:.2f}.bp".format(name, resscale)
with df.io.VTXWriter(sz.mesh.comm, filename, [sz.T_i, sz.vs_i, sz.vw_i]) as vtx:
    vtx.write(0.0)
# zip the .bp folder so that it can be downloaded from Jupyter lab
if __name__ == "__main__" and "__file__" not in globals():
    zipfilename = filename.with_suffix(".zip")
    get_ipython().system('zip -r $zipfilename $filename')


# ## Finish up

# Convert this notebook to a python script (making sure to save first)

# In[ ]:


if __name__ == "__main__" and "__file__" not in globals():
    from ipylab import JupyterFrontEnd
    app = JupyterFrontEnd()
    app.commands.execute('docmanager:save')
    get_ipython().system('jupyter nbconvert --NbConvertApp.export_format=script --ClearOutputPreprocessor.enabled=True 39_N_Philippines.ipynb')


# In[ ]:




