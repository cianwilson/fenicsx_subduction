#!/usr/bin/env python
# coding: utf-8

# # Subduction Zone Steady State Suite
# 
# Authors: Kidus Teshome, Cian Wilson

# ## 01 Alaska Peninsula

# ### Preamble

# In[ ]:


import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
sys.path.append(os.path.join(basedir, os.path.pardir))


# Start by loading everything we need from `sz_problem` and also set our default plotting preferences.

# In[ ]:


from sz_suite_ss import *
import pyvista as pv
if __name__ == "__main__" and "__file__" in globals():
    pv.OFF_SCREEN = True
if __name__ == "__main__":
    output_folder = pathlib.Path("output")
    output_folder.mkdir(exist_ok=True, parents=True)


# ### Steady state solution

# In[ ]:


name = "01_Alaska_Peninsula"
resscale = 2.0
sz = solve_steadystate_sz(name, resscale)


# In[ ]:


plotter = utils.plot_scalar(sz.T_i, scale=sz.T0, gather=True, cmap='coolwarm')
utils.plot_vector_glyphs(sz.vw_i, plotter=plotter, gather=True, factor=0.1, color='k', scale=utils.mps_to_mmpyr(sz.v0))
utils.plot_vector_glyphs(sz.vs_i, plotter=plotter, gather=True, factor=0.1, color='k', scale=utils.mps_to_mmpyr(sz.v0))
utils.plot_show(plotter)
utils.plot_save(plotter, "sz_suite_ss_{}_solution_{:.2f}.png".format("01", resscale))


# In[ ]:


filename = output_folder / "sz_suite_ss_{}_solution_{:.2f}.png".format("01", resscale)
with df.io.VTXWriter(sz.mesh.comm, filename.with_suffix(".bp"), [sz.T_i, sz.vs_i, sz.vw_i]) as vtx:
    vtx.write(0.0)


# ## Finish up

# Convert this notebook to a python script (making sure to save first)

# In[ ]:


if __name__ == "__main__" and "__file__" not in globals():
    from ipylab import JupyterFrontEnd
    app = JupyterFrontEnd()
    app.commands.execute('docmanager:save')
    get_ipython().system('jupyter nbconvert --NbConvertApp.export_format=script --ClearOutputPreprocessor.enabled=True 01_Alaska_Peninsula.ipynb')


# In[ ]:




