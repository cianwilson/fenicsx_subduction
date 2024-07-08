#!/usr/bin/env python
# coding: utf-8

# # Subduction Zone Steady State Suite
# 
# Authors: Kidus Teshome, Cian Wilson

# ## Implementation

# ### Preamble

# Start by loading everything we need from `sz_problem` and also set our default plotting preferences.

# In[ ]:


from sz_problem import *
import pyvista as pv
if __name__ == "__main__" and "__file__" in globals():
    pv.OFF_SCREEN = True


# In[ ]:


def solve_steadystate_sz(name, resscale):
    szdict = allsz_params[name]
    slab = create_slab(szdict['xs'], szdict['ys'], resscale, szdict['lc_depth'])
    geom = create_sz_geometry(slab, resscale, szdict['sztype'], szdict['io_depth'], szdict['extra_width'], 
                                 szdict['coast_distance'], szdict['lc_depth'], szdict['uc_depth'])
    sz = SubductionProblem(geom, **szdict)
    
    sz.solve_steadystate_dislocationcreep()

    return sz


# ## Finish up

# Convert this notebook to a python script (making sure to save first)

# In[ ]:


if __name__ == "__main__" and "__file__" not in globals():
    from ipylab import JupyterFrontEnd
    app = JupyterFrontEnd()
    app.commands.execute('docmanager:save')
    get_ipython().system('jupyter nbconvert --NbConvertApp.export_format=script --ClearOutputPreprocessor.enabled=True sz_suite_ss.ipynb')


# In[ ]:




