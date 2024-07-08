#!/usr/bin/env python
# coding: utf-8

# # Subduction Zone Benchmark
# 
# Authors: Kidus Teshome, Cian Wilson

# ## Convergence testing

# ### Preamble

# Start by loading everything we need from `sz_problem` and also set our default plotting preferences.

# In[ ]:


from sz_problem import *
import pyvista as pv
if __name__ == "__main__" and "__file__" in globals():
    pv.OFF_SCREEN = True


# ### Benchmark case 1

# In[ ]:


def solve_benchmark_case1(resscale):
    xs = [0.0, 140.0, 240.0, 400.0]
    ys = [0.0, -70.0, -120.0, -200.0]
    lc_depth = 40
    uc_depth = 15
    coast_distance = 0
    extra_width = 0
    sztype = 'continental'
    io_depth_1 = 139
    A      = 100.0      # age of subducting slab (Myr)
    qs     = 0.065      # surface heat flux (W/m^2)
    Vs     = 100.0      # slab speed (mm/yr)

    slab = create_slab(xs, ys, resscale, lc_depth)
    geom = create_sz_geometry(slab, resscale, sztype, io_depth_1, extra_width, 
                               coast_distance, lc_depth, uc_depth)
    sz = SubductionProblem(geom, A=A, Vs=Vs, sztype=sztype, qs=qs)

    sz.solve_steadystate_isoviscous()

    return sz


# In[ ]:


if __name__ == "__main__":
    resscales = [4.0, 2.0, 1.0]
    diagnostics_case1 = []
    for resscale in resscales:
        sz = solve_benchmark_case1(resscale)
        diagnostics_case1.append((resscale, sz.get_diagnostics()))


# In[ ]:


if __name__ == "__main__":
    print('')
    print('{:<12} {:<12} {:<12} {:<12} {:<12} {:<12}'.format('resscale', 'T_ndof', 'T_{200,-100}', 'Tbar_s', 'Tbar_w', 'Vrmsw'))
    for resscale, diag in diagnostics_case1:
        print('{:<12.4g} {:<12d} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}'.format(resscale, *diag))    


# For comparison here are the values reported for case 1 using [TerraFERMA](https://terraferma.github.io) in [Wilson & van Keken, 2023](http://dx.doi.org/10.1186/s40645-023-00588-6):
# 
# | `resscale` | $T_{\text{ndof}} $ | $T_{(200,-100)}^*$ | $\overline{T}_s^*$ | $ \overline{T}_w^* $ |  $V_{\text{rms},w}^*$ |
# | - | - | - | - | - | - |
# | 2.0 | 21403  | 517.17 | 451.83 | 926.62 | 34.64 |
# | 1.0 | 83935  | 516.95 | 451.71 | 926.33 | 34.64 |
# | 0.5 | 332307 | 516.86 | 451.63 | 926.15 | 34.64 |

# ### Benchmark case 2

# In[ ]:


def solve_benchmark_case2(resscale):
    xs = [0.0, 140.0, 240.0, 400.0]
    ys = [0.0, -70.0, -120.0, -200.0]
    lc_depth = 40
    uc_depth = 15
    coast_distance = 0
    extra_width = 0
    sztype = 'continental'
    io_depth_2 = 154
    A      = 100.0      # age of subducting slab (Myr)
    qs     = 0.065      # surface heat flux (W/m^2)
    Vs     = 100.0      # slab speed (mm/yr)

    slab = create_slab(xs, ys, resscale, lc_depth)
    geom = create_sz_geometry(slab, resscale, sztype, io_depth_2, extra_width, 
                               coast_distance, lc_depth, uc_depth)
    sz = SubductionProblem(geom, A=A, Vs=Vs, sztype=sztype, qs=qs)

    sz.solve_steadystate_dislocationcreep()

    return sz


# In[ ]:


if __name__ == "__main__":
    resscales = [4.0, 2.0, 1.0]
    diagnostics_case2 = []
    for resscale in resscales:
        sz = solve_benchmark_case2(resscale)
        diagnostics_case2.append((resscale, sz.get_diagnostics()))


# In[ ]:


if __name__ == "__main__":
    print('')
    print('{:<12} {:<12} {:<12} {:<12} {:<12} {:<12}'.format('resscale', 'T_ndof', 'T_{200,-100}', 'Tbar_s', 'Tbar_w', 'Vrmsw'))
    for resscale, diag in diagnostics_case2:
        print('{:<12.4g} {:<12d} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}'.format(resscale, *diag))    


# For comparison here are the values reported for case 2 using [TerraFERMA](https://terraferma.github.io) in [Wilson & van Keken, 2023](http://dx.doi.org/10.1186/s40645-023-00588-6):
# 
# | `resscale` | $T_{\text{ndof}} $ | $T_{(200,-100)}^*$ | $\overline{T}_s^*$ | $ \overline{T}_w^* $ |  $V_{\text{rms},w}^*$ |
# | - | - | - | - | - | - |
# | 2.0 | 21403  | 683.05 | 571.58 | 936.65 | 40.89 |
# | 1.0 | 83935 | 682.87 | 572.23 | 936.11 | 40.78 |
# | 0.5 | 332307 | 682.80 | 572.05 | 937.37 | 40.77 |

# ## Finish up

# Convert this notebook to a python script (making sure to save first)

# In[ ]:


if __name__ == "__main__" and "__file__" not in globals():
    from ipylab import JupyterFrontEnd
    app = JupyterFrontEnd()
    app.commands.execute('docmanager:save')
    get_ipython().system('jupyter nbconvert --NbConvertApp.export_format=script --ClearOutputPreprocessor.enabled=True sz_benchmark.ipynb')


# In[ ]:




