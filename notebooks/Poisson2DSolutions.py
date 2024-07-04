#!/usr/bin/env python
# coding: utf-8

# # Poisson Example 2D Solutions

# ## Load

# In[ ]:


from Poisson2D import *


# ## Testing

# In[ ]:


def evaluate_error(T_i):
    """
    A python function to evaluate the l2 norm of the error in 
    the two dimensional Poisson problem given a known analytical
    solution.
    """
    # Define the exact solution
    x  = ufl.SpatialCoordinate(T_i.function_space.mesh)
    Te = ufl.exp(x[0] + x[1]/2.)
    
    # Define the error between the exact solution and the given
    # approximate solution
    l2err = df.fem.assemble_scalar(df.fem.form((T_i - Te)*(T_i - Te)*ufl.dx))
    l2err = T_i.function_space.mesh.comm.allreduce(l2err, op=MPI.SUM)**0.5
    
    # Return the l2 norm of the error
    return l2err


# In[ ]:


if __name__ == "__main__":
    # Open a figure for plotting
    fig = pl.figure()
    
    # Make an output folder
    output_folder = pathlib.Path("output")
    output_folder.mkdir(exist_ok=True, parents=True)
    
    # List of polynomial orders to try
    ps = [1, 2]
    # List of resolutions to try
    nelements = [10, 20, 40, 80, 160, 320]
    # Keep track of whether we get the expected order of convergence
    test_passes = True
    # Loop over the polynomial orders
    for p in ps:
    # Accumulate the errors
        errors_l2_a = []
        # Loop over the resolutions
        for ne in nelements:
            # Solve the 2D Poisson problem
            T_i = solve_poisson_2d(ne, p)
            # Evaluate the error in the approximate solution
            l2error = evaluate_error(T_i)
            # Print to screen and save
            print('ne = ', ne, ', l2error = ', l2error)
            errors_l2_a.append(l2error)
        
        # Work out the order of convergence at this p
        hs = 1./np.array(nelements)/p
        
        # Write the errors to disk
        with open(output_folder / '2d_poisson_convergence_p{}.csv'.format(p), 'w') as f:
            np.savetxt(f, np.c_[nelements, hs, errors_l2_a], delimiter=',', 
                    header='nelements, hs, l2errs')
        
        # Fit a line to the convergence data
        fit = np.polyfit(np.log(hs), np.log(errors_l2_a),1)
        print("***********  order of accuracy p={}, order={:.2f}".format(p,fit[0]))
        
        # log-log plot of the error  
        pl.loglog(hs,errors_l2_a,'o-',label='p={}, order={:.2f}'.format(p,fit[0]))
        
        # Test if the order of convergence is as expected
        test_passes = test_passes and fit[0] > p+0.9
    
    # Tidy up the ploy
    pl.xlabel('h')
    pl.ylabel('||e||_2')
    pl.grid()
    pl.title('Convergence')
    pl.legend()
    pl.savefig(output_folder / '2d_poisson_convergence.pdf')
    
    print("***********  convergence figure in output/poisson_convergence.pdf")
    # Check if we passed the test
    assert(test_passes)


# ## Finish up

# Convert this notebook to a python script (making sure to save first)

# In[ ]:


if __name__ == "__main__" and "__file__" not in globals():
    from ipylab import JupyterFrontEnd
    app = JupyterFrontEnd()
    app.commands.execute('docmanager:save')
    get_ipython().system('jupyter nbconvert --NbConvertApp.export_format=script --ClearOutputPreprocessor.enabled=True Poisson2DSolutions.ipynb')


# In[ ]:




