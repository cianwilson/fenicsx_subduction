from mpi4py import MPI
import dolfinx as df
import pyvista as pv
import numpy as np

try:
    pv.start_xvfb()
except OSError:
    pass

Myr_to_s = lambda a: a*1.e6*365.25*24*60*60
nondim_to_K = lambda T: T + 273.15
mmpyr_to_mps = lambda v: v*1.0e-3/365.25/24/60/60
mps_to_mmpyr = lambda v: v*1.0e3*365.25*24*60*60

def create_submesh(mesh, cell_indices, cell_tags=None, facet_tags=None):
    """
    Function to return a submesh based on the cell indices provided.

    Arguments:
      * mesh         - original (parent) mesh
      * cell_indices - cell indices of the parent mesh to include in the submesh

    Keyword Arguments:
      * cell_tags    - cell tags on parent mesh that will be mapped and returned relative to submesh (default=None)
      * facet_tags   - facet tags on parent mesh that will be mapped and returned relative to submesh (default=None)

    Returns:
      * submesh            - submesh of mesh given cell_indices
      * submesh_cell_tags  - cell tags relative to submesh if cell_tags provided (otherwise None)
      * submesh_facet_tags - facet tags relative to submesh if facet_tags provided (otherwise None)
      * submesh_cell_map   - map from submesh cells to parent cells
    """
    tdim = mesh.topology.dim
    fdim = tdim-1
    submesh, submesh_cell_map, submesh_vertex_map, submesh_geom_map = \
                  df.mesh.create_submesh(mesh, tdim, cell_indices)
    submesh.topology.create_connectivity(fdim, tdim)

    # if cell_tags are provided then map to the submesh
    submesh_cell_tags = None
    if cell_tags is not None:
        submesh_cell_tags_indices = []
        submesh_cell_tags_values  = []
        # loop over the submesh cells, checking if they're included in
        # the parent cell_tags
        for i,parentind in enumerate(submesh_cell_map):
            parent_cell_tags_indices = np.argwhere(cell_tags.indices==parentind)
            if parent_cell_tags_indices.shape[0]>0:
                submesh_cell_tags_indices.append(i)
                submesh_cell_tags_values.append(cell_tags.values[parent_cell_tags_indices[0][0]])
        submesh_cell_tags_indices = np.asarray(submesh_cell_tags_indices)
        submesh_cell_tagsvalues  = np.asarray(submesh_cell_tags_values)

        # create a new meshtags object
        # indices should already be sorted by construction
        submesh_cell_tags = df.mesh.meshtags(mesh, tdim, 
                                             submesh_cell_tags_indices, 
                                             submesh_cell_tags_values)
            

    # if facet_tags are provided then map to the submesh
    submesh_facet_tags = None
    if facet_tags is not None:
        # parent facet to vertices adjacency list
        f2vs = mesh.topology.connectivity(fdim, 0)

        # submesh facet to vertices adjaceny list
        submesh.topology.create_connectivity(fdim, 0)
        submesh_f2vs = submesh.topology.connectivity(fdim, 0)
        # create a map from the parent vertices to the submesh facets
        # (only for the facets that exist in the submesh)
        submesh_parentvs2subf = dict()
        for i in range(submesh_f2vs.num_nodes):
            submesh_parentvs2subf[tuple(sorted([submesh_vertex_map[j] for j in submesh_f2vs.links(i)]))] = i

        # loop over the facet_tags and map from the parent facet to the submesh facet
        # via the vertices, copying over the facet_tag values
        submesh_facet_tags_indices = []
        submesh_facet_tags_values  = []
        for i,parentind in enumerate(facet_tags.indices):
            subind = submesh_parentvs2subf.get(tuple(sorted(f2vs.links(parentind))), None)
            if subind is not None:
                submesh_facet_tags_indices.append(subind)
                submesh_facet_tags_values.append(facet_tags.values[i])
        submesh_facet_tags_indices = np.asarray(submesh_facet_tags_indices)
        submesh_facet_tags_values  = np.asarray(submesh_facet_tags_values)

        perm = np.argsort(submesh_facet_tags_indices)
        submesh_facet_tags = df.mesh.meshtags(mesh, fdim, 
                                              submesh_facet_tags_indices[perm], 
                                              submesh_facet_tags_values[perm])
    
    return submesh, submesh_cell_tags, submesh_facet_tags, submesh_cell_map

def get_first_cells(x, mesh):
    """
    Given a list of points and a mesh, return the first cells that each point lies in in the mesh.

    Arguments:
      * x    - coordinates of points
      * mesh - mesh

    Returns:
      * first_cells - a list of cells corresponding to each point in x
    """
    tree = df.geometry.bb_tree(mesh, mesh.geometry.dim)
    xl = x
    if len(x.shape)==1: xl = [x]
    first_cells = []
    for x0 in xl:
        cell_candidates = df.geometry.compute_collisions_points(tree, x0)
        cell = df.geometry.compute_colliding_cells(mesh, cell_candidates, x0).array
        assert len(cell) > 0
        first_cells.append(cell[0])
    return first_cells

def plot_mesh(mesh, tags=None):
    """
    Plot a dolfinx mesh using pyvista.

    Arguments:
      * mesh - the mesh to plot

    Keyword Arguments:
      * tags - mesh tags to color plot by (either cell or facet, default=None)
    """
    # Create VTK mesh
    cells, types, x = df.plot.vtk_mesh(mesh)
    grid = pv.UnstructuredGrid(cells, types, x)

    tdim = mesh.topology.dim
    fdim = tdim - 1
    if tags is not None:
        marker = np.zeros(mesh.topology.index_map(2).size_local)
        if tags.dim == tdim:
            for i, ind in enumerate(tags.indices):
                marker[ind] = tags.values[i]
        elif tags.dim == fdim:
            mesh.topology.create_connectivity(fdim, tdim)
            fcc = mesh.topology.connectivity(fdim, tdim)
            for f,v in enumerate(tags.values):
                for c in fcc.links(tags.indices[f]):
                    marker[c] = v
        else:
            raise Exception("Unknown tag dimension!") 
        grid.cell_data["Marker"] = marker
        grid.set_active_scalars("Marker")
    
    plotter = pv.Plotter(window_size=[800,800])
    plotter.add_mesh(grid, show_edges=True, show_scalar_bar=False)
    #plotter.show_bounds()
    plotter.view_xy()
    plotter.show()

def plot_scalar(scalar, show_edges=False, log_scale=False, scale=1.0):
    """
    Plot a dolfinx scalar Function using pyvista.

    Arguments:
      * scalar     - the dolfinx Function to plot

    Keyword Arguments:
      * show_edges - plot the mesh facets (default=False)
      * log_scale  - use a log scale colormap (default=False)
      * scale      - a scalar scale factor that the values are multipled by (default=1.0)
    """
    # Create VTK mesh
    if scalar.function_space.element.space_dimension==1:
        cells, types, x = df.plot.vtk_mesh(scalar.function_space.mesh)
    else:
        cells, types, x = df.plot.vtk_mesh(scalar.function_space)
    grid = pv.UnstructuredGrid(cells, types, x)

    if scalar.function_space.element.space_dimension==1:
        tdim = scalar.function_space.mesh.topology.dim
        cell_imap = scalar.function_space.mesh.topology.index_map(tdim)
        num_cells = cell_imap.size_local + cell_imap.num_ghosts
        perm = [scalar.function_space.dofmap.cell_dofs(c)[0] for c in range(num_cells)]
        grid.cell_data[scalar.name] = scalar.x.array.real[perm]*scale if not log_scale else np.log10(scalar.x.array.real[perm]*scale)
    else:
        grid.point_data[scalar.name] = scalar.x.array.real*scale if not log_scale else np.log10(scalar.x.array.real*scale)
    grid.set_active_scalars(scalar.name)
    
    plotter = pv.Plotter(window_size=[800,800])
    plotter.add_mesh(grid, show_edges=show_edges, show_scalar_bar=True, cmap='coolwarm')
    #plotter.show_bounds()
    plotter.view_xy()
    plotter.show()

def plot_vector(vector, show_edges=False, glyph_factor=4, scale=1.0):
    """
    Plot a dolfinx vector Function using pyvista.

    Arguments:
      * vector       - the dolfinx Function to plot

    Keyword Arguments:
      * show_edges   - plot the mesh facets (default=False)
      * glyph_factor - scale for glyph size (default=4)
      * scale        - a scalar scale factor that the values are multipled by (default=1.0)
    """
    # Create VTK mesh
    cells, types, x = df.plot.vtk_mesh(vector.function_space)
    grid = pv.UnstructuredGrid(cells, types, x)

    values = np.zeros((x.shape[0], 3))
    values[:, :len(vector)] = vector.x.array.real.reshape((x.shape[0], len(vector)))*scale
    grid[vector.name] = values
    geom = pv.Arrow()
    glyphs = grid.glyph(orient=vector.name, factor=glyph_factor, geom=geom)
    
    plotter = pv.Plotter(window_size=[800,800])
    plotter.add_mesh(grid, show_edges=show_edges, show_scalar_bar=False, cmap='coolwarm')
    plotter.add_mesh(glyphs, cmap='coolwarm', show_scalar_bar=True)
    #plotter.show_bounds()
    plotter.view_xy()
    plotter.show()
