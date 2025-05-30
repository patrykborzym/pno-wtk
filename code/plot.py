import fenics as fe
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import meshio

def plot_magnetic_potential(b_solution_z, mesh):
    """
    Plot the heat map of the z-component of the magnetic potential in the x-y plane for the last 5 z-values under half of the domain.

    Parameters:
        b_solution_z (Function): Z-component of the magnetic vector potential (scalar field).
        mesh (Mesh): The computational mesh.
    """
    coordinates = mesh.coordinates()
    z_coords = np.unique(coordinates[:, 2])

    # Filter z-values to include only those under half of the domain
    z_half = z_coords[z_coords <= z_coords.max() / 2]

    # Take the last 5 z-values
    z_values = z_half[-5:]

    x_min, x_max = coordinates[:, 0].min(), coordinates[:, 0].max()
    y_min, y_max = coordinates[:, 1].min(), coordinates[:, 1].max()

    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)

    for z in z_values:
        # Evaluate the z-component of the magnetic potential at the given z-plane
        points = np.array([fe.Point(xi, yi, z) for xi, yi in zip(X.flatten(), Y.flatten())])
        b_plane = np.array([b_solution_z(fe.Point(p[0], p[1], z)) for p in points]).reshape(X.shape)

        # Create a 2D heat map using Matplotlib
        plt.figure(figsize=(8, 6))
        contour = plt.contourf(X, Y, b_plane, levels=50, cmap="plasma")
        plt.colorbar(contour, label="Magnetic Potential (A/m)")

        # Overlay the mesh using Matplotlib
        for cell in fe.cells(mesh):
            cell_coords = np.array(cell.get_vertex_coordinates()).reshape((-1, 3))  # Convert to NumPy array and reshape
            x_coords = np.append(cell_coords[:, 0], cell_coords[0, 0])  # Close the loop for the cell
            y_coords = np.append(cell_coords[:, 1], cell_coords[0, 1])
            plt.plot(x_coords, y_coords, color="black", linewidth=0.5)

        plt.title(f"Magnetic Potential (z-component) at z = {z:.3f} m")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.show()

def plot_magnetic_potential_magnitude_force(f_z, mesh):
 # Extract unique z-values from the mesh
    coordinates = mesh.coordinates()
    z_values = np.unique(coordinates[:, 2])

    # Compute the total sum of the magnetic potential for each z-value
    z_totals = []
    for z in z_values:
        # Get all points in the mesh at the given z-plane
        points = coordinates[np.abs(coordinates[:, 2] - z) < 1e-6]
        if len(points) > 0:
            # Sum the magnetic potential values at these points
            potential_sum = sum(f_z(fe.Point(p[0], p[1], z)) for p in points)
            z_totals.append(potential_sum)
        else:
            z_totals.append(0.0)

    # Plot the total magnetic potential as a function of z
    plt.figure(figsize=(8, 6))
    plt.plot(z_values, z_totals, marker="o", linestyle="-", color="blue")
    plt.title("Total Magnetic Potential in z-direction")
    plt.xlabel("z (m)")
    plt.ylabel("Total Magnetic Potential (A/m)")
    plt.grid()
    plt.show()

def plot_magnetic_potential_magnitude_potential(b_solution_z, mesh):
    """
    Calculate and plot the total magnetic potential (z-component) for all z-values in the mesh.

    Parameters:
        b_solution_z (Function): Z-component of the magnetic vector potential (scalar field).
        mesh (Mesh): The computational mesh.
    """
    # Extract unique z-values from the mesh
    coordinates = mesh.coordinates()
    z_values = np.unique(coordinates[:, 2])

    # Compute the total sum of the z-component of the magnetic potential for each z-value
    z_totals = []
    for z in z_values:
        # Get all points in the mesh at the given z-plane
        points = coordinates[np.abs(coordinates[:, 2] - z) < 1e-6]
        if len(points) > 0:
            # Sum the z-component of the magnetic potential values at these points
            potential_sum = sum(b_solution_z(fe.Point(p[0], p[1], z)) for p in points)
            z_totals.append(potential_sum)
        else:
            z_totals.append(0.0)

    # Plot the total magnetic potential (z-component) as a function of z
    plt.figure(figsize=(8, 6))
    plt.plot(z_values, z_totals, marker="o", linestyle="-", color="blue")
    plt.title("Total Magnetic Potential (z-component) in z-direction")
    plt.xlabel("z (m)")
    plt.ylabel("Total Magnetic Potential (A/m)")
    plt.grid()
    plt.show()


def mu_values_visualisation(mesh, subdomains, mu):
    
    with fe.XDMFFile(mesh.mpi_comm(), "mesh.xdmf") as f:
        f.write(mesh)
    with fe.XDMFFile(mesh.mpi_comm(), "subdomains.xdmf") as f:
        f.write(subdomains)
    with fe.XDMFFile(mesh.mpi_comm(), "mu.xdmf") as f:
        f.write(mu)

    meshio_mesh = meshio.read("mesh.xdmf")
    meshio.write("mesh.vtk", meshio_mesh)
    pv_mesh = pv.read("mesh.vtk")

    pv_mesh.cell_data["Subdomains"] = subdomains.array().astype(float)
    mu_vals = mu.vector().get_local()
    if len(mu_vals) == pv_mesh.n_cells:
        pv_mesh.cell_data["Magnetic Permeability"] = mu_vals
    else:
        interp_vals = np.zeros(pv_mesh.n_cells)
        for i, cell in enumerate(pv_mesh.cell_centers().points):
            interp_vals[i] = mu(fe.Point(*cell))
        pv_mesh.cell_data["Magnetic Permeability"] = interp_vals

    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, scalars="Subdomains", cmap="viridis", opacity=0.5, show_edges=True)
    plotter.add_scalar_bar(title="Subdomains")
    plotter.show()

    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, scalars="Magnetic Permeability", cmap="plasma", opacity=0.5, show_edges=True)
    plotter.add_scalar_bar(title="Magnetic Permeability")
    plotter.show()


def plot_2d_domain_markings_and_mu(mesh, subdomains, mu):
    """
    Plot 2D heat maps of the domain markings and mu values in the x-y plane for the last 5 z-values under half of the domain.

    Parameters:
        mesh (Mesh): The computational mesh.
        subdomains (MeshFunction): Subdomain markings.
        mu (Function): Magnetic permeability values.
    """
    coordinates = mesh.coordinates()
    z_coords = np.unique(coordinates[:, 2])

    # Filter z-values to include only those under half of the domain
    z_half = z_coords[z_coords <= z_coords.max() / 2]

    # Take the last 5 z-values
    z_values = z_half

    x_min, x_max = coordinates[:, 0].min(), coordinates[:, 0].max()
    y_min, y_max = coordinates[:, 1].min(), coordinates[:, 1].max()

    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)

    for z in z_values:
        # Evaluate subdomain markings at the given z-plane
        points = np.array([fe.Point(xi, yi, z) for xi, yi in zip(X.flatten(), Y.flatten())])
        markings_plane = np.zeros(X.shape)
        for i, p in enumerate(points):
            cell = fe.Cell(mesh, mesh.bounding_box_tree().compute_first_entity_collision(p))
            if cell.contains(p):
                markings_plane.flat[i] = subdomains[cell.index()]
            else:
                markings_plane.flat[i] = 0  # Default to air if no cell contains the point

        plt.figure(figsize=(8, 6))
        contour = plt.contourf(X, Y, markings_plane, levels=np.arange(0, 3), cmap="viridis")
        plt.colorbar(contour, label="Domain Markings")
        plt.title(f"Domain Markings at z = {z:.3f} m")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.show()

        # Evaluate mu values at the given z-plane
        mu_plane = np.array([mu(p) for p in points]).reshape(X.shape)

        plt.figure(figsize=(8, 6))
        contour = plt.contourf(X, Y, mu_plane, levels=50, cmap="plasma")
        plt.colorbar(contour, label="Magnetic Permeability (H/m)")
        plt.title(f"Magnetic Permeability at z = {z:.3f} m")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.show()