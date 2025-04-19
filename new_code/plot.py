import fenics as fe
import matplotlib.pyplot as plt
import numpy as np

def plot_magnetic_potential(b_solution, mesh):
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
        b_plane = np.array([b_solution(p)[2] for p in points]).reshape(X.shape)

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

def plot_magnetic_potential_magnitude_potential(b_solution, mesh):
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
            potential_sum = sum(b_solution(fe.Point(p[0], p[1], z)) for p in points)
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
