import fenics as fe
import matplotlib.pyplot as plt
import numpy as np

def plot_magnetic_potential(b_solution, mesh, z_values):
    """
    Plot the magnetic vector potential solution in the x-y plane for different z-values.

    Parameters:
        b_solution (Function): Magnetic vector potential solution.
        mesh (Mesh): The computational mesh.
        z_values (list): Z-positions for which to plot the magnetic potential.
    """
    coordinates = mesh.coordinates()
    x_min, x_max = coordinates[:, 0].min(), coordinates[:, 0].max()
    y_min, y_max = coordinates[:, 1].min(), coordinates[:, 1].max()

    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)

    for z in z_values:
        # Evaluate the z-component of the magnetic potential at the given z-plane
        points = np.array([fe.Point(xi, yi, z) for xi, yi in zip(X.flatten(), Y.flatten())])
        b_plane = np.array([b_solution(p)[2] for p in points]).reshape(X.shape)

        # Create a 2D plot using Matplotlib
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

def plot_magnetic_potential_magnitude(b_solution, mesh, z_values):
    """
    Plot the magnitude of the magnetic potential in the z-direction as a function of the z-direction.

    Parameters:
        b_solution (Function): Magnetic vector potential solution.
        mesh (Mesh): The computational mesh.
        z_values (list): Z-positions for which to compute the magnetic potential magnitude.
    """
    z_magnitudes = []

    for z in z_values:
        # Compute the magnitude of the magnetic potential in the z-direction
        magnitude = 0.0
        for cell in fe.cells(mesh):
            cell_coords = np.array(cell.get_vertex_coordinates()).reshape((-1, 3))
            for coord in cell_coords:
                if abs(coord[2] - z) < 1e-6:  # Check if the point is in the z-plane
                    magnitude += np.linalg.norm(b_solution(fe.Point(coord[0], coord[1], coord[2])))

        z_magnitudes.append(magnitude)

    # Plot the magnitude as a function of z
    plt.figure(figsize=(8, 6))
    plt.plot(z_values, z_magnitudes, marker="o", linestyle="-", color="blue")
    plt.title("Magnitude of Magnetic Potential in z-direction")
    plt.xlabel("z (m)")
    plt.ylabel("Magnetic Potential Magnitude")
    plt.grid()
    plt.show()


