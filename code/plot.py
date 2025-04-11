import matplotlib.pyplot as plt
import numpy as np

def plot_magnetic_force(f_z, mesh, electromagnet_radius, electromagnet_height, metal_sheet_positions, z_values):
    """
    Plot the magnetic force in the x-y plane for different z-directions, including electromagnets and metal sheets.

    Parameters:
        f_z (Function): Magnetic force in the z-direction.
        mesh (Mesh): The computational mesh.
        electromagnet_radius (float): Radius of the electromagnets.
        electromagnet_height (float): Height of the electromagnets.
        metal_sheet_positions (list): Z-positions of the metal sheets.
        z_values (list): Z-positions for which to plot the magnetic force.
    """
    LENGTH = mesh.coordinates()[:, 0].max()
    WIDTH = mesh.coordinates()[:, 1].max()
    NUM_ELECTROMAGNETS = int(np.sqrt(len(mesh.coordinates())))

    # Create a mesh grid for plotting
    x = np.linspace(0, LENGTH, 100)
    y = np.linspace(0, WIDTH, 100)
    X, Y = np.meshgrid(x, y)

    for z in z_values:
        # Evaluate the magnetic force at the given z-plane
        f_z_plane = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                f_z_plane[i, j] = f_z(fe.Point(X[i, j], Y[i, j], z))

        # Plot the magnetic force
        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, f_z_plane, levels=50, cmap="viridis")
        plt.colorbar(label="Magnetic Force (N/m^3)")
        plt.title(f"Magnetic Force in x-y Plane at z = {z:.3f} m")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")

        # Overlay electromagnets
        for i in range(NUM_ELECTROMAGNETS):
            for j in range(NUM_ELECTROMAGNETS):
                center_x = (i - (NUM_ELECTROMAGNETS - 1) / 2) * electromagnet_radius * 2 + LENGTH / 2
                center_y = (j - (NUM_ELECTROMAGNETS - 1) / 2) * electromagnet_radius * 2 + WIDTH / 2
                circle = plt.Circle((center_x, center_y), electromagnet_radius, color="red", fill=False, label="Electromagnet")
                plt.gca().add_artist(circle)

        # Overlay metal sheets
        for sheet_z in metal_sheet_positions:
            if abs(sheet_z - z) < electromagnet_height / 2:
                rect = plt.Rectangle((0, 0), LENGTH, WIDTH, color="blue", alpha=0.3, label="Metal Sheet")
                plt.gca().add_artist(rect)

        plt.legend(["Electromagnet", "Metal Sheet"], loc="upper right")
        plt.show()
