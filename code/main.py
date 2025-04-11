import fenics as fe
import matplotlib.pyplot as plt
from physics import define_weak_formulation, compute_magnetic_force
from geometry import create_mesh_and_subdomains
import numpy as np
from plot import plot_magnetic_force

# Define constants
LENGTH = 1.0  # Length of the domain (meters)
WIDTH = 1.0   # Width of the domain (meters)
HEIGHT = 1.0  # Height of the domain (meters)
MU_ELECTROMAGNET = 4 * fe.pi * 1e-7  # Electromagnet permeability (H/m)
MU_AIR = 1.0  # Air permeability
EFFECTIVE_CONDUCTIVITY = 0.283  # 28.3% effective surface
METAL_SHEET_THICKNESS = 0.0007
METAL_SHEET_POSITIONS = [0.5 - 0.00035 - i * 0.0002 for i in range(4)]  # Centered in z-direction
NUM_ELECTROMAGNETS = 6
ELECTROMAGNET_RADIUS = 0.01  # 20 mm diameter
ELECTROMAGNET_HEIGHT = 0.015  # 15 mm length

# Calculate current magnitude based on parameters
ELECTROMAGNET_AREA = fe.pi * ELECTROMAGNET_RADIUS**2  # Cross-sectional area of an electromagnet
DESIRED_MAGNETIC_FIELD = 1.0  # Desired magnetic field strength (T)
CURRENT_MAGNITUDE = DESIRED_MAGNETIC_FIELD / (MU_ELECTROMAGNET * ELECTROMAGNET_AREA)

# Solver function from solver.py
def solve_weak_formulation(V, weak_form, bc):
    """
    Solve the weak formulation for the magnetic vector potential A.

    Parameters:
        V (FunctionSpace): Function space for the magnetic vector potential.
        weak_form (Form): Weak formulation of the problem.
        bc (DirichletBC): Boundary condition.

    Returns:
        Function: Solved magnetic vector potential A.
    """
    # Define the solution function
    A = fe.Function(V)

    # Solve the weak formulation
    fe.solve(fe.lhs(weak_form) == fe.rhs(weak_form), A, bc)

    return A

# Create mesh and subdomains
mesh, mu, J, V = create_mesh_and_subdomains(
    LENGTH, WIDTH, HEIGHT, MU_ELECTROMAGNET, MU_AIR, CURRENT_MAGNITUDE,
    EFFECTIVE_CONDUCTIVITY, METAL_SHEET_THICKNESS, METAL_SHEET_POSITIONS,
    NUM_ELECTROMAGNETS, ELECTROMAGNET_RADIUS, ELECTROMAGNET_HEIGHT
)

# Define the weak formulation
V, weak_form, v, A = define_weak_formulation(mesh, mu, J, EFFECTIVE_CONDUCTIVITY, METAL_SHEET_POSITIONS, METAL_SHEET_THICKNESS)

# Solve the weak formulation
A_solution = solve_weak_formulation(V, weak_form, None)

# Compute the magnetic force in the z-direction
f_z = compute_magnetic_force(mesh, A_solution, mu)

# Save the results for visualization
fe.File("magnetic_vector_potential.pvd") << A_solution
fe.File("magnetic_force_z.pvd") << f_z

# Define z-positions for plotting
z_plot_positions = [0.1, 0.2, 0.3, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5]  # Example z-positions near the center

# Plot the magnetic force
plot_magnetic_force(f_z, mesh, ELECTROMAGNET_RADIUS, ELECTROMAGNET_HEIGHT, METAL_SHEET_POSITIONS, z_plot_positions)

print("Simulation complete. Results saved to files.")