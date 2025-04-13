import fenics as fe
import numpy as np
import matplotlib as plt
from physics import compute_magnetic_force
from plot import plot_magnetic_potential, plot_magnetic_potential_magnitude  # Import the new function
from physics import calculate_solution

def main():
    # Define metal sheet parameters
    num_metal_sheets = 4
    metal_sheet_length = 0.2
    metal_sheet_width = 0.2
    metal_sheet_thickness = 0.0007
    effective_conductivity = 0.283  # 28.3% effective surface
    mu_metal_sheet_real = 4 * fe.pi * 1e-7  # Metal sheet permeability (H/m)
    mu_metal_sheet = mu_metal_sheet_real * effective_conductivity  # Adjusted for effective conductivity

    # General constants
    mu_air = 1.0  # Air permeability
    distance_between_metal_sheets = 0.0002

    # Define electromagnet parameters
    num_electromagnets_length = 6
    num_electromagnets_width = 6
    electromagnet_radius = 0.03  # 20 mm diameter
    electromagnet_height = 0.015  # 15 mm length
    mu_electromagnet = 4 * fe.pi * 1e-7  # Electromagnet permeability (H/m)

    # Define the mesh and function space
    length_of_domain = metal_sheet_length * 2
    width_of_domain = metal_sheet_width * 2
    height_of_domain = ( electromagnet_height / 2 + num_metal_sheets * (metal_sheet_thickness + distance_between_metal_sheets) \
                       + (metal_sheet_thickness + distance_between_metal_sheets) / 2 ) * 4  # Corrected to scalar

    # Calculate current magnitude based on parameters
    electromagnet_area = fe.pi * electromagnet_radius**2  # Cross-sectional area of an electromagnet
    desired_magnetic_field = 1.0 * 10 ** (-8)  # Desired magnetic field strength (T)
    current_magnitude = desired_magnetic_field / (mu_electromagnet * electromagnet_area)

    # Initial resolution
    resolution = 20

    # Attempt to solve the system, retrying with reduced mesh resolution if necessary
    max_retries = 10
    retries = 0
    while retries < max_retries:
        try:
            # Perform all calculations and solve the system
            mesh, nedelec_first_kind, b_solution, mu, domain = calculate_solution(
                resolution, length_of_domain, width_of_domain, height_of_domain, mu_air,
                num_electromagnets_length, num_electromagnets_width, electromagnet_radius,
                electromagnet_height, mu_electromagnet, num_metal_sheets, metal_sheet_length,
                metal_sheet_width, metal_sheet_thickness, mu_metal_sheet, distance_between_metal_sheets,
                current_magnitude
            )
            print("Solver succeeded.")
            break
        except RuntimeError as e:
            print(f"RuntimeError encountered during solve: {e}")
            if retries < max_retries - 1:
                print("Retrying with reduced mesh resolution...")
                resolution = max(10, resolution - 10)  # Reduce resolution by 10 each retry
                print(f"New resolution: {resolution}")
                retries += 1
            else:
                print("Maximum retries reached. Exiting.")
                return

    # Compute the magnetic force in the z-direction
    f_z = compute_magnetic_force(mesh, b_solution, domain, mu_air, mu_electromagnet, mu_metal_sheet)

    # Define z-values for plotting
    z_values = np.linspace(0, height_of_domain, 5)

    # Plot the magnitude of the magnetic force in the z-direction
    plot_magnetic_potential_magnitude(f_z, mesh, z_values)

    # Plot the magnitude of the magnetic potential in the z-direction
    plot_magnetic_potential_magnitude(b_solution, mesh, z_values)

    # Plot the magnetic vector potential
    plot_magnetic_potential(b_solution, mesh, z_values)

if __name__ == "__main__":
    main()