import fenics as fe
import numpy as np
import matplotlib as plt
from physics import compute_magnetic_force
from plot import plot_magnetic_potential, plot_magnetic_potential_magnitude_force, plot_magnetic_potential_magnitude_potential
from physics import calculate_solution

def main():
    # Define metal sheet parameters
    num_metal_sheets = 4
    metal_sheet_length = 0.2
    metal_sheet_width = 0.2
    metal_sheet_thickness = 0.0007
    density_metal_sheet = 7800  # Density of the metal sheet (kg/m^3)
    effective_conductivity = 0.283  # 28.3% effective surface
    mu_metal_sheet_real = 4 * fe.pi * 1e-7  # Metal sheet permeability (H/m)
    mu_metal_sheet = mu_metal_sheet_real * effective_conductivity  # Adjusted for effective conductivity

    # General constants
    mu_air = 1.0  # Air permeability
    distance_between_metal_sheets = 0.0002
    g = 9.81  # Gravitational acceleration (m/s^2)

    # Define electromagnet parameters
    num_electromagnets_length = 1
    num_electromagnets_width = 1
    electromagnet_radius = 0.2  # 20 mm diameter
    electromagnet_height = 0.015  # 15 mm length
    mu_electromagnet = 4 * fe.pi * 1e-7  # Electromagnet permeability (H/m)

    # Define the mesh and function space
    length_of_domain = metal_sheet_length * 2
    width_of_domain = metal_sheet_width * 2
    height_of_domain = ( electromagnet_height / 2 + num_metal_sheets * (metal_sheet_thickness + distance_between_metal_sheets) \
                       + (metal_sheet_thickness + distance_between_metal_sheets) / 2 ) * 4  # Corrected to scalar

    # Calculate the gravitational force acting on the first metal sheet
    volume_metal_sheet = metal_sheet_length * metal_sheet_width * metal_sheet_thickness
    gravitational_force = volume_metal_sheet * density_metal_sheet * g

    # Iteratively calculate the desired magnetic field
    tolerance = 1e-6  # Convergence tolerance
    max_iterations = 100  # Maximum number of iterations
    desired_magnetic_field = 1e-8  # Initial guess for the magnetic field (T)
    iteration = 0

    while iteration < max_iterations:
        # Calculate current magnitude based on the current guess for the magnetic field
        electromagnet_area = fe.pi * electromagnet_radius**2  # Cross-sectional area of an electromagnet
        current_magnitude = desired_magnetic_field / (mu_electromagnet * electromagnet_area)

        # Initial resolution
        resolution = 20

        # Attempt to solve the system, retrying with reduced mesh resolution if necessary
        max_retries = 10
        retries = 0
        while retries < max_retries:
            try:
                # Perform all calculations and solve the system
                mesh, nedelec_first_kind, a_solution, mu, domain = calculate_solution(
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
                retries += 1
                if retries < max_retries:
                    print("Retrying with reduced mesh resolution...")
                    resolution = max(10, resolution - 10)  # Reduce resolution by 10 each retry
                    print(f"New resolution: {resolution}")
                else:
                    print("Maximum retries reached. Exiting.")
                    return

        # Magnetic field B = curl(A)
        b_solution = fe.curl(a_solution)

        # Compute the magnetic force in the z-direction
        f_z = compute_magnetic_force(mesh, b_solution, domain, mu_air, mu_electromagnet, mu_metal_sheet)

        # Integrate the magnetic force over the first metal sheet
        try:
            dx_metal_sheet = fe.Measure("dx", domain=domain)
            magnetic_force = fe.assemble(f_z * dx_metal_sheet(2))  # Subdomain ID 2 corresponds to the first metal sheet
        except RuntimeError as e:
            print(f"Error during magnetic force computation: {e}")
            magnetic_force = 0.0

        # Check for convergence
        if magnetic_force == 0.0:
            print("Magnetic force is zero. Adjusting the magnetic field to avoid division by zero.")
            desired_magnetic_field *= 1.1  # Increase the magnetic field slightly to avoid zero division
            iteration += 1
            continue

        if abs(magnetic_force - gravitational_force) < tolerance:
            print(f"Converged to desired magnetic field: {desired_magnetic_field} T")
            break

        # Update the guess for the magnetic field
        desired_magnetic_field *= gravitational_force / magnetic_force
        iteration += 1
        print(f"Iteration {iteration}: Updated magnetic field = {desired_magnetic_field} T")

    if iteration == max_iterations:
        print("Failed to converge to the desired magnetic field within the maximum number of iterations.")
        return

    # Extract the z-component of the magnetic field as a scalar field
    V_scalar = fe.FunctionSpace(mesh, "CG", 1)
    b_solution_z = fe.project(b_solution[2], V_scalar)

    # Plot the magnitude of the magnetic force in the z-direction
    plot_magnetic_potential_magnitude_force(f_z, mesh)

    # Plot the magnitude of the magnetic potential in the z-direction
    plot_magnetic_potential_magnitude_potential(b_solution_z, mesh)

    # Plot the magnetic vector potential (z-component)
    plot_magnetic_potential(b_solution_z, mesh)

if __name__ == "__main__":
    main()