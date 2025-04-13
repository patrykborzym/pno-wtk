import fenics as fe
import matplotlib.pyplot as plt
from geometry import create_mesh_and_subdomains
from physics import CurrentDensity
from physics import define_weak_form
from physics import compute_magnetic_force

N_POINTS_P_AXIS = 12
FORCING_MAGNITUDE = 1.0

def main():
    # Define metal sheet parameters
    num_metal_sheets = 4
    metal_sheet_length = 0.2
    metal_sheet_width = 0.2
    metal_sheet_thickness = 0.0007
    effective_conductivity = 0.283  # 28.3% effective surface
    mu_metal_sheet = 4 * fe.pi * 1e-7  # Metal sheet permeability (H/m)

    # General constants
    mu_air = 1.0  # Air permeability
    distance_between_metal_sheets = 0.0002

    # Define electromagnet parameters
    num_electromagnets_length = 6
    num_electromagnets_width = 6
    electromagnet_radius = 0.01  # 20 mm diameter
    electromagnet_height = 0.015  # 15 mm length
    mu_electromagnet = 4 * fe.pi * 1e-7  # Electromagnet permeability (H/m)

    # Define the mesh and function space
    length_of_domain = metal_sheet_length * 2
    width_of_domain = metal_sheet_width * 2
    height_of_domain = [electromagnet_height / 2 + num_metal_sheets * (metal_sheet_thickness + distance_between_metal_sheets)
                         + (metal_sheet_thickness + distance_between_metal_sheets) / 2 ] * 4 

    # Create mesh and subdomains
    mesh, mu = create_mesh_and_subdomains(length_of_domain, width_of_domain, height_of_domain, mu_air,
                               num_electromagnets_length, num_electromagnets_width, electromagnet_radius, electromagnet_height, mu_electromagnet, 
                               num_metal_sheets, metal_sheet_length, metal_sheet_width, metal_sheet_thickness, mu_metal_sheet, distance_between_metal_sheets)

    # Calculate current magnitude based on parameters
    electromagnet_area = fe.pi * electromagnet_radius**2  # Cross-sectional area of an electromagnet
    desired_magnetic_field = 1.0  # Desired magnetic field strength (T)
    current_magnitude = desired_magnetic_field / (mu_electromagnet * electromagnet_area)
    J = CurrentDensity(length_of_domain, width_of_domain, num_electromagnets_length,
                    num_electromagnets_width, electromagnet_radius, current_magnitude, degree=1)
    
    nedelec_first_kind, weak_form_lhs, weak_form_rhs, v_test, u_trial = define_weak_form(mesh, mu, J)
    
    
    # Boundary Conditions
    def boundary_boolean_function(x, on_boundary):
        return on_boundary

    homogeneous_dirichlet_boundary_condition = fe.DirichletBC(
        nedelec_first_kind,
        fe.Constant(0.0),
        boundary_boolean_function,
    )

    # Finite Element Assembly and Linear System solve
    b_solution = fe.Function(nedelec_first_kind)
    fe.solve(
        weak_form_lhs == weak_form_rhs,
        b_solution,
        homogeneous_dirichlet_boundary_condition,
    )

    # Compute the magnetic force in the z-direction
    f_z = compute_magnetic_force(mesh, b_solution, mu)

    # Visualize
    c = fe.plot(b_solution, mode="color")
    plt.colorbar(c)
    fe.plot(mesh)

    plt.show()

if __name__ == "__main__":
    main()