import fenics as fe
import numpy as np
from geometry import create_mesh_and_subdomains

def generate_current_density(mesh, length_of_domain, width_of_domain,
                             num_electromagnets_length, num_electromagnets_width,
                             electromagnet_radius, electromagnet_height, current_magnitude, height_of_domain):
    """
    Generate the current density J, considering the position of the electromagnets in the mesh.
    """
    # Scalar space (used to create component functions)
    V_scalar = fe.FunctionSpace(mesh, 'CG', 1)
    coords = V_scalar.tabulate_dof_coordinates().reshape((-1, 3))

    # Compute all magnet centers
    z_center = height_of_domain / 2  # Center height of the electromagnets
    centers = np.array([
        [
            (i - (num_electromagnets_length - 1) / 2) * electromagnet_radius * 2 + length_of_domain / 2,
            (j - (num_electromagnets_width - 1) / 2) * electromagnet_radius * 2 + width_of_domain / 2,
            z_center
        ]
        for i in range(num_electromagnets_length)
        for j in range(num_electromagnets_width)
    ])

    # Vectorized computation of Jz
    Jz_vals = np.zeros(coords.shape[0])
    for cx, cy, cz in centers:
        dx = coords[:, 0] - cx
        dy = coords[:, 1] - cy
        dz = coords[:, 2] - cz
        mask = (dx**2 + dy**2 <= electromagnet_radius**2) & (abs(dz) <= electromagnet_height / 2)
        polarity = 1 if (cx + cy) % 2 == 0 else -1
        Jz_vals[mask] = polarity * current_magnitude

    # Create the vector function for J
    V_J = fe.VectorFunctionSpace(mesh, "CG", 1)
    J = fe.Function(V_J)
    J.vector().set_local(np.hstack([np.zeros_like(Jz_vals), np.zeros_like(Jz_vals), Jz_vals]))
    J.vector().apply("insert")

    return J


def define_weak_form(mesh, domain, mu_air, mu_electromagnet, mu_metal_sheet, J):
    """
    Define the weak formulation for the magnetic vector potential A, using a fe.Function to handle mu as a spatially varying coefficient.
    """
    # Create a function space for the weak formulation
    nedelec_first_kind = fe.FunctionSpace(mesh, "Nedelec 1st kind H(curl)", 1)

    # Trial and Test Functions
    u_trial = fe.TrialFunction(nedelec_first_kind)
    v_test = fe.TestFunction(nedelec_first_kind)

    # Define mu as a spatially varying coefficient using a fe.Function
    mu = fe.Function(fe.FunctionSpace(mesh, "DG", 0))
    mu_values = mu.vector().get_local()
    domain_array = domain.array()
    mu_values[domain_array == 0] = mu_air
    mu_values[domain_array == 1] = mu_electromagnet
    mu_values[domain_array == 2] = mu_metal_sheet
    mu.vector().set_local(mu_values)
    mu.vector().apply("insert")

    # Weak formulation for the magnetic vector potential
    weak_form_lhs = (1 / mu) * fe.inner(fe.curl(u_trial), fe.curl(v_test)) * fe.dx
    weak_form_rhs = fe.inner(J, v_test) * fe.dx  # Incorporate current density J as the source term

    return nedelec_first_kind, weak_form_lhs, weak_form_rhs

def compute_magnetic_force(mesh, b_solution, domain, mu_air, mu_electromagnet, mu_metal_sheet):
    """
    Compute the magnetic force in the z-direction, considering different mu values for each material.
    """

    # Define a piecewise function for mu based on the domain
    mu_values = fe.Function(fe.FunctionSpace(mesh, "DG", 0))
    mu_array = mu_values.vector().get_local()
    domain_array = domain.array()
    mu_array[domain_array == 0] = mu_air
    mu_array[domain_array == 1] = mu_electromagnet
    mu_array[domain_array == 2] = mu_metal_sheet
    mu_values.vector().set_local(mu_array)
    mu_values.vector().apply("insert")

    # Compute the magnetic field B = curl(A)
    B = fe.curl(b_solution)

    # Ensure B is projected onto a continuous function space for accurate visualization
    V_vector = fe.VectorFunctionSpace(mesh, "CG", 1)
    B_projected = fe.project(B, V_vector)

    # Components of the Maxwell stress tensor
    Bx, By, Bz = B_projected[0], B_projected[1], B_projected[2]
    T_xz = (1 / mu_values) * Bx * Bz
    T_yz = (1 / mu_values) * By * Bz
    T_zz = (1 / mu_values) * Bz**2 - (1 / (2 * mu_values)) * (Bx**2 + By**2 + Bz**2)

    # Magnetic force in the z-direction
    f_z = fe.div(fe.as_vector([T_xz, T_yz, T_zz]))

    # Project the force onto a function space for visualization
    V_scalar = fe.FunctionSpace(mesh, "CG", 1)
    f_z_projected = fe.project(f_z, V_scalar)

    return f_z_projected

def calculate_solution(resolution, length_of_domain, width_of_domain, height_of_domain, mu_air,
                       num_electromagnets_length, num_electromagnets_width, electromagnet_radius,
                       electromagnet_height, mu_electromagnet, num_metal_sheets, metal_sheet_length,
                       metal_sheet_width, metal_sheet_thickness, mu_metal_sheet, distance_between_metal_sheets,
                       current_magnitude):

    # Create mesh and subdomains
    mesh, mu, domain = create_mesh_and_subdomains(
        length_of_domain, width_of_domain, height_of_domain, mu_air,
        num_electromagnets_length, num_electromagnets_width, electromagnet_radius,
        electromagnet_height, mu_electromagnet, num_metal_sheets, metal_sheet_length,
        metal_sheet_width, metal_sheet_thickness, mu_metal_sheet, distance_between_metal_sheets,
        resolution=resolution
    )

    # Generate current density
    J = generate_current_density(
        mesh, length_of_domain, width_of_domain,
        num_electromagnets_length, num_electromagnets_width,
        electromagnet_radius, electromagnet_height, current_magnitude, height_of_domain
    )

    # Define the weak formulation
    nedelec_first_kind, weak_form_lhs, weak_form_rhs = define_weak_form(
        mesh, domain, mu_air, mu_electromagnet, mu_metal_sheet, J
    )

    # Define the homogeneous Dirichlet boundary condition
    def boundary_boolean_function(x, on_boundary):
        return on_boundary and (
            fe.near(x[0], 0) or fe.near(x[0], length_of_domain) or
            fe.near(x[1], 0) or fe.near(x[1], width_of_domain) or
            fe.near(x[2], 0) or fe.near(x[2], height_of_domain)
        )

    homogeneous_dirichlet_boundary_condition = fe.DirichletBC(
        nedelec_first_kind,
        fe.Constant((0.0, 0.0, 0.0)),  # Zero vector for homogeneous Dirichlet condition
        boundary_boolean_function,
    )


    # Solve the system using a robust direct solver (MUMPS)
    a_solution = fe.Function(nedelec_first_kind)
    problem = fe.LinearVariationalProblem(
        weak_form_lhs, weak_form_rhs, a_solution, homogeneous_dirichlet_boundary_condition
    )
    solver = fe.LinearVariationalSolver(problem)

    # Configure the solver for robustness and consistency
    solver.parameters["linear_solver"] = "mumps"  # Use MUMPS direct solver
    solver.parameters["preconditioner"] = "none"  # No preconditioner needed for direct solvers
    solver.parameters["krylov_solver"]["absolute_tolerance"] = 1e-12  # High precision
    solver.parameters["krylov_solver"]["relative_tolerance"] = 1e-12  # High precision
    solver.parameters["krylov_solver"]["maximum_iterations"] = 10000  # Ensure sufficient iterations
    solver.parameters["krylov_solver"]["monitor_convergence"] = True  # Monitor convergence

    solver.solve()

    return mesh, nedelec_first_kind, a_solution, mu, domain