import fenics as fe
import numpy as np
from geometry import create_mesh_and_subdomains

def generate_current_density(mesh, length_of_domain, width_of_domain,
                             num_electromagnets_length, num_electromagnets_width,
                             electromagnet_radius, electromagnet_height, current_magnitude, height_of_domain):
    """
    Generate the current density J, considering the position of the electromagnets in the mesh.
    """
    # Use cell-based operations to define the current density
    V_scalar = fe.FunctionSpace(mesh, 'DG', 0)  # Use DG space for cell-based operations
    Jz = fe.Function(V_scalar)

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

    # Assign current density values cell by cell
    Jz_values = Jz.vector().get_local()
    for cell in fe.cells(mesh):
        cell_midpoint = cell.midpoint()
        for cx, cy, cz in centers:
            dx = cell_midpoint.x() - cx
            dy = cell_midpoint.y() - cy
            dz = cell_midpoint.z() - cz
            if dx**2 + dy**2 <= electromagnet_radius**2 and abs(dz) <= electromagnet_height / 2:
                polarity = 1 if (cx + cy) % 2 == 0 else -1
                Jz_values[cell.index()] = polarity * current_magnitude

    Jz.vector().set_local(Jz_values)
    Jz.vector().apply("insert")

    # Create the vector function for J
    V_J = fe.VectorFunctionSpace(mesh, "CG", 1)
    J = fe.Function(V_J)

    # Assign the z-component of the current density to the vector field
    Jz_projected = fe.project(Jz, fe.FunctionSpace(mesh, "CG", 1))  # Project Jz into CG space
    J.assign(fe.project(fe.as_vector([fe.Constant(0), fe.Constant(0), Jz_projected]), V_J))  # Use fe.project instead of fe.interpolate

    return J

class MuExpression(fe.UserExpression):
    """
    Define a spatially varying magnetic permeability (mu) as a UserExpression.
    """
    def __init__(self, domain, mu_air, mu_electromagnet, mu_metal_sheet, **kwargs):
        super().__init__(**kwargs)
        self.domain = domain
        self.mu_air = mu_air
        self.mu_electromagnet = mu_electromagnet
        self.mu_metal_sheet = mu_metal_sheet

    def eval_cell(self, values, x, cell):
        subdomain_id = self.domain[cell.index]
        if subdomain_id == 0:  # Air
            values[0] = self.mu_air
        elif subdomain_id == 1:  # Electromagnet
            values[0] = self.mu_electromagnet
        else:  # Metal sheets
            values[0] = self.mu_metal_sheet

    def value_shape(self):
        return ()

def define_weak_form(mesh, domain, mu, J):
    """
    Define the weak formulation for the magnetic vector potential A, using a simpler approach for mu.
    """
    # Create a function space for the weak formulation
    nedelec_first_kind = fe.FunctionSpace(mesh, "Nedelec 1st kind H(curl)", 1)

    # Trial and Test Functions
    u_trial = fe.TrialFunction(nedelec_first_kind)
    v_test = fe.TestFunction(nedelec_first_kind)

    # Weak formulation for the magnetic vector potential
    weak_form_lhs = (1 / mu) * fe.inner(fe.curl(u_trial), fe.curl(v_test)) * fe.dx
    weak_form_rhs = fe.inner(J, v_test) * fe.dx  # Incorporate current density J as the source term

    return nedelec_first_kind, weak_form_lhs, weak_form_rhs

def compute_magnetic_force(mesh, b_solution, domain, mu):
    """
    Compute the magnetic force in the z-direction using the Maxwell stress tensor and integrating over the surface of the metal sheet.
    """
    # Interpolate mu into a CG space for safe computations
    V1 = fe.FunctionSpace(mesh, "CG", 1)
    mu_interp = fe.interpolate(mu, V1)

    # Compute the magnetic field B = curl(A)
    B = fe.curl(b_solution)

    # Compute the Maxwell stress tensor components
    T = fe.as_tensor(
        [
            [(1 / mu_interp) * (B[i] * B[j] - (1 if i == j else 0) * 0.5 * fe.inner(B, B)) for j in range(3)]
            for i in range(3)
        ]
    )

    # Define the surface normal vector
    n = fe.FacetNormal(mesh)

    # Compute the magnetic force as the surface integral of the Maxwell stress tensor
    force_z = fe.assemble(fe.dot(T * n, fe.as_vector([0, 0, 1])) * fe.ds(subdomain_data=domain, subdomain_id=2))

    return force_z

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
        mesh, domain, mu, J
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

    # Solve the system using the MUMPS direct solver
    a_solution = fe.Function(nedelec_first_kind)
    problem = fe.LinearVariationalProblem(
        weak_form_lhs, weak_form_rhs, a_solution, homogeneous_dirichlet_boundary_condition
    )
    solver = fe.LinearVariationalSolver(problem)

    # Configure the solver for maximum robustness
    solver.parameters["linear_solver"] = "mumps"  # Use MUMPS direct solver
    solver.parameters["preconditioner"] = "none"  # No preconditioner needed for direct solvers
    solver.parameters["krylov_solver"]["absolute_tolerance"] = 1e-14  # High precision
    solver.parameters["krylov_solver"]["relative_tolerance"] = 1e-14  # High precision
    solver.parameters["krylov_solver"]["maximum_iterations"] = 10000  # Ensure sufficient iterations
    solver.parameters["krylov_solver"]["monitor_convergence"] = True  # Monitor convergence
    solver.parameters["krylov_solver"]["nonzero_initial_guess"] = False  # Start with zero initial guess

    solver.solve()

    return mesh, nedelec_first_kind, a_solution, mu, domain