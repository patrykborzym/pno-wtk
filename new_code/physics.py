import fenics as fe
import numpy as np

def generate_current_density(mesh, length_of_domain, width_of_domain,
                             num_electromagnets_length, num_electromagnets_width,
                             electromagnet_radius, current_magnitude):
    # Scalar space (used to create component functions)
    V_scalar = fe.FunctionSpace(mesh, 'CG', 1)
    coords = V_scalar.tabulate_dof_coordinates().reshape((-1, 3))

    # Compute all magnet centers
    centers = np.array([
        [
            (i - (num_electromagnets_length - 1) / 2) * electromagnet_radius * 2 + length_of_domain / 2,
            (j - (num_electromagnets_width - 1) / 2) * electromagnet_radius * 2 + width_of_domain / 2,
            i, j
        ]
        for i in range(num_electromagnets_length)
        for j in range(num_electromagnets_width)
    ])

    # Vectorized computation of Jz
    Jz_vals = np.zeros(coords.shape[0])
    for cx, cy, i, j in centers:
        dx = coords[:, 0] - cx
        dy = coords[:, 1] - cy
        mask = dx**2 + dy**2 <= electromagnet_radius**2
        polarity = 1 if (i + j) % 2 == 0 else -1
        Jz_vals[mask] = polarity * current_magnitude

    # Create the vector function for J
    V_J = fe.VectorFunctionSpace(mesh, "CG", 1)
    J = fe.Function(V_J)
    J.vector().set_local(np.hstack([np.zeros_like(Jz_vals), np.zeros_like(Jz_vals), Jz_vals]))
    J.vector().apply("insert")

    return J

def define_weak_form(mesh, mu, J):
    """
    Define the weak formulation for the magnetic vector potential A.
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

def compute_magnetic_force(mesh, b_solution, mu):
    # Magnetic field B = curl(A)
    B = fe.curl(b_solution)

    # Components of the Maxwell stress tensor
    Bx, By, Bz = B[0], B[1], B[2]
    T_xz = (1 / mu) * Bx * Bz
    T_yz = (1 / mu) * By * Bz
    T_zz = (1 / mu) * Bz**2 - (1 / (2 * mu)) * (Bx**2 + By**2 + Bz**2)

    # Magnetic force in the z-direction
    f_z = fe.div(fe.as_vector([T_xz, T_yz, T_zz]))

    # Project the force onto a function space for visualization
    V = fe.FunctionSpace(mesh, "CG", 1)
    f_z_projected = fe.project(f_z, V)

    return f_z_projected