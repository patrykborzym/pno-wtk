import fenics as fe

class CurrentDensity(fe.UserExpression):
    def __init__(self, length_of_domain, width_of_domain, num_electromagnets_length,
                 num_electromagnets_width, electromagnet_radius, current_magnitude, **kwargs):
        super().__init__(**kwargs)
        self.centers = []
        self.radius = electromagnet_radius
        self.current_magnitude = current_magnitude
        self.num_electromagnets_length = num_electromagnets_length
        self.num_electromagnets_width = num_electromagnets_width

        # Calculate electromagnet positions and store with their grid indices
        for i in range(num_electromagnets_length):
            for j in range(num_electromagnets_width):
                center_x = (i - (num_electromagnets_length - 1) / 2) * electromagnet_radius * 2 + length_of_domain / 2
                center_y = (j - (num_electromagnets_width - 1) / 2) * electromagnet_radius * 2 + width_of_domain / 2
                self.centers.append((center_x, center_y, i, j))

    def eval(self, values, x):
        values[0] = 0.0  # Jx
        values[1] = 0.0  # Jy
        values[2] = 0.0  # Jz (default)

        for cx, cy, i, j in self.centers:
            dx = x[0] - cx
            dy = x[1] - cy
            r_squared = dx**2 + dy**2
            if r_squared <= self.radius**2:
                # Alternate polarity in a checkerboard pattern
                polarity = 1 if (i + j) % 2 == 0 else -1
                # Uniform current density
                values[2] = polarity * self.current_magnitude
                break

    def value_shape(self):
        return (3,)


def define_weak_form(mesh, mu, J):
    """
    Define the weak formulation for the magnetic vector potential A.
    """
    # Create a function space for the weak formulation
    nedelec_first_kind = fe.FunctionSpace(mesh, "Nedelec 1st kind H(curl)", 1)

    # Trial and test functions
    A = fe.TrialFunction(nedelec_first_kind)
    v = fe.TestFunction(nedelec_first_kind)

    # Trial and Test Functions
    u_trial = fe.TrialFunction(nedelec_first_kind)
    v_test = fe.TestFunction(nedelec_first_kind)

    # Magnetic field B = curl(A)
    B = fe.curl(u_trial)

    # Weak formulation for the magnetic vector potential
    # weak_form = (1 / mu) * fe.inner(fe.curl(u_trial), fe.curl(v_test)) * fe.dx - fe.inner(J, v_test) * fe.dx
    weak_form_lhs = (1 / mu) * fe.inner(fe.curl(u_trial), fe.curl(v_test)) * fe.dx
    weak_form_rhs = fe.inner(J, v_test) * fe.dx

    return nedelec_first_kind, weak_form_lhs, weak_form_rhs, v_test, u_trial

def compute_magnetic_force(mesh, u_trial, mu):
    # Magnetic field B = curl(A)
    B = fe.curl(u_trial)

    # Components of the Maxwell stress tensor
    Bx, By, Bz = B[0], B[1], B[2]
    T_xz = (1 / mu) * Bx * Bz
    T_yz = (1 / mu) * By * Bz
    T_zz = (1 / mu) * Bz**2 - (1 / (2 * mu)) * (Bx**2 + By**2 + Bz**2)

    # Magnetic force in the z-direction
    f_z = -fe.div(fe.as_vector([T_xz, T_yz, T_zz]))

    # Project the force onto a function space for visualization
    V = fe.FunctionSpace(mesh, "CG", 1)
    f_z_projected = fe.project(f_z, V)

    return f_z_projected