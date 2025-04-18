import fenics as fe

def define_weak_formulation(mesh, mu, J, effective_conductivity, sheet_positions, sheet_thickness):
    """
    Define the weak formulation for the magnetic vector potential A, including subdomains and boundary conditions.

    Parameters:
        mesh (Mesh): The computational mesh.
        mu (Function): Magnetic permeability (subdomain-specific).
        J (Expression): Current density.
        effective_conductivity (float): Effective conductivity of the metal sheets.
        sheet_positions (list): Z-positions of the metal sheets.
        sheet_thickness (float): Thickness of the metal sheets.

    Returns:
        tuple: (FunctionSpace, weak_form, test_function, trial_function)
    """
    # Define function space for the magnetic vector potential A
    V = fe.FunctionSpace(mesh, "Nedelec 1st kind H(curl)", 1)

    # Trial and test functions
    A = fe.TrialFunction(V)
    v = fe.TestFunction(V)

    # Magnetic field B = curl(A)
    B = fe.curl(A)

    # Ensure J is a vector field compatible with v
    if J.ufl_shape != v.ufl_shape:
        raise ValueError(f"Shape mismatch: J has shape {J.ufl_shape}, but v has shape {v.ufl_shape}")

    # Ensure A is a vector field
    if A.ufl_shape != (3,):
        raise ValueError("Magnetic vector potential A must be a vector field.")

    # Weak formulation for the magnetic vector potential
    weak_form = (1 / mu) * fe.inner(fe.curl(A), fe.curl(v)) * fe.dx - fe.inner(J, v) * fe.dx

    # Add metal sheets as regions with reduced conductivity
    subdomains = fe.MeshFunction("size_t", mesh, 3, 0)
    for i, z in enumerate(sheet_positions):
        sheet_region = fe.CompiledSubDomain("near(x[2], z, tol)", z=z, tol=sheet_thickness / 2)
        sheet_region.mark(subdomains, i + 1)

    dx = fe.Measure("dx", domain=mesh, subdomain_data=subdomains)
    for i, z in enumerate(sheet_positions):
        weak_form += effective_conductivity * fe.inner(fe.curl(A), fe.curl(v)) * dx(i + 1)

    # Apply boundary conditions using a penalty method
    penalty = 1e10  # Large penalty parameter
    boundary = fe.CompiledSubDomain("on_boundary")
    ds = fe.Measure("ds", domain=mesh, subdomain_data=subdomains)
    weak_form += penalty * fe.inner(A, v) * ds

    return V, weak_form, v, A


def compute_magnetic_force(mesh, A, mu):
    """
    Compute the magnetic force in the z-direction using the weak formulation.

    Parameters:
        mesh (Mesh): The computational mesh.
        A (Function): The solved magnetic vector potential.
        mu (Expression or Constant): Magnetic permeability.

    Returns:
        Function: Magnetic force in the z-direction.
    """
    # Magnetic field B = curl(A)
    B = fe.curl(A)

    # Ensure B is a valid vector field
    if B.ufl_shape != (3,):
        raise ValueError("Magnetic field B must be a vector field derived from the curl of A.")

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