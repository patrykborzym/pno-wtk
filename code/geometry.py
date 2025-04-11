import fenics as fe

def create_mesh_and_subdomains(length, width, height, mu_electromagnet, mu_air, current_magnitude,
                                effective_conductivity, metal_sheet_thickness, metal_sheet_positions,
                                num_electromagnets, electromagnet_radius, electromagnet_height):
    """
    Create the computational mesh, define subdomains for electromagnets and air,
    and set up physical parameters.

    Returns:
        tuple: (mesh, mu, J, V)
    """
    # Define the computational domain and mesh
    mesh = fe.BoxMesh(fe.Point(0, 0, 0), fe.Point(length, width, height), 20, 20, 20)

    # Define subdomains for electromagnets
    electromagnet_domains = fe.MeshFunction("size_t", mesh, 3, 0)

    class ElectromagnetDomain(fe.SubDomain):
        def __init__(self, center_x, center_y, radius, height, z_center):
            super().__init__()
            self.center_x = center_x
            self.center_y = center_y
            self.radius = radius
            self.height = height
            self.z_center = z_center

        def inside(self, x, on_boundary):
            r = ((x[0] - self.center_x)**2 + (x[1] - self.center_y)**2)**0.5
            return r <= self.radius and abs(x[2] - self.z_center) <= self.height / 2

    # Create a 6x6 grid of electromagnets
    z_center = height / 2
    for i in range(num_electromagnets):
        for j in range(num_electromagnets):
            center_x = (i - (num_electromagnets - 1) / 2) * electromagnet_radius * 2 + length / 2
            center_y = (j - (num_electromagnets - 1) / 2) * electromagnet_radius * 2 + width / 2
            electromagnet = ElectromagnetDomain(center_x, center_y, electromagnet_radius, electromagnet_height, z_center)
            electromagnet.mark(electromagnet_domains, 1)

    # Define air regions
    class AirDomain(fe.SubDomain):
        def inside(self, x, on_boundary):
            return True  # Default to air everywhere else

    air = AirDomain()
    air.mark(electromagnet_domains, 2)

    # Define magnetic permeability for each subdomain
    mu = fe.Function(fe.FunctionSpace(mesh, "DG", 0))
    mu.vector()[:] = fe.interpolate(fe.Constant(mu_air), mu.function_space()).vector()
    mu.vector()[electromagnet_domains.array() == 1] = mu_electromagnet

    # Define the current density J
    class CurrentDensity(fe.UserExpression):
        def eval(self, values, x):
            electromagnet_spacing = length / num_electromagnets
            i = int(x[0] // electromagnet_spacing)
            j = int(x[1] // electromagnet_spacing)
            if 0 <= i < num_electromagnets and 0 <= j < num_electromagnets:
                polarity = 1 if (i + j) % 2 == 0 else -1
                values[0] = 0.0  # Jx
                values[1] = 0.0  # Jy
                values[2] = polarity * current_magnitude  # Jz
            else:
                values[0] = values[1] = values[2] = 0.0
        def value_shape(self):
            return (3,)

    J = CurrentDensity(degree=1)

    # Create a function space for the weak formulation
    V = fe.FunctionSpace(mesh, "Nedelec 1st kind H(curl)", 1)

    return mesh, mu, J, V
