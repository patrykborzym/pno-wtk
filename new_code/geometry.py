import fenics as fe
from plot import plot_2d_domain_markings_and_mu, mu_values_visualisation

# Unified subdomain marker
class MaterialSubdomain(fe.SubDomain):
    def __init__(self, electromagnet_centers, electromagnet_radius, electromagnet_height,
                 metal_sheet_params, domain_height, length_of_domain, width_of_domain):
        super().__init__()
        self.e_centers = electromagnet_centers
        self.e_radius = electromagnet_radius
        self.e_height = electromagnet_height
        self.domain_height = domain_height
        self.length_of_domain = length_of_domain
        self.width_of_domain = width_of_domain
        self.metal_sheet_params = metal_sheet_params  # tuple of (num, length, width, thickness, distance)

    def inside(self, x, on_boundary):
        # Electromagnets (marker 1)
        for cx, cy in self.e_centers:
            if (x[0]-cx)**2 + (x[1]-cy)**2 <= self.e_radius**2 and \
               abs(x[2] - self.domain_height / 2) <= self.e_height / 2:
                return 1

        # Metal Sheets (markers 2, 3, 4, ...)
        num_sheets, m_length, m_width, m_thick, m_dist = self.metal_sheet_params
        for i in range(num_sheets):
            z_center = (self.domain_height - self.e_height) / 2 - \
                       (m_dist + m_thick)/2 - i * (m_dist + m_thick)
            if ((self.length_of_domain - m_length) / 2 <= x[0] <= (self.length_of_domain + m_length) / 2) and \
               ((self.width_of_domain - m_width) / 2 <= x[1] <= (self.width_of_domain + m_width) / 2) and \
               (z_center - m_thick/2 <= x[2] <= z_center + m_thick/2):
                return 2 + i  # Assign unique marker for each sheet

        return 0

def create_mesh_and_subdomains(length, width, height, mu_air,
                               num_em_len, num_em_wid, em_radius, em_height, mu_em,
                               num_sheets, sheet_length, sheet_width, sheet_thickness, mu_sheet,
                               sheet_spacing, resolution):

    mesh = fe.BoxMesh(fe.Point(0, 0, 0), fe.Point(length, width, height), resolution, resolution, resolution)

    subdomains = fe.MeshFunction("size_t", mesh, mesh.topology().dim())
    subdomains.set_all(0)

    # Electromagnet centers
    e_centers = [
        ((i - (num_em_len - 1)/2) * em_radius * 2 + length / 2,
         (j - (num_em_wid - 1)/2) * em_radius * 2 + width / 2)
        for i in range(num_em_len)
        for j in range(num_em_wid)
    ]

    subdomain_marker = MaterialSubdomain(
        e_centers, em_radius, em_height,
        (num_sheets, sheet_length, sheet_width, sheet_thickness, sheet_spacing),
        height, length, width
    )

    for cell in fe.cells(mesh):
        midpoint = cell.midpoint()
        marker = subdomain_marker.inside(midpoint, False)
        subdomains[cell] = marker

    class MuExpression(fe.UserExpression):
        """
        Define a spatially varying magnetic permeability (mu) as a UserExpression.
        """
        def __init__(self, subdomains, mu_air, mu_em, mu_sheet, **kwargs):
            super().__init__(**kwargs)
            self.subdomains = subdomains
            self.mu_air = mu_air
            self.mu_em = mu_em
            self.mu_sheet = mu_sheet

        def eval_cell(self, values, x, cell):
            marker = self.subdomains[cell.index]
            if marker == 1:  # Electromagnet
                values[0] = self.mu_em
            elif marker >= 2:  # Metal sheets (markings 2, 3, 4, ...)
                values[0] = self.mu_sheet
            else:  # Air
                values[0] = self.mu_air

        def value_shape(self):
            return ()

    mu_expr = MuExpression(subdomains, mu_air, mu_em, mu_sheet, degree=0)
    V0 = fe.FunctionSpace(mesh, "DG", 0)
    mu = fe.project(mu_expr, V0)

    # Call the 2D plotting function
    #mu_values_visualisation(mesh, subdomains, mu)
    #plot_2d_domain_markings_and_mu(mesh, subdomains, mu)

    return mesh, mu, subdomains