import fenics as fe

def create_mesh_and_subdomains(length_of_domain, width_of_domain, height_of_domain, mu_air,
                               num_electromagnets_length, num_electromagnets_width, electromagnet_radius, electromagnet_height, mu_electromagnet, 
                               num_metal_sheets, metal_sheet_length, metal_sheet_width, metal_sheet_thickness, mu_metal_sheet, distance_between_metal_sheets):
    mesh = fe.BoxMesh(fe.Point(0, 0, 0), fe.Point(length_of_domain, width_of_domain, height_of_domain), 50, 50, 50)

    # Define subdomains
    domain = fe.MeshFunction("size_t", mesh, 3, 0)

    # Define electromagnet subdomain
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

    z_center = height_of_domain / 2
    for i in range(num_electromagnets_length):
        for j in range(num_electromagnets_width):
            center_x = (i - (num_electromagnets_length - 1) / 2) * electromagnet_radius * 2 + length_of_domain / 2
            center_y = (j - (num_electromagnets_width - 1) / 2) * electromagnet_radius * 2 + width_of_domain / 2
            electromagnet = ElectromagnetDomain(center_x, center_y, electromagnet_radius, electromagnet_height, z_center)
            electromagnet.mark(domain, 1)

    # Define metal sheet regions
    class MetalSheetDomain(fe.SubDomain):
        def __init__(self, length, width, thickness, metal_sheet_position_center, domain_length, domain_width, domain_height):
            super().__init__()
            self.length = length
            self.width = width
            self.thickness = thickness
            self.metal_sheet_position_center = metal_sheet_position_center
            self.domain_length = domain_length
            self.domain_width = domain_width
            self.domain_height = domain_height
        
        def inside(self, x, on_boundary):
            x_position = (self.domain_length - self.length) / 2 <= x[0] <= (self.domain_length + self.length) / 2
            y_position = (self.domain_width - self.width) / 2 <= x[1] <= (self.domain_width + self.width) / 2
            z_position = self.metal_sheet_position_center - self.thickness / 2 <= x[2] <= self.metal_sheet_position_center + self.thickness / 2
            return x_position and y_position and z_position
        
    for i in range(num_metal_sheets):
        metal_sheet_position_center = (height_of_domain - electromagnet_height) / 2 - (distance_between_metal_sheets + metal_sheet_thickness) / 2 - i * (distance_between_metal_sheets + metal_sheet_thickness)
        metal_sheet = MetalSheetDomain(metal_sheet_length, metal_sheet_width, metal_sheet_thickness, metal_sheet_position_center, length_of_domain, width_of_domain, height_of_domain)
        metal_sheet.mark(domain, 2)

    # Define air regions
    class AirDomain(fe.SubDomain):
        def inside(self, x, on_boundary):
            return True  # Default to air everywhere else
        
    air = AirDomain()
    air.mark(domain, 3)

    # Define magnetic permeability for each subdomain
    mu = fe.Function(fe.FunctionSpace(mesh, "DG", 0))
    mu.vector()[:] = fe.interpolate(fe.Constant(mu_air), mu.function_space()).vector()
    mu.vector()[domain.array() == 1] = mu_electromagnet
    mu.vector()[domain.array() == 2] = mu_metal_sheet

    return mesh, mu