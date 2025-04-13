import fenics as fe

def create_mesh_and_subdomains(length_of_domain, width_of_domain, height_of_domain, mu_air,
                               num_electromagnets_length, num_electromagnets_width, electromagnet_radius, electromagnet_height, mu_electromagnet, 
                               num_metal_sheets, metal_sheet_length, metal_sheet_width, metal_sheet_thickness, mu_metal_sheet, distance_between_metal_sheets):
    # Reduce mesh resolution to 30x30x30
    mesh = fe.BoxMesh(fe.Point(0, 0, 0), fe.Point(length_of_domain, width_of_domain, height_of_domain), 50, 50, 50)

    # Define subdomains
    domain = fe.MeshFunction("size_t", mesh, 3, 0)

    # Efficient marking of electromagnet subdomains
    z_center = height_of_domain / 2
    electromagnet_centers = [
        (
            (i - (num_electromagnets_length - 1) / 2) * electromagnet_radius * 2 + length_of_domain / 2,
            (j - (num_electromagnets_width - 1) / 2) * electromagnet_radius * 2 + width_of_domain / 2
        )
        for i in range(num_electromagnets_length)
        for j in range(num_electromagnets_width)
    ]

    for cell in fe.cells(mesh):
        x = cell.midpoint()
        for cx, cy in electromagnet_centers:
            r = ((x[0] - cx)**2 + (x[1] - cy)**2)**0.5
            if r <= electromagnet_radius and abs(x[2] - z_center) <= electromagnet_height / 2:
                domain[cell] = 1
                break

    # Efficient marking of metal sheet subdomains
    for i in range(num_metal_sheets):
        metal_sheet_position_center = (height_of_domain - electromagnet_height) / 2 - (distance_between_metal_sheets + metal_sheet_thickness) / 2 - i * (distance_between_metal_sheets + metal_sheet_thickness)
        for cell in fe.cells(mesh):
            x = cell.midpoint()
            if (
                (length_of_domain - metal_sheet_length) / 2 <= x[0] <= (length_of_domain + metal_sheet_length) / 2 and
                (width_of_domain - metal_sheet_width) / 2 <= x[1] <= (width_of_domain + metal_sheet_width) / 2 and
                metal_sheet_position_center - metal_sheet_thickness / 2 <= x[2] <= metal_sheet_position_center + metal_sheet_thickness / 2
            ):
                domain[cell] = 2

    # Define magnetic permeability for each subdomain
    mu = fe.Function(fe.FunctionSpace(mesh, "DG", 0))
    mu_values = mu.vector().get_local()
    domain_array = domain.array()
    mu_values[domain_array == 1] = mu_electromagnet
    mu_values[domain_array == 2] = mu_metal_sheet
    mu_values[domain_array == 0] = mu_air
    mu.vector().set_local(mu_values)
    mu.vector().apply("insert")

    return mesh, mu