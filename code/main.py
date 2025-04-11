from dolfin import *
from physics import define_weak_formulation, compute_magnetic_force
from solver import solve_weak_formulation

# Define the computational domain and mesh
length = 1.0  # Length of the domain (meters)
width = 1.0   # Width of the domain (meters)
mesh = RectangleMesh(Point(0, 0), Point(length, width), 50, 50)

# Define physical parameters
mu = Constant(4 * pi * 1e-7)  # Magnetic permeability (H/m)
current_magnitude = 1e6       # Current density magnitude (A/m^2)
magnet_spacing = 0.1          # Spacing between magnets (meters)

# Define the current density J (alternating polarity array)
class CurrentDensity(UserExpression):
    def eval(self, values, x):
        magnet_index = int(x[0] // magnet_spacing)
        if magnet_index % 2 == 0:
            values[0] = 0.0  # Jx
            values[1] = 0.0  # Jy
            values[2] = current_magnitude  # Jz
        else:
            values[0] = 0.0
            values[1] = 0.0
            values[2] = -current_magnitude
    def value_shape(self):
        return (3,)

J = CurrentDensity(degree=1)

# Define the weak formulation
V, weak_form, v, A = define_weak_formulation(mesh, mu, J)

# Solve the weak formulation
A_solution = solve_weak_formulation(V, weak_form)

# Compute the magnetic force in the z-direction
f_z = compute_magnetic_force(mesh, A_solution, mu)

# Save the results for visualization
File("magnetic_vector_potential.pvd") << A_solution
File("magnetic_force_z.pvd") << f_z

print("Simulation complete. Results saved to files.")