import nbformat
import sys

nb_path = r"D:\Documents\SFU\Phys395-ComputationalPhysics\computational-physics\Week7\Guide7_Ahilan_kumaresan.ipynb"

# Read notebook
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# We want to replace everything from cell index 2 onwards.
# Cell 0: imports
# Cell 1: ## Task 1:
# We will keep cells 0 and 1, and clear the rest.
nb.cells = nb.cells[:2]

# Add Problem 1 (Analysis)
md1 = """1. Analysis:
Laplace's equation in 2D is:
$$ \\frac{\\partial^2 V}{\\partial x^2} + \\frac{\\partial^2 V}{\\partial y^2} = 0 $$

Using the central finite-difference approximation for the second derivatives with grid spacing $h$:
$$ \\frac{\\partial^2 V}{\\partial x^2} \\approx \\frac{V_{i-1,j} - 2V_{i,j} + V_{i+1,j}}{h^2} $$
$$ \\frac{\\partial^2 V}{\\partial y^2} \\approx \\frac{V_{i,j-1} - 2V_{i,j} + V_{i,j+1}}{h^2} $$

Substituting these into Laplace's equation:
$$ \\frac{V_{i-1,j} - 2V_{i,j} + V_{i+1,j}}{h^2} + \\frac{V_{i,j-1} - 2V_{i,j} + V_{i,j+1}}{h^2} = 0 $$

Multiply both sides by $h^2$:
$$ V_{i-1,j} - 2V_{i,j} + V_{i+1,j} + V_{i,j-1} - 2V_{i,j} + V_{i,j+1} = 0 $$

Rearranging to solve for $V_{i,j}$:
$$ 4V_{i,j} = V_{i-1,j} + V_{i+1,j} + V_{i,j-1} + V_{i,j+1} $$
$$ V_{i,j} = \\frac{1}{4} \\left( V_{i-1,j} + V_{i+1,j} + V_{i,j-1} + V_{i,j+1} \\right) $$
Which is the required finite-difference equation for the potential at grid point $(i, j)$."""
nb.cells.append(nbformat.v4.new_markdown_cell(md1))

# Add Problem 2 & 3 (Jacobi + Heatmap)
md2 = "2. & 3. Jacobi update formula and heatmap:"
nb.cells.append(nbformat.v4.new_markdown_cell(md2))

code2 = """# Boundary conditions: V=1 along bottom and right, V=0 along top and left.
# Initial guess for interior = 0.
N = 101
V_jacobi = np.zeros((N, N))

# Setting boundary conditions
# Using standard (row, col) indexing where row=0 is top, row=-1 is bottom
# col=0 is left, col=-1 is right
V_jacobi[-1, :] = 1.0  # Bottom wall
V_jacobi[:, -1] = 1.0  # Right wall

tolerance = 1e-6
max_iter = 50000
iterations_jacobi = 0

V_new = V_jacobi.copy()

for k in range(max_iter):
    # Update all interior points simultaneously (Jacobi)
    V_new[1:-1, 1:-1] = 0.25 * (V_jacobi[0:-2, 1:-1] + V_jacobi[2:, 1:-1] + 
                                V_jacobi[1:-1, 0:-2] + V_jacobi[1:-1, 2:])
    
    # Calculate difference
    diff = np.max(np.abs(V_new - V_jacobi))
    
    V_jacobi[:] = V_new[:]
    iterations_jacobi += 1
    
    if diff <= tolerance:
        break

print(f"Jacobi method took {iterations_jacobi} iterations to converge.")

plt.figure(figsize=(6, 5))
# origin='upper' is default for imshow, meaning row 0 is at top
plt.imshow(V_jacobi, cmap='hot', interpolation='none')
plt.colorbar(label='Potential (V)')
plt.title('Potential (Jacobi Method)')
plt.xlabel('x index (j)')
plt.ylabel('y index (i)')
plt.show()"""
nb.cells.append(nbformat.v4.new_code_cell(code2))

# Add Problem 4 (Gauss-Seidel + Heatmap)
md3 = "4. Gauss-Seidel update formula and heatmap:"
nb.cells.append(nbformat.v4.new_markdown_cell(md3))

code3 = """V_gs = np.zeros((N, N))

# Boundary conditions
V_gs[-1, :] = 1.0  # Bottom wall
V_gs[:, -1] = 1.0  # Right wall

iterations_gs = 0

for k in range(max_iter):
    max_diff = 0.0
    
    # Update interior points sequentially (Gauss-Seidel)
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            old_val = V_gs[i, j]
            V_gs[i, j] = 0.25 * (V_gs[i - 1, j] + V_gs[i + 1, j] + V_gs[i, j - 1] + V_gs[i, j + 1])
            
            diff = abs(V_gs[i, j] - old_val)
            if diff > max_diff:
                max_diff = diff
                
    iterations_gs += 1
    
    if max_diff <= tolerance:
        break

print(f"Gauss-Seidel method took {iterations_gs} iterations to converge.")

plt.figure(figsize=(6, 5))
plt.imshow(V_gs, cmap='hot', interpolation='none')
plt.colorbar(label='Potential (V)')
plt.title('Potential (Gauss-Seidel Method)')
plt.xlabel('x index (j)')
plt.ylabel('y index (i)')
plt.show()"""
nb.cells.append(nbformat.v4.new_code_cell(code3))

# Add Problem 5 & 6 (Capacitor)
md4 = "5. & 6. Parallel-plate capacitor potential:"
nb.cells.append(nbformat.v4.new_markdown_cell(md4))

code4 = """# Grounded box (all boundaries = 0)
V_cap = np.zeros((N, N))
V_new_cap = np.zeros((N, N))

# From the diagram's text, it is likely the plates have a length of 40 grid points (4 cm on 10cm box)
# and are separated by 30 grid points (3 cm on 10cm box).
# We place them symmetrically in the 101x101 grid.
plate_x_start = 30
plate_x_end = 70
plate1_y = 35
plate2_y = 65

# Set boundary conditions for plates
V_cap[plate1_y, plate_x_start:plate_x_end+1] = 1.0   # +1 V plate
V_cap[plate2_y, plate_x_start:plate_x_end+1] = -1.0  # -1 V plate
V_new_cap[:] = V_cap[:]

iterations_cap = 0

for k in range(max_iter):
    V_new_cap[1:-1, 1:-1] = 0.25 * (V_cap[0:-2, 1:-1] + V_cap[2:, 1:-1] + 
                                    V_cap[1:-1, 0:-2] + V_cap[1:-1, 2:])
    
    # Enforce plate potentials continuously as they act as internal boundaries
    V_new_cap[plate1_y, plate_x_start:plate_x_end+1] = 1.0
    V_new_cap[plate2_y, plate_x_start:plate_x_end+1] = -1.0
    
    diff = np.max(np.abs(V_new_cap - V_cap))
    V_cap[:] = V_new_cap[:]
    iterations_cap += 1
    
    if diff <= tolerance:
        break

print(f"Jacobi method for capacitor took {iterations_cap} iterations to converge.")

plt.figure(figsize=(6, 5))
plt.imshow(V_cap, cmap='seismic', interpolation='none', vmin=-1, vmax=1)
plt.colorbar(label='Potential (V)')
plt.title('Potential of Parallel-Plate Capacitor')
plt.xlabel('x index (j)')
plt.ylabel('y index (i)')
plt.show()"""
nb.cells.append(nbformat.v4.new_code_cell(code4))

# Add Problem 7 (Analysis)
md7 = """7. Analysis:
For an ideal *infinite* parallel-plate capacitor, the electric field lines are perfectly uniform and straight between the plates resulting in equipotential lines that are parallel to the plates. Outside the region between the plates, the potential would be zero (or constant).
However, in this finite numerical solution, we can clearly see the "fringing fields"—the equipotential lines curve outwards near the edges of the plates. Furthermore, because the entire setup is enclosed in a grounded box (V=0 at the boundaries), the potential smoothly transitions to zero at the walls rather than remaining strictly uniform or zero everywhere outside the plates."""
nb.cells.append(nbformat.v4.new_markdown_cell(md7))

# Write notebook back
with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Notebook updated successfully.")
