import numpy as np

# Problem 2 & 3
N = 101
V_jacobi = np.zeros((N, N))
V_jacobi[-1, :] = 1.0  # Bottom wall
V_jacobi[:, -1] = 1.0  # Right wall

tolerance = 1e-6
max_iter = 50000
iterations_jacobi = 0
V_new = V_jacobi.copy()

for k in range(max_iter):
    V_new[1:-1, 1:-1] = 0.25 * (V_jacobi[0:-2, 1:-1] + V_jacobi[2:, 1:-1] + 
                                V_jacobi[1:-1, 0:-2] + V_jacobi[1:-1, 2:])
    diff = np.max(np.abs(V_new - V_jacobi))
    V_jacobi[:] = V_new[:]
    iterations_jacobi += 1
    if diff <= tolerance:
        break

print(f"Jacobi method took {iterations_jacobi} iterations to converge.")

# Problem 4
V_gs = np.zeros((N, N))
V_gs[-1, :] = 1.0  # Bottom wall
V_gs[:, -1] = 1.0  # Right wall
iterations_gs = 0
for k in range(max_iter):
    max_diff = 0.0
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

# Problem 5
V_cap = np.zeros((N, N))
V_new_cap = np.zeros((N, N))
plate_x_start = 30
plate_x_end = 70
plate1_y = 35
plate2_y = 65
V_cap[plate1_y, plate_x_start:plate_x_end+1] = 1.0
V_cap[plate2_y, plate_x_start:plate_x_end+1] = -1.0
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
