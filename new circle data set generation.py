#using scalar function to define the circle and to generate the normal using the gradient of the scalar fucntion at every point
import numpy as np
import pandas as pd


# Parameters
grid_size = 5
domain_size = 1.0
samples_per_cell = 500
cell_size = domain_size / grid_size
num_circles = 10

# Cell center utility
def cell_center(i, j):
    x = (i + 0.5) * cell_size
    y = (j + 0.5) * cell_size
    return x, y

# Interface detection
def is_interface(vf):
    return (vf > 0.0) & (vf < 1.0)

def get_interface_cells(vf, eps=1e-6):
    interface_cells = []
    grid_size = vf.shape[0]
    for j in range(grid_size):
        for i in range(grid_size):
            f = vf[j, i]
            if eps < f < 1.0 - eps:
                interface_cells.append((i, j))
    return interface_cells

# Volume fraction Monte Carlo estimation
def gen_volumefracs(circle_center, circle_radius): 
    vf = np.zeros((grid_size, grid_size))
    bin_vf = np.zeros((grid_size, grid_size), dtype=int)
    
    for i in range(grid_size):
        for j in range(grid_size):
            x0 = i * cell_size
            y0 = j * cell_size
            xs = np.random.uniform(x0, x0 + cell_size, samples_per_cell)
            ys = np.random.uniform(y0, y0 + cell_size, samples_per_cell)
            inside = ((xs - circle_center[0])**2 + (ys - circle_center[1])**2) <= circle_radius**2
            vf_val = np.sum(inside) / samples_per_cell
            vf[j, i] = vf_val
            
            # Binary field logic
            if 0 < vf_val < 1:
                bin_vf[j, i] = 1 if vf_val > 0.5 else 0
            else:
                bin_vf[j, i] = int(vf_val)  # 0 or 1 as is

    return vf, bin_vf


# Compute normals from volume fraction
def comp_normals(vf, cell_size):
    interface_cells = get_interface_cells(vf)
    grid_size = vf.shape[0]
    normals = {}

    for (i, j) in interface_cells:
        if 0 < i < grid_size - 1:
            dfdx = (vf[j, i + 1] - vf[j, i - 1]) / (2 * cell_size)
        elif i == 0:
            dfdx = (vf[j, i + 1] - vf[j, i]) / cell_size
        elif i == grid_size - 1:
            dfdx = (vf[j, i] - vf[j, i - 1]) / cell_size

        if 0 < j < grid_size - 1:
            dfdy = (vf[j + 1, i] - vf[j - 1, i]) / (2 * cell_size)
        elif j == 0:
            dfdy = (vf[j + 1, i] - vf[j, i]) / cell_size
        elif j == grid_size - 1:
            dfdy = (vf[j, i] - vf[j - 1, i]) / cell_size

        nx = -dfdx
        ny = -dfdy
        norm = np.hypot(nx, ny)
        if norm > 0:
            normals[(i, j)] = (nx / norm, ny / norm)

    return normals

# Data generation loop
dataset_vf = []
dataset_bf = []

for idx in range(num_circles):
    circle_center = (np.random.uniform(0.1, 0.7), np.random.uniform(0.1, 0.7))#(0.2, 0.5)  # You can randomize
    circle_radius = np.random.uniform(0.05, 0.3)        # You can randomize
    vf, bf = gen_volumefracs(circle_center, circle_radius)
    vf_normals = comp_normals(vf, cell_size)

    for (i, j), (nx_vf, ny_vf) in vf_normals.items():
        # φ-based normal computation
        phi = np.zeros((3, 3))
        for di in range(-1, 2):
            for dj in range(-1, 2):
                x, y = cell_center(i + di, j + dj)
                phi[dj + 1, di + 1] = np.sqrt((x - circle_center[0])**2 + (y - circle_center[1])**2) - circle_radius

        dphidx = (phi[1, 2] - phi[1, 0]) / (2 * cell_size)
        dphidy = (phi[2, 1] - phi[0, 1]) / (2 * cell_size)
        grad_phi = np.array([dphidx, dphidy])
        norm_phi = np.linalg.norm(grad_phi)
        if norm_phi != 0:
            nx_phi, ny_phi = grad_phi / norm_phi
        else:
            nx_phi, ny_phi = 0.0, 0.0

        # Data point for vf dataset
        data_point_vf = {
            'circle_id': idx,
            'center_x': circle_center[0],
            'center_y': circle_center[1],
            'radius': circle_radius,
            'i': i,
            'j': j,
            'nx': nx_phi,
            'ny': ny_phi
        }

        # Data point for bf dataset
        data_point_bf = {
            'circle_id': idx,
            'center_x': circle_center[0],
            'center_y': circle_center[1],
            'radius': circle_radius,
            'i': i,
            'j': j,
            'nx': nx_phi,
            'ny': ny_phi
        }

        # Add vf and bf grid values
        for row in range(grid_size):
            for col in range(grid_size):
                data_point_vf[f'vf_{row}{col}'] = vf[row, col]
                data_point_bf[f'vf_{row}{col}'] = bf[row, col]

        dataset_vf.append(data_point_vf)
        dataset_bf.append(data_point_bf)

# Save datasets
df_vf = pd.DataFrame(dataset_vf)
df_bf = pd.DataFrame(dataset_bf)

df_vf.to_csv('vf_new_gen.csv', index=False)
df_bf.to_csv('bf_new_gen.csv', index=False)

print("VF data saved to vf_new_gen.csv.")
print("Binary VF data saved to bf_new_gen.csv.")
