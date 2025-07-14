import numpy as np
import pandas as pd

grid_size = 5
domain_size = 1.0
samples_per_cell = 500
cell_size = domain_size / grid_size
num_circles = 1

def gen_volumefracs(circle_center, circle_radius):
    vf = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            x0 = i * cell_size
            y0 = j * cell_size
            xs = np.random.uniform(x0, x0 + cell_size, samples_per_cell)
            ys = np.random.uniform(y0, y0 + cell_size, samples_per_cell)
            inside = ((xs - circle_center[0])**2 + (ys - circle_center[1])**2) <= circle_radius**2
            vf[j, i] = np.sum(inside) / samples_per_cell
    return vf


def get_interface_cells(vf, eps=1e-6):
    interface_cells = []
    grid_size = vf.shape[0]
    for j in range(grid_size):
        for i in range(grid_size):
            f = vf[j, i]
            if eps < f < 1.0 - eps:  # stricter interface check
                interface_cells.append((i, j))
    return interface_cells

#computes normals at teh interface by first selecting the cells on the interface
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

"""#computing normals using height functions

def comp_normals(vf, cell_size):
    grid_size = vf.shape[0]
    normals = {}

    # Column-wise centroids for dh/dx
    h_x = np.full(grid_size, np.nan)
    for i in range(grid_size):
        col = vf[:, i]
        col_sum = np.sum(col)
        if 0.0 < col_sum < grid_size:
            y_indices = np.arange(grid_size)
            h_x[i] = np.sum(col * y_indices) / col_sum

    #row-wise centroids for dh/dy
    h_y = np.full(grid_size, np.nan)
    for j in range(grid_size):
        row = vf[j, :]
        row_sum = np.sum(row)
        if 0.0 < row_sum < grid_size:
            x_indices = np.arange(grid_size)
            h_y[j] = np.sum(row * x_indices) / row_sum

    for j in range(2, grid_size - 2):
        for i in range(2, grid_size - 2):
            f = vf[j, i]
            if 0.0001 < f < 0.9999:  # Interface cell
                valid_dx = not np.isnan(h_x[i-2:i+3]).any()
                valid_dy = not np.isnan(h_y[j-2:j+3]).any()

                if not (valid_dx or valid_dy):
                    continue  # Neither derivative computable

                dhdx = 0.0
                dhdy = 0.0

                if valid_dx:
                    dhdx = (-h_x[i+2] + 8*h_x[i+1] - 8*h_x[i-1] + h_x[i-2]) / (12 * cell_size)

                if valid_dy:
                    dhdy = (-h_y[j+2] + 8*h_y[j+1] - 8*h_y[j-1] + h_y[j-2]) / (12 * cell_size)

                nx, ny = -dhdx, -dhdy
                norm = np.hypot(nx, ny)
                if norm > 0:
                    normals[(i, j)] = (nx / norm, ny / norm)

    return normals
""" 
"""#computing normals using 5 point differentiation
def comp_normals(vf, cell_size):
    grid_size = vf.shape[0]
    normals = {}
    interface_heights = np.full(grid_size, np.nan)
    for i in range(grid_size):
        column = vf[:, i]
        column_sum = np.sum(column)
        if 0 < column_sum < grid_size:  # There's an interface in this column
            y_indices = np.arange(grid_size)
            centroid = np.sum(column * y_indices) / column_sum
            interface_heights[i] = centroid * cell_size  # convert to physical y-position
    for i in range(2, grid_size - 2):
        window = interface_heights[i - 2:i + 3]
        if not np.isnan(window).any():
            # Central difference (5-point stencil)
            dhdx = (-window[4] + 8 * window[3] - 8 * window[1] + window[0]) / (12 * cell_size)

            # Assign normal to interface cells in the column
            for j in range(1, grid_size - 1):
                f = vf[j, i]
                if 0.0001 < f < 0.9999:
                    nx = -dhdx
                    ny = 1.0
                    norm = np.hypot(nx, ny)
                    normals[(i, j)] = (nx / norm, ny / norm)

    return normals


"""

## Dataset generation
dataset = []
for idx in range(num_circles):
    circle_center = (0.5, 0.5)#(np.random.uniform(0.1, 0.7), np.random.uniform(0.1, 0.7))
    circle_radius = 0.3#np.random.uniform(0.05, 0.3)
    vf = gen_volumefracs(circle_center, circle_radius)
    normals = comp_normals(vf, cell_size)

    for (i, j), (nx, ny) in normals.items():
        data_point = {
            'circle_id': idx,
            'center_x': circle_center[0],
            'center_y': circle_center[1],
            'radius': circle_radius,
            'i': i, 'j': j,
            **{f'vf_{r}{c}': vf[r, c] for r in range(5) for c in range(5)},
            'nx': nx, 'ny': ny
        }
        dataset.append(data_point)

# Save dataset
df = pd.DataFrame(dataset)
df.to_csv('vf_new gen.csv', index=False)
print("dataframe saved!.")

