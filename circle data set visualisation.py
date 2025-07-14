import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

grid_size = 5
domain_size = 1.0
cell_size = domain_size / grid_size

df = pd.read_csv("bf_new_gen.csv")
circle_ids = df['circle_id'].unique()

for circle_id in circle_ids:
    sub_df = df[df['circle_id'] == circle_id]


    vf = np.zeros((grid_size, grid_size))
    vf_values = sub_df.iloc[0][[f'vf_{r}{c}' for r in range(grid_size) for c in range(grid_size)]].values
    vf = vf_values.reshape((grid_size, grid_size))

    center_x = sub_df['center_x'].iloc[0]
    center_y = sub_df['center_y'].iloc[0]
    radius = sub_df['radius'].iloc[0]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, domain_size)
    ax.set_ylim(0, domain_size)
    ax.set_aspect('equal')
    for i in range(grid_size):
        for j in range(grid_size):
            x0 = i * cell_size
            y0 = j * cell_size
            rect = plt.Rectangle((x0, y0), cell_size, cell_size, edgecolor='black', facecolor='none')
            ax.add_patch(rect)
            ax.text(x0 + cell_size / 2, y0 + cell_size / 2, f"{vf[j, i]:.2f}", 
                    ha='center', va='center', fontsize=9, color='black')

    for _, row in sub_df.iterrows():
        i, j = int(row['i']), int(row['j'])
        nx, ny = row['nx'], row['ny']
        x0 = i * cell_size + cell_size / 2
        y0 = j * cell_size + cell_size / 2
        ax.arrow(x0, y0, 0.05 * nx, 0.05 * ny, head_width=0.01, head_length=0.01, fc='red', ec='red')


    if center_x is not None and center_y is not None and radius is not None:
        circle_patch = plt.Circle((center_x, center_y), radius, color='Red', fill=False, linestyle='-', linewidth=1.5)
        ax.add_patch(circle_patch)

    plt.title(f"Circle ID: {circle_id}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print("all circles generated!")
#https://colab.research.google.com/drive/1o6EoYeuDCXELhqKb1l3NQF273aJ7M0j9#scrollTo=IUSGAQiUna0A