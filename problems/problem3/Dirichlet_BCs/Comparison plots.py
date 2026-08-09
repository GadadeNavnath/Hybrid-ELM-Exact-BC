#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from numpy.polynomial.legendre import legval

# ==========================================================
# Figure 1 : Dirichlet Boundary Error Comparison
# ==========================================================

# ----------------------------------------------------------
# Load data
# ----------------------------------------------------------

boundary_pts = np.load("data/sweden_boundary_points.npy")

stage1 = np.loadtxt("data/stage1_solution.dat")
stage2 = np.loadtxt("data/stage2_solution.dat")

# ----------------------------------------------------------
# Boundary solutions
# ----------------------------------------------------------

N_boundary = boundary_pts.shape[0]

u_stage1 = stage1[-N_boundary:, 2]
u_stage2 = stage2[-N_boundary:, 2]

# ----------------------------------------------------------
# Exact solution
# ----------------------------------------------------------

x = boundary_pts[:, 0]
y = boundary_pts[:, 1]

u_exact = np.sin(np.pi * x) * np.sin(np.pi * y)

# ----------------------------------------------------------
# Boundary errors
# ----------------------------------------------------------

err_stage1 = np.maximum(np.abs(u_stage1 - u_exact), 1e-16)
err_stage2 = np.maximum(np.abs(u_stage2 - u_exact), 1e-16)

idx = np.arange(1, N_boundary + 1)

# ----------------------------------------------------------
# Plot
# ----------------------------------------------------------

plt.figure(figsize=(7, 4))

plt.semilogy(
    idx,
    err_stage1,
    "b-",
    linewidth=1.5,
    label="Stage 1",
)

plt.semilogy(
    idx,
    err_stage2,
    "r-",
    linewidth=1.5,
    label="Stage 2",
)

plt.xlabel("Boundary point index")
plt.ylabel("Absolute Dirichlet BC error")

plt.xlim(1, N_boundary)

plt.xticks(
    np.arange(0, N_boundary + 1, 100),
    fontsize=11,
)

plt.tick_params(labelsize=11)

plt.grid(
    True,
    which="both",
    linestyle="--",
    alpha=0.3,
)

plt.legend(
    fontsize=11,
    frameon=False,
    loc="best",
)

plt.tight_layout()

plt.savefig(
    "fig3_bnd_error1.pdf",
    dpi=600,
    bbox_inches="tight",
)

plt.show()


# ==========================================================
# Figure 2 : Dirichlet Boundary Error from Basis Expansion
# ==========================================================

# ----------------------------------------------------------
# Load points
# ----------------------------------------------------------

interior_pts = np.load("data/sweden_interior_points.npy")
boundary_pts = np.load("data/sweden_boundary_points.npy")

xi_int = interior_pts[:, 0]
yi_int = interior_pts[:, 1]

xi_b = boundary_pts[:, 0]
yi_b = boundary_pts[:, 1]

# ----------------------------------------------------------
# Scaling (same as solver)
# ----------------------------------------------------------

x_all = np.concatenate([xi_int, xi_b])
y_all = np.concatenate([yi_int, yi_b])

xmin, xmax = x_all.min(), x_all.max()
ymin, ymax = y_all.min(), y_all.max()

sx = 2.0 / (xmax - xmin)
sy = 2.0 / (ymax - ymin)


def scale_x(x):
    return sx * (x - xmin) - 1.0


def scale_y(y):
    return sy * (y - ymin) - 1.0


# ----------------------------------------------------------
# Legendre basis
# ----------------------------------------------------------

Nx = 15
Ny = 25


def legendre_matrix_2d(x, y, Nx, Ny):

    xs = scale_x(x)
    ys = scale_y(y)

    cols = []

    for i in range(Nx):

        Px = legval(xs, [0] * i + [1])

        for j in range(Ny):

            Py = legval(ys, [0] * j + [1])

            cols.append(Px * Py)

    return np.vstack(cols).T


# ----------------------------------------------------------
# Boundary basis matrix
# ----------------------------------------------------------

P_b = legendre_matrix_2d(
    xi_b,
    yi_b,
    Nx,
    Ny,
)

# ----------------------------------------------------------
# Load coefficients
# ----------------------------------------------------------

data = np.load("data/solution_coefficients.npz")

beta_stage1 = data["beta_stage1"]
beta_stage2 = data["beta_stage2"]
stage2_component = data["stage2_component"]

# ----------------------------------------------------------
# Boundary solutions
# ----------------------------------------------------------

u_stage1 = P_b @ beta_stage1

boundary_correction = stage2_component[-len(boundary_pts):]

u_stage2 = (
    P_b @ beta_stage2
    + boundary_correction
)

# ----------------------------------------------------------
# Exact solution
# ----------------------------------------------------------

u_exact = (
    np.sin(np.pi * xi_b)
    * np.sin(np.pi * yi_b)
)

# ----------------------------------------------------------
# Boundary errors
# ----------------------------------------------------------

err_stage1 = np.maximum(
    np.abs(u_stage1 - u_exact),
    1e-16,
)

err_stage2 = np.maximum(
    np.abs(u_stage2 - u_exact),
    1e-16,
)

idx = np.arange(1, len(boundary_pts) + 1)

# ----------------------------------------------------------
# Plot
# ----------------------------------------------------------

plt.figure(figsize=(7, 4))

plt.semilogy(
    idx,
    err_stage1,
    "b-",
    linewidth=1.5,
    label="Stage 1",
)

plt.semilogy(
    idx,
    err_stage2,
    "r-",
    linewidth=1.5,
    label="Stage 2",
)

plt.xlabel("Boundary point index")
plt.ylabel("Absolute Dirichlet BC error")

plt.xlim(1, len(boundary_pts))

plt.xticks(
    np.arange(0, len(boundary_pts) + 1, 100),
    fontsize=11,
)

plt.tick_params(labelsize=11)

plt.grid(
    True,
    which="both",
    linestyle="--",
    alpha=0.3,
)

plt.legend(
    fontsize=11,
    frameon=False,
    loc="best",
)

plt.tight_layout()

plt.savefig(
    "fig3_bnd_error2.pdf",
    dpi=600,
    bbox_inches="tight",
)

plt.show()


# ==========================================================
# Figure 3 : Stage-2 Absolute Error
# ==========================================================

# ----------------------------------------------------------
# Load boundary
# ----------------------------------------------------------

boundary_pts = np.load("data/sweden_boundary_points.npy")

xi_b = boundary_pts[:, 0]
yi_b = boundary_pts[:, 1]

# ----------------------------------------------------------
# Load Stage-2 solution
# ----------------------------------------------------------

stage2 = np.loadtxt("data/stage2_solution.dat")

x_all = stage2[:, 0]
y_all = stage2[:, 1]
u_pred = stage2[:, 2]

# ----------------------------------------------------------
# Exact solution
# ----------------------------------------------------------

u_exact = (
    np.sin(np.pi * x_all)
    * np.sin(np.pi * y_all)
)

# ----------------------------------------------------------
# Absolute error
# ----------------------------------------------------------

err_after = np.abs(u_pred - u_exact)

rms_error = np.sqrt(np.mean(err_after**2))

print("")
print("Stage-2 RMS Error")
print("-----------------")
print(f"RMS Error : {rms_error:.2e}")
print("")

vmin_after = err_after.min()
vmax_after = err_after.max()

# ----------------------------------------------------------
# Plot
# ----------------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 6))

ax.plot(
    xi_b,
    yi_b,
    color="black",
    linewidth=0.6,
    alpha=0.7,
    zorder=3,
)

sc = ax.scatter(
    x_all,
    y_all,
    c=err_after,
    s=0.5,
    vmin=vmin_after,
    vmax=vmax_after,
    cmap="viridis",
)

ax.set_aspect(
    "equal",
    adjustable="box",
)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")

ax.set_xlim(
    x_all.min(),
    x_all.max(),
)

ax.set_ylim(
    y_all.min(),
    y_all.max(),
)

# ----------------------------------------------------------
# Colorbar
# ----------------------------------------------------------

cbar = fig.colorbar(
    sc,
    ax=ax,
    shrink=1,
    pad=0.02,
)

cbar.set_label("Absolute error")

formatter = ScalarFormatter(useMathText=False)
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0))

cbar.formatter = formatter
cbar.update_ticks()

plt.tight_layout()

plt.savefig(
    "fig3d_3.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.15,
)

plt.show()


# In[ ]:




