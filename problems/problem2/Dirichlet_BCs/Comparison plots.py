#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D
from numpy.polynomial.legendre import legval


# ==========================================================
# Figure 1 : Boundary Correction Comparison
# ==========================================================

# ----------------------------------------------------------
# Load IELM Stage-2 Component
# ----------------------------------------------------------

interior_pts = np.load("data/cardioid_interior_points.npy")
boundary_pts = np.load("data/cardioid_boundary_points.npy")

data = np.load("data/solution_coefficients.npz")

stage2_component = np.abs(data["stage2_component"])

xi_int = interior_pts[:, 0]
yi_int = interior_pts[:, 1]

xi_b = boundary_pts[:, 0]
yi_b = boundary_pts[:, 1]

x_ielm = np.concatenate([xi_int, xi_b])
y_ielm = np.concatenate([yi_int, yi_b])

# ----------------------------------------------------------
# Load Lagaris Boundary Correction
# ----------------------------------------------------------

baseline = np.loadtxt("data/boundary_correction.dat")

x_lag = baseline[:, 0]
y_lag = baseline[:, 1]
bc_lag = np.abs(baseline[:, 2])

# ----------------------------------------------------------
# Figure
# ----------------------------------------------------------

fig = plt.figure(figsize=(14, 6))

# ==========================================================
# Baseline
# ==========================================================

ax1 = fig.add_subplot(
    121,
    projection="3d",
)

surf1 = ax1.plot_trisurf(
    x_lag,
    y_lag,
    bc_lag,
    cmap="YlGnBu",
    linewidth=0.05,
    edgecolor="k",
    antialiased=True,
)

ax1.set_xlabel(r"$x$", fontsize=13)
ax1.set_ylabel(r"$y$", fontsize=13)

# ----------------------------------------------------------
# Colorbar
# ----------------------------------------------------------

cbar1 = fig.colorbar(
    surf1,
    ax=ax1,
    shrink=0.7,
    pad=0.07,
)

fmt1 = ScalarFormatter(useMathText=True)
fmt1.set_powerlimits((0, 0))

cbar1.formatter = fmt1
cbar1.update_ticks()

cbar1.set_label(
    r"$ RBFNN\ correction$",
    fontsize=12,
)

fmt = ScalarFormatter(useMathText=False)
fmt.set_scientific(True)
fmt.set_powerlimits((0, 0))

ax1.zaxis.set_major_formatter(fmt)
ax1.ticklabel_format(
    axis="z",
    style="sci",
    scilimits=(0, 0),
)

cbar1.formatter = fmt
cbar1.update_ticks()

# ==========================================================
# IELM
# ==========================================================

ax2 = fig.add_subplot(
    122,
    projection="3d",
)

surf2 = ax2.plot_trisurf(
    x_ielm,
    y_ielm,
    stage2_component,
    cmap="YlGnBu",
    linewidth=0.05,
    edgecolor="k",
    antialiased=True,
)

ax2.set_xlabel(r"$x$", fontsize=13)
ax2.set_ylabel(r"$y$", fontsize=13)

# ----------------------------------------------------------
# Colorbar
# ----------------------------------------------------------

cbar2 = fig.colorbar(
    surf2,
    ax=ax2,
    shrink=0.7,
    pad=0.07,
)

fmt2 = ScalarFormatter(useMathText=True)
fmt2.set_powerlimits((0, 0))

cbar2.formatter = fmt2
cbar2.update_ticks()

cbar2.set_label(
    r"$ RBFNN\ correction$",
    fontsize=12,
)

fmt2 = ScalarFormatter(useMathText=False)
fmt2.set_scientific(True)
fmt2.set_powerlimits((0, 0))

ax2.zaxis.set_major_formatter(fmt2)

ax2.ticklabel_format(
    axis="z",
    style="sci",
    scilimits=(0, 0),
)

cbar2.formatter = fmt2
cbar2.update_ticks()

# ----------------------------------------------------------
# Uniform appearance
# ----------------------------------------------------------

for ax in [ax1, ax2]:

    ax.tick_params(labelsize=11)

    ax.grid(True)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")

plt.subplots_adjust(
    left=0.03,
    right=0.98,
    bottom=0.03,
    top=0.90,
    wspace=0.08,
)

plt.savefig(
    "fig2_bnd_cor.pdf",
    dpi=600,
    bbox_inches="tight",
)

plt.show()

# ==========================================================
# Figure 2 : Boundary Error Comparison
# ==========================================================

# ----------------------------------------------------------
# Load Stage-1 and Stage-2 solutions
# ----------------------------------------------------------

stage1 = np.loadtxt("data/stage1_solution.dat")
stage2 = np.loadtxt("data/stage2_solution.dat")

# ----------------------------------------------------------
# Boundary solutions (last 600 points)
# ----------------------------------------------------------

u_stage1 = stage1[-600:, 2]
u_stage2 = stage2[-600:, 2]

# ----------------------------------------------------------
# Exact solution
# ----------------------------------------------------------

x = boundary_pts[:, 0]
y = boundary_pts[:, 1]

u_exact = np.log(1.0 + x**2 + y**2)

# ----------------------------------------------------------
# Boundary errors
# ----------------------------------------------------------

err_stage1 = np.abs(u_stage1 - u_exact)
err_stage2 = np.abs(u_stage2 - u_exact)

# Avoid log(0)

eps = 1e-16

err_stage1 = np.maximum(err_stage1, eps)
err_stage2 = np.maximum(err_stage2, eps)

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

plt.xlim(1, 600)

plt.xticks(
    np.arange(0, 601, 100),
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
    "fig2_bnd_error1.pdf",
    dpi=600,
    bbox_inches="tight",
)

plt.show()

# ==========================================================
# Figure 3 : Boundary Error Comparison (Legendre Expansion)
# ==========================================================

# ----------------------------------------------------------
# Interior and boundary coordinates
# ----------------------------------------------------------

xi_int = interior_pts[:, 0]
yi_int = interior_pts[:, 1]

xi_b = boundary_pts[:, 0]
yi_b = boundary_pts[:, 1]

# ----------------------------------------------------------
# Compute scaling (same as solver)
# ----------------------------------------------------------

x_all = np.concatenate([xi_int, xi_b])
y_all = np.concatenate([yi_int, yi_b])

xmin = x_all.min()
xmax = x_all.max()

ymin = y_all.min()
ymax = y_all.max()

sx = 2.0 / (xmax - xmin)
sy = 2.0 / (ymax - ymin)


def scale_x(x):
    return sx * (x - xmin) - 1.0


def scale_y(y):
    return sy * (y - ymin) - 1.0


# ----------------------------------------------------------
# Legendre basis
# ----------------------------------------------------------

Nx = 22
Ny = 26


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

beta_stage1 = data["beta_stage1"]
beta_stage2 = data["beta_stage2"]
stage2_component = data["stage2_component"]

# ----------------------------------------------------------
# Stage-1 boundary solution
# ----------------------------------------------------------

u_stage1 = P_b @ beta_stage1

# ----------------------------------------------------------
# Stage-2 boundary solution
# ----------------------------------------------------------

boundary_correction = stage2_component[-len(boundary_pts):]

u_stage2 = (
    P_b @ beta_stage2
    + boundary_correction
)

# ----------------------------------------------------------
# Exact boundary solution
# ----------------------------------------------------------

u_exact = np.log(
    1.0
    + xi_b**2
    + yi_b**2
)

# ----------------------------------------------------------
# Boundary errors
# ----------------------------------------------------------

err_stage1 = np.abs(u_stage1 - u_exact)
err_stage2 = np.abs(u_stage2 - u_exact)

eps = 1e-16

err_stage1 = np.maximum(
    err_stage1,
    eps,
)

err_stage2 = np.maximum(
    err_stage2,
    eps,
)

idx = np.arange(
    1,
    len(boundary_pts) + 1,
)

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

plt.xlim(
    1,
    len(boundary_pts),
)

plt.xticks(
    np.arange(
        0,
        len(boundary_pts) + 1,
        100,
    ),
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
    "fig2_bnd_error2.pdf",
    dpi=600,
    bbox_inches="tight",
)

plt.show()

# ==========================================================
# Figure 4 : Stage-2 Absolute Error
# ==========================================================

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

u_exact = np.log(
    1.0
    + x_all**2
    + y_all**2
)

# ----------------------------------------------------------
# Absolute error
# ----------------------------------------------------------

err_after = np.abs(
    u_pred - u_exact
)

# ----------------------------------------------------------
# RMS Error
# ----------------------------------------------------------

rms_error = np.sqrt(
    np.mean(err_after**2)
)

print("")
print("Stage-2 RMS Error")
print("-----------------")
print(f"RMS Error : {rms_error:.2e}")
print("")

vmin_after = err_after.min()
vmax_after = err_after.max()

# ----------------------------------------------------------
# Figure
# ----------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(6, 5)
)

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

formatter = ScalarFormatter(
    useMathText=False
)

formatter.set_scientific(True)
formatter.set_powerlimits((0, 0))

cbar.formatter = formatter
cbar.update_ticks()

# ----------------------------------------------------------
# Save figure
# ----------------------------------------------------------

plt.tight_layout()

plt.savefig(
    "fig2d_3.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.15,
)

plt.show()


# In[ ]:




