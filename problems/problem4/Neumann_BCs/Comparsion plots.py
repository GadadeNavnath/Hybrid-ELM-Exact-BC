#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ==========================================================
# Exact solution
# ==========================================================

def u_exact(x, y, z):
    return np.exp(x) * y**2 + (z**2 + 2.0) * np.sin(y)


# ==========================================================
# Figure 1 : Neumann Boundary Error Comparison
# ==========================================================

# ----------------------------------------------------------
# Load Stage-1 and Stage-2 boundary data
# ----------------------------------------------------------

stage1 = np.loadtxt("data/neumann_boundary_error_stage1.dat")
stage2 = np.loadtxt("data/neumann_boundary_error_stage2.dat")

# ----------------------------------------------------------
# Absolute boundary errors
# ----------------------------------------------------------

err_stage1 = np.maximum(stage1[:, 5], 1e-16)
err_stage2 = np.maximum(stage2[:, 5], 1e-16)

N_boundary = len(err_stage1)
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
plt.ylabel("Absolute Neumann BC error")

plt.xlim(1, N_boundary)

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
    "fig4_bnd_error1.pdf",
    dpi=600,
    bbox_inches="tight",
)

plt.show()


# ==========================================================
# Figure 2 : Neumann Error Comparison
# ==========================================================

# ----------------------------------------------------------
# Load Stage-1 Neumann error
# ----------------------------------------------------------

stage1 = np.loadtxt("data/stage1_neumann_error.dat", skiprows=1)

err_stage1 = np.maximum(stage1[:, -1], 1e-16)

# ----------------------------------------------------------
# Load Stage-2 Neumann error
# ----------------------------------------------------------

stage2 = np.loadtxt("data/stage2_neumann_error.dat", skiprows=1)

err_stage2 = np.maximum(stage2[:, -1], 1e-16)

idx = np.arange(1, len(err_stage1) + 1)

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
plt.ylabel("Absolute Neumann BC error")

plt.xlim(1, len(idx))

plt.xticks(
    np.arange(0, len(idx) + 1, 100),
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
    "fig4_bnd_error2.pdf",
    dpi=600,
    bbox_inches="tight",
)

plt.show()


# ==========================================================
# Figure 3 : Stage-2 Absolute Error
# ==========================================================

# ----------------------------------------------------------
# Load Stage-2 solution
# ----------------------------------------------------------

stage2 = np.loadtxt(
    "data/stage2_solution_all.dat",
    skiprows=1,
)

x_all = stage2[:, 0]
y_all = stage2[:, 1]
z_all = stage2[:, 2]
u_stage2 = stage2[:, 3]

# ----------------------------------------------------------
# Absolute error
# ----------------------------------------------------------

u_true = u_exact(x_all, y_all, z_all)

err_stage2 = np.abs(u_stage2 - u_true)

rms_error = np.sqrt(np.mean(err_stage2**2))

print("")
print("Stage-2 RMS Error")
print("-----------------")
print(f"RMS Error : {rms_error:.2e}")
print("")

# ----------------------------------------------------------
# Plot
# ----------------------------------------------------------

fig = plt.figure(figsize=(6, 6))

ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    x_all,
    y_all,
    z_all,
    c=err_stage2,
    cmap="viridis",
    s=0.5,
    vmin=0.0,
    vmax=np.max(err_stage2),
)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$z$")

ax.set_box_aspect((1, 1, 1))

# ----------------------------------------------------------
# Colorbar
# ----------------------------------------------------------

cbar = fig.colorbar(
    scatter,
    ax=ax,
    shrink=0.6,
    pad=0.03,
)

cbar.set_label("Absolute error")

formatter = ScalarFormatter(useMathText=False)
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0))

cbar.formatter = formatter
cbar.update_ticks()

plt.tight_layout()

plt.savefig(
    "fig4n_3.pdf",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

plt.close()


# In[ ]:




