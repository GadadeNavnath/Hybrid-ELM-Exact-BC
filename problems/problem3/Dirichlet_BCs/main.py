#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import time

from numpy.polynomial.legendre import (
    legval,
    legder
)

from scipy.linalg import (
    lu_factor,
    lu_solve
)

from scipy.interpolate import griddata

from shapely.geometry import (
    Point,
    Polygon
)


print("")
print("------STAGE 1------")


# ==========================================================
# 1. Load Sweden Interior and Boundary Points
# ==========================================================
interior_pts = np.load(
    "data/sweden_interior_points.npy"
)

boundary_pts = np.load(
    "data/sweden_boundary_points.npy"
)

print("Interior points :", interior_pts.shape[0])

print("Boundary points :", boundary_pts.shape[0])


xi_int, yi_int = interior_pts[:, 0], interior_pts[:, 1]

xi_b, yi_b = boundary_pts[:, 0], boundary_pts[:, 1]


# ==========================================================
# 2. Shifted Legendre Scaling
# ==========================================================
x_all_tmp = np.concatenate([
    xi_int,
    xi_b
])

y_all_tmp = np.concatenate([
    yi_int,
    yi_b
])

xmin, xmax = x_all_tmp.min(), x_all_tmp.max()

ymin, ymax = y_all_tmp.min(), y_all_tmp.max()

sx = 2.0 / (xmax - xmin)

sy = 2.0 / (ymax - ymin)


def scale_x(x):

    return sx * (x - xmin) - 1.0


def scale_y(y):

    return sy * (y - ymin) - 1.0


# ==========================================================
# 3. Legendre Tensor-Product Basis
# ==========================================================
Nx = 15
Ny = 25

n_unknowns = Nx * Ny

print("Total unknowns =", n_unknowns)


# ==========================================================
# 2D Tensor Product Basis Matrix
# ==========================================================
def legendre_matrix_2d(x, y, Nx, Ny):

    x = scale_x(x)

    y = scale_y(y)

    cols = []

    for i in range(Nx):

        Px = legval(
            x,
            [0] * i + [1]
        )

        for j in range(Ny):

            Py = legval(
                y,
                [0] * j + [1]
            )

            cols.append(Px * Py)

    return np.vstack(cols).T


# ==========================================================
# Second Derivative Matrices
# ==========================================================
def legendre_derivative_matrices_2d(
    x,
    y,
    Nx,
    Ny
):

    x = scale_x(x)

    y = scale_y(y)

    dxx_cols = []

    dyy_cols = []

    for i in range(Nx):

        ci = [0] * i + [1]

        d2ci = legder(
            legder(ci)
        )

        Px = legval(x, ci)

        d2Px = legval(x, d2ci)

        for j in range(Ny):

            cj = [0] * j + [1]

            d2cj = legder(
                legder(cj)
            )

            Py = legval(y, cj)

            d2Py = legval(y, d2cj)

            dxx_cols.append(
                (sx**2) * d2Px * Py
            )

            dyy_cols.append(
                (sy**2) * Px * d2Py
            )

    return (
        np.vstack(dxx_cols).T,
        np.vstack(dyy_cols).T
    )


# ==========================================================
# Assemble Basis Matrices
# ==========================================================
start1 = time.time()

P_int = legendre_matrix_2d(
    xi_int,
    yi_int,
    Nx,
    Ny
)

P_b = legendre_matrix_2d(
    xi_b,
    yi_b,
    Nx,
    Ny
)

dxx_int, dyy_int = legendre_derivative_matrices_2d(
    xi_int,
    yi_int,
    Nx,
    Ny
)


# ==========================================================
# 4. Exact Solution and RHS
# ==========================================================
def u_exact(x, y):

    return (
        np.sin(np.pi * x)
        * np.sin(np.pi * y)
    )


def rhs_pde(x, y):

    return (
        2 * np.pi**2
        * np.sin(np.pi * x)
        * np.sin(np.pi * y)
        + (
            np.sin(np.pi * x)
            * np.sin(np.pi * y)
        )**3
    )


RHS_int = rhs_pde(
    xi_int,
    yi_int
)

U_b_exact = u_exact(
    xi_b,
    yi_b
)


# ==========================================================
# 5. Initial Linear Solve
# ==========================================================
A_int = -(
    dxx_int + dyy_int
)

A_lin = np.vstack([
    A_int,
    P_b
])

R_lin = np.concatenate([
    RHS_int,
    U_b_exact
])

beta = np.linalg.lstsq(
    A_lin,
    R_lin,
    rcond=None
)[0]


res_hist_stage1 = []

res_hist_stage2 = []


# ==========================================================
# 6. Gauss-Newton Iteration
# ==========================================================
print("\nGauss-Newton Iteration\n")

tol = 1e-12

maxit = 4

for it in range(maxit):

    u_int = P_int @ beta

    uxx = dxx_int @ beta

    uyy = dyy_int @ beta

    R_int = (
        -(uxx + uyy)
        + u_int**3
        - RHS_int
    )

    R_b = (
        P_b @ beta
        - U_b_exact
    )

    R = np.concatenate([
        R_int,
        R_b
    ])

    res = np.linalg.norm(R)

    res_hist_stage1.append(res)

    print(
        f"Iter {it}: ||R|| = {res:.2e}"
    )

    if res < tol:
        break

    J_int = (
        -(dxx_int + dyy_int)
        + (3 * u_int**2)[:, None] * P_int
    )

    J = np.vstack([
        J_int,
        P_b
    ])

    delta = np.linalg.lstsq(
        J,
        -R,
        rcond=None
    )[0]

    beta += delta


# ==========================================================
# 7. All-Point Matrices
# ==========================================================
x_all = np.concatenate([
    xi_int,
    xi_b
])

y_all = np.concatenate([
    yi_int,
    yi_b
])

P_all = legendre_matrix_2d(
    x_all,
    y_all,
    Nx,
    Ny
)


# ==========================================================
# 8. Global Error Before Exact BC Enforcement
# ==========================================================
u_before = P_all @ beta

u_exact_all = u_exact(
    x_all,
    y_all
)

end1 = time.time()

print("")
print(
    "Stage 1 time:",
    np.round((end1 - start1), 2),
    "s"
)

print("")

abs_error_before = np.abs(
    u_before - u_exact_all
)

print(
    "Global Errors before exact BC enforcement"
)

print(
    f"RMS error  = "
    f"{np.sqrt(np.mean(abs_error_before**2)):.2e}"
)

print(
    f"Linf error = "
    f"{np.max(abs_error_before):.2e}"
)


# ==========================================================
# STAGE 2
# ==========================================================
print("")
print("------STAGE 2------")


# ==========================================================
# 9. Gaussian RBF
# ==========================================================
start2 = time.time()


def rbf_matrix(
    x,
    y,
    cx,
    cy,
    lam
):

    dx = x[:, None] - cx[None, :]

    dy = y[:, None] - cy[None, :]

    return np.exp(
        -lam * (dx**2 + dy**2)
    )


lam = 1000.0

cx, cy = (
    xi_b.copy(),
    yi_b.copy()
)

A_bc = rbf_matrix(
    xi_b,
    yi_b,
    cx,
    cy,
    lam
)

LU, piv = lu_factor(A_bc)

AinvPb = lu_solve(
    (LU, piv),
    P_b
)


dx = xi_int[:, None] - cx[None, :]

dy = yi_int[:, None] - cy[None, :]

r2 = dx**2 + dy**2

Phi = np.exp(-lam * r2)

Phi_int = Phi

Phi_xx = (
    4 * lam**2 * dx**2
    - 2 * lam
) * Phi

Phi_yy = (
    4 * lam**2 * dy**2
    - 2 * lam
) * Phi


correction = Phi_int @ AinvPb

effective_P = P_int - correction

L_reduced = -(
    dxx_int
    + dyy_int
    + Phi_xx @ (-AinvPb)
    + Phi_yy @ (-AinvPb)
)


# ==========================================================
# 10. Reduced Newton Iteration
# ==========================================================
def residual(beta):

    u_b_pred = P_b @ beta

    q = lu_solve(
        (LU, piv),
        U_b_exact - u_b_pred
    )

    u = (
        P_int @ beta
        + Phi_int @ q
    )

    uxx = (
        dxx_int @ beta
        + Phi_xx @ q
    )

    uyy = (
        dyy_int @ beta
        + Phi_yy @ q
    )

    R = (
        -(uxx + uyy)
        + u**3
        - RHS_int
    )

    return R, q, u


print("\nReduced Newton\n")

for it in range(2):

    R, q, u = residual(beta)

    res = np.linalg.norm(R)

    res_hist_stage2.append(res)

    print(
        f"Reduced Iter {it}: "
        f"||R|| = {res:.2e}"
    )

    if res < tol:
        break

    J = (
        L_reduced
        + (3 * u**2)[:, None]
        * effective_P
    )

    delta = np.linalg.lstsq(
        J,
        -R,
        rcond=None
    )[0]

    beta += delta


# ==========================================================
# Final q
# ==========================================================
u_b_pred = P_b @ beta

q = lu_solve(
    (LU, piv),
    U_b_exact - u_b_pred
)


# ==========================================================
# 11. Final Error
# ==========================================================
Phi_all = rbf_matrix(
    x_all,
    y_all,
    cx,
    cy,
    lam
)

u_all = (
    P_all @ beta
    + Phi_all @ q
)

end2 = time.time()

print("")
print(
    "Stage 2 time:",
    np.round((end2 - start2), 2),
    "s"
)

print("")

abs_error_after = np.abs(
    u_all - u_exact_all
)

print("")

print(
    f"RMS error  = "
    f"{np.sqrt(np.mean(abs_error_after**2)):.2e}"
)

print(
    f"Linf error = "
    f"{np.max(abs_error_after):.2e}"
)


# ==========================================================
# Boundary Condition Verification
# ==========================================================
psi_boundary = u_b_pred + A_bc @ q

dirichlet_error = np.abs(
    psi_boundary - U_b_exact
)

print("")

print(
    f"Max BC error  = "
    f"{np.max(dirichlet_error):.2e}"
)

print(
    f"Mean BC error = "
    f"{np.mean(dirichlet_error):.2e}"
)


# ==========================================================
# Absolute Error Plots
# ==========================================================
err_before = abs_error_before

err_after = abs_error_after


# ==========================================================
# Color Limits
# ==========================================================
vmin_before = 0.0

vmax_before = err_before.max()

vmin_after = 0.0

vmax_after = err_after.max()


# ==========================================================
# Figure 1: Error BEFORE BC Enforcement
# ==========================================================
fig1, ax1 = plt.subplots(figsize=(6, 6))

ax1.plot(
    xi_b,
    yi_b,
    color="black",
    linewidth=0.6,
    alpha=0.7,
    zorder=3
)

sc1 = ax1.scatter(
    x_all,
    y_all,
    c=err_before,
    s=0.5,
    vmin=vmin_before,
    vmax=vmax_before,
    cmap="viridis"
)

ax1.set_aspect(
    "equal",
    adjustable="box"
)

ax1.set_xlabel(r"$x$")

ax1.set_ylabel(r"$y$")

ax1.set_xlim(
    x_all.min(),
    x_all.max()
)

ax1.set_ylim(
    y_all.min(),
    y_all.max()
)


# ==========================================================
# Colorbar
# ==========================================================
cbar1 = fig1.colorbar(
    sc1,
    ax=ax1,
    shrink=1,
    pad=0.02
)

cbar1.set_label(
    "Absolute error"
)


# ==========================================================
# Save Figure
# ==========================================================
plt.tight_layout()

plt.savefig(
    "fig3d_1.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.15
)

plt.show()

plt.close(fig1)


# ==========================================================
# Figure 2: Error AFTER BC Enforcement
# ==========================================================
fig2, ax2 = plt.subplots(figsize=(6, 6))

ax2.plot(
    xi_b,
    yi_b,
    color="black",
    linewidth=0.6,
    alpha=0.7,
    zorder=3
)

sc2 = ax2.scatter(
    x_all,
    y_all,
    c=err_after,
    s=0.5,
    vmin=vmin_after,
    vmax=vmax_after,
    cmap="viridis"
)

ax2.set_aspect(
    "equal",
    adjustable="box"
)

ax2.set_xlabel(r"$x$")

ax2.set_ylabel(r"$y$")

ax2.set_xlim(
    x_all.min(),
    x_all.max()
)

ax2.set_ylim(
    y_all.min(),
    y_all.max()
)


# ==========================================================
# Colorbar
# ==========================================================
cbar2 = fig2.colorbar(
    sc2,
    ax=ax2,
    shrink=1,
    pad=0.02
)

cbar2.set_label(
    "Absolute error"
)


# ==========================================================
# Save Figure
# ==========================================================
plt.tight_layout()

plt.savefig(
    "fig3d_2.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.15
)

plt.show()

plt.close(fig2)

# ==========================================================
# Convergence Plot
# ==========================================================
iters1 = np.arange(len(res_hist_stage1))

iters2 = np.arange(len(res_hist_stage2))

max_iter = max(
    len(res_hist_stage1),
    len(res_hist_stage2)
)

plt.figure(figsize=(5, 4))


# ==========================================================
# Stage 1 Residual History
# ==========================================================
plt.semilogy(
    iters1,
    res_hist_stage1,
    marker='o',
    markersize=4,
    markerfacecolor='white',
    markeredgewidth=1,
    linewidth=1,
    color='black',
    label='Stage 1'
)


# ==========================================================
# Stage 2 Residual History
# ==========================================================
plt.semilogy(
    iters2,
    res_hist_stage2,
    marker='s',
    markersize=4,
    markerfacecolor='white',
    markeredgewidth=1,
    linewidth=1,
    color='red',
    label='Stage 2'
)


# ==========================================================
# Axis Labels
# ==========================================================
plt.xlabel("Gauss–Newton Iteration")

plt.ylabel(r"$\|R\|$")


# ==========================================================
# Integer Iteration Ticks
# ==========================================================
plt.xticks(np.arange(max_iter))


# ==========================================================
# Grid and Legend
# ==========================================================
plt.grid(
    True,
    linestyle='-',
    linewidth=0.5,
    alpha=0.5
)

plt.legend(frameon=False)


# ==========================================================
# Save Figure
# ==========================================================
plt.tight_layout()

plt.savefig(
    "fig3d_res.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.15
)

plt.show()


# ==========================================================
# Exact Solution Plot
# ==========================================================
HEIGHT = 1.7
# ==========================================================
# Construct Sweden Polygon
# ==========================================================
domain_polygon = Polygon(
    np.column_stack([xi_b, yi_b])
)


# ==========================================================
# Exact Solution
# ==========================================================
U_exact_all = u_exact(
    x_all,
    y_all
)


# ==========================================================
# Create Regular Grid
# ==========================================================
Nxg = 700

Nyg = 700

xg = np.linspace(
    x_all.min(),
    x_all.max(),
    Nxg
)

yg = np.linspace(
    y_all.min(),
    y_all.max(),
    Nyg
)

Xg, Yg = np.meshgrid(
    xg,
    yg
)


# ==========================================================
# Interpolate Exact Solution
# ==========================================================
Ug = griddata(
    (x_all, y_all),
    U_exact_all,
    (Xg, Yg),
    method="cubic"
)


# ==========================================================
# Mask Points Outside Sweden
# ==========================================================
mask = np.zeros(
    Xg.shape,
    dtype=bool
)

for i in range(Xg.shape[0]):

    for j in range(Xg.shape[1]):

        if not domain_polygon.contains(
            Point(Xg[i, j], Yg[i, j])
        ):

            mask[i, j] = True


Ug = np.ma.array(
    Ug,
    mask=mask
)


# ==========================================================
# Create Figure
# ==========================================================
fig, ax = plt.subplots(
    figsize=(6, 6)
)


# ==========================================================
# Filled Contour Plot
# ==========================================================
cf = ax.contourf(
    Xg,
    Yg,
    Ug,
    levels=100,
    cmap="viridis"
)


# ==========================================================
# Boundary Curve
# ==========================================================
ax.plot(
    xi_b,
    yi_b,
    color="black",
    linewidth=0.6,
    alpha=0.7,
    zorder=3
)


# ==========================================================
# Axis Settings
# ==========================================================
ax.set_aspect(
    "equal",
    adjustable="box"
)

ax.set_xlabel(r"$x$")

ax.set_ylabel(r"$y$")

ax.set_xlim(-1, 1)

ax.set_ylim(-HEIGHT, HEIGHT)


# ==========================================================
# Colorbar
# ==========================================================
cbar = fig.colorbar(
    cf,
    ax=ax,
    shrink=1,
    pad=0.02
)

cbar.set_label(
    r"$u(x,y)$"
)


# ==========================================================
# Save Figure
# ==========================================================
plt.tight_layout()

plt.savefig(
    "fig3_exact.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.15
)

plt.show()

plt.close(fig)


# In[ ]:




