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


print("")
print("------STAGE 1------")


# ==========================================================
# 1. Load Sweden Points
# ==========================================================
interior_pts = np.load(
    "data/sweden_interior_points.npy"
)

boundary_pts = np.load(
    "data/sweden_boundary_points.npy"
)

xi_int, yi_int = (
    interior_pts[:, 0],
    interior_pts[:, 1]
)

xi_b, yi_b = (
    boundary_pts[:, 0],
    boundary_pts[:, 1]
)

print(
    "Interior points :",
    interior_pts.shape[0]
)

print(
    "Boundary points :",
    boundary_pts.shape[0]
)


# ==========================================================
# 2. Load Boundary Normals and K Matrix
# ==========================================================
boundary_normals = np.load(
    "data/sweden_normals.npy"
)

K = np.load(
    "data/K.npy"
)

LU_K, piv_K = lu_factor(K)

print(
    "Boundary normals :",
    boundary_normals.shape[0]
)

print(
    "K matrix shape :",
    K.shape
)


# ==========================================================
# 3. Bounding Box Scaling
# ==========================================================
x_all_tmp = np.concatenate([
    xi_int,
    xi_b
])

y_all_tmp = np.concatenate([
    yi_int,
    yi_b
])

xmin, xmax = (
    x_all_tmp.min(),
    x_all_tmp.max()
)

ymin, ymax = (
    y_all_tmp.min(),
    y_all_tmp.max()
)

sx = 2.0 / (xmax - xmin)

sy = 2.0 / (ymax - ymin)


def scale_x(x):

    return sx * (x - xmin) - 1.0


def scale_y(y):

    return sy * (y - ymin) - 1.0


# ==========================================================
# 4. Legendre Basis
# ==========================================================
Nx = 15

Ny = 25

n_unknowns = Nx * Ny

print(
    "Total unknowns =",
    n_unknowns
)


# ==========================================================
# 2D Tensor Product Basis Matrix
# ==========================================================
def legendre_matrix_2d(
    x,
    y,
    Nx,
    Ny
):

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
# First Derivative Matrices
# ==========================================================
def legendre_first_derivative_matrices_2d(
    x,
    y,
    Nx,
    Ny
):

    x = scale_x(x)

    y = scale_y(y)

    dx_cols = []

    dy_cols = []

    for i in range(Nx):

        ci = [0] * i + [1]

        dci = legder(ci)

        Px = legval(x, ci)

        dPx = legval(x, dci)

        for j in range(Ny):

            cj = [0] * j + [1]

            dcj = legder(cj)

            Py = legval(y, cj)

            dPy = legval(y, dcj)

            dx_cols.append(
                sx * dPx * Py
            )

            dy_cols.append(
                sy * Px * dPy
            )

    return (
        np.vstack(dx_cols).T,
        np.vstack(dy_cols).T
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

P_all = legendre_matrix_2d(
    np.concatenate([xi_int, xi_b]),
    np.concatenate([yi_int, yi_b]),
    Nx,
    Ny
)

dxx_int, dyy_int = legendre_derivative_matrices_2d(
    xi_int,
    yi_int,
    Nx,
    Ny
)

dx_b, dy_b = legendre_first_derivative_matrices_2d(
    xi_b,
    yi_b,
    Nx,
    Ny
)


# ==========================================================
# 5. Exact Solution
# ==========================================================
def u_exact(x, y):

    return (
        np.sin(np.pi * x)
        * np.sin(np.pi * y)
    )


def rhs_pde(x, y):

    u = (
        np.sin(np.pi * x)
        * np.sin(np.pi * y)
    )

    return (
        2 * np.pi**2 * u
        + u**3
    )


def grad_u_exact(x, y):

    ux = (
        np.pi
        * np.cos(np.pi * x)
        * np.sin(np.pi * y)
    )

    uy = (
        np.pi
        * np.sin(np.pi * x)
        * np.cos(np.pi * y)
    )

    return ux, uy


RHS_int = rhs_pde(
    xi_int,
    yi_int
)

ux_b, uy_b = grad_u_exact(
    xi_b,
    yi_b
)

g_neumann = (
    ux_b * boundary_normals[:, 0]
    + uy_b * boundary_normals[:, 1]
)


# ==========================================================
# 6. Build Neumann Polynomial Operator
# ==========================================================
nx = boundary_normals[:, 0][:, None]

ny = boundary_normals[:, 1][:, None]

B_neu = (
    nx * dx_b
    + ny * dy_b
)


# ==========================================================
# Neumann-Consistent Initialization
# ==========================================================
A_int = -(
    dxx_int + dyy_int
) + P_int

R_int = RHS_int

A_lin = np.vstack([
    A_int,
    B_neu
])

R_lin = np.concatenate([
    R_int,
    g_neumann
])

beta = np.linalg.lstsq(
    A_lin,
    R_lin,
    rcond=None
)[0]


res_hist_stage1 = []

res_hist_stage2 = []


# ==========================================================
# 7. Gauss-Newton Iteration
# ==========================================================
print("")

print("Gauss-Newton Iteration")

tol = 1e-10

maxit = 4

for it in range(maxit):

    u_int = P_int @ beta

    R_int = (
        -(dxx_int @ beta + dyy_int @ beta)
        + u_int**3
        - RHS_int
    )

    flux_poly = B_neu @ beta

    R_neu = flux_poly - g_neumann

    R = np.concatenate([
        R_int,
        R_neu
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
        + (3 * u_int**2)[:, None]
        * P_int
    )

    J = np.vstack([
        J_int,
        B_neu
    ])

    delta = np.linalg.lstsq(
        J,
        -R,
        rcond=None
    )[0]

    beta += delta


# ==========================================================
# Global Error Before Exact BC Enforcement
# ==========================================================
x_all = np.concatenate([
    xi_int,
    xi_b
])

y_all = np.concatenate([
    yi_int,
    yi_b
])

u_poly_all = P_all @ beta

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

abs_error_before = np.abs(
    u_poly_all - u_exact_all
)

print("")

print(
    "Global Errors before exact BC enforcement"
)

print(
    f"RMS error = "
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
# 8. RBF Neumann Enforcement
# ==========================================================
start2 = time.time()

lam = 600.0

alpha = 0.9999

hx, hy = 1e-4, 1e-4


def rbf_matrix(
    x,
    y,
    cx,
    cy,
    lam,
    alpha,
    hx,
    hy
):

    dx = (
        x[:, None]
        - alpha * cx[None, :]
        + hx
    )

    dy = (
        y[:, None]
        - alpha * cy[None, :]
        + hy
    )

    return np.exp(
        -lam * (dx**2 + dy**2)
    )


cx, cy = (
    xi_b.copy(),
    yi_b.copy()
)


# ==========================================================
# Interior RBF Pieces
# ==========================================================
dx = (
    xi_int[:, None]
    - alpha * cx[None, :]
    + hx
)

dy = (
    yi_int[:, None]
    - alpha * cy[None, :]
    + hy
)

r2 = dx**2 + dy**2

Phi = np.exp(
    -lam * r2
)

Phi_int = Phi

Phi_xx = (
    4 * lam**2 * dx**2
    - 2 * lam
) * Phi

Phi_yy = (
    4 * lam**2 * dy**2
    - 2 * lam
) * Phi


KinvB = lu_solve(
    (LU_K, piv_K),
    B_neu
)

effective_P = (
    P_int
    - Phi_int @ KinvB
)

L_reduced = (
    dxx_int
    + dyy_int
    - Phi_xx @ KinvB
    - Phi_yy @ KinvB
)


# ==========================================================
# Residual
# ==========================================================
def residual(beta):

    ux_poly = dx_b @ beta

    uy_poly = dy_b @ beta

    flux_poly = (
        ux_poly * boundary_normals[:, 0]
        + uy_poly * boundary_normals[:, 1]
    )

    rhs_neu = (
        g_neumann - flux_poly
    )

    q = lu_solve(
        (LU_K, piv_K),
        rhs_neu
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


# ==========================================================
# 9. Reduced Newton Iteration
# ==========================================================
print("")

print("Reduced Newton")

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
        -L_reduced
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
# Final q Evaluation
# ==========================================================
ux_poly = dx_b @ beta

uy_poly = dy_b @ beta

flux_poly = (
    ux_poly * boundary_normals[:, 0]
    + uy_poly * boundary_normals[:, 1]
)

rhs_neu = (
    g_neumann - flux_poly
)

q = lu_solve(
    (LU_K, piv_K),
    rhs_neu
)


# ==========================================================
# Global Error After Exact BC Enforcement
# ==========================================================
Phi_all = rbf_matrix(
    x_all,
    y_all,
    cx,
    cy,
    lam,
    alpha,
    hx,
    hy
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

abs_error_after = np.abs(
    u_all - u_exact_all
)

print("")

print(
    f"RMS error = "
    f"{np.sqrt(np.mean(abs_error_after**2)):.2e}"
)

print(
    f"Linf error = "
    f"{np.max(abs_error_after):.2e}"
)


# ==========================================================
# Neumann BC Error
# ==========================================================
flux_rbf = K @ q

flux_total = (
    flux_poly + flux_rbf
)

neumann_error = np.abs(
    flux_total - g_neumann
)

print("")

print(
    f"Max Neumann error  = "
    f"{np.max(neumann_error):.2e}"
)

print(
    f"Mean Neumann error = "
    f"{np.mean(neumann_error):.2e}"
)

print(
    f"Norm(q) = "
    f"{np.linalg.norm(q):.2e}"
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

cbar1 = fig1.colorbar(
    sc1,
    ax=ax1,
    shrink=1,
    pad=0.02
)

cbar1.set_label(
    "Absolute error"
)

plt.tight_layout()

plt.savefig(
    "fig3n_1.pdf",
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

cbar2 = fig2.colorbar(
    sc2,
    ax=ax2,
    shrink=1,
    pad=0.02
)

cbar2.set_label(
    "Absolute error"
)

plt.tight_layout()

plt.savefig(
    "fig3n_2.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.15
)

plt.show()

plt.close(fig2)

# ==========================================================
# Convergence Plot
# ==========================================================
iters1 = np.arange(
    len(res_hist_stage1)
)

iters2 = np.arange(
    len(res_hist_stage2)
)

max_iter = max(
    len(res_hist_stage1),
    len(res_hist_stage2)
)


# ==========================================================
# Create Figure
# ==========================================================
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
plt.xticks(
    np.arange(max_iter)
)


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
    "fig3n_res.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.15
)

plt.show()


# In[ ]:




