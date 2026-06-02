#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import legval, legder
from scipy.linalg import lu_factor, lu_solve

# ==========================================================
# Problem Setup
# ==========================================================
print("\n------STAGE 1------")

# Number of Legendre basis functions in each direction
N_basis = 10

# Total number of unknown coefficients
N_unknowns = N_basis**3

# Gauss--Newton settings
max_iter = 4
tol = 1e-12

# ==========================================================
# Physical Domain
# ==========================================================
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0
zmin, zmax = 0.0, 1.0

# ==========================================================
# Scaling Factors
# Used for transforming derivatives from reference space
# ==========================================================
scale_d2x = (2.0 / (xmax - xmin))**2
scale_d2y = (2.0 / (ymax - ymin))**2
scale_d2z = (2.0 / (zmax - zmin))**2

# ==========================================================
# Load Interior and Boundary Points
# ==========================================================
interior_pts = np.load("data/shell_interior_points.npy")
boundary_pts = np.load("data/shell_boundary_points.npy")

# Number of points
N_int = interior_pts.shape[0]
N_bnd = boundary_pts.shape[0]

print("Interior points :", N_int)
print("Boundary points :", N_bnd)
print("Unknowns        :", N_unknowns)

# ==========================================================
# Extract Physical Coordinates
# ==========================================================
x_int = interior_pts[:, 0]
y_int = interior_pts[:, 1]
z_int = interior_pts[:, 2]

x_b = boundary_pts[:, 0]
y_b = boundary_pts[:, 1]
z_b = boundary_pts[:, 2]

# ==========================================================
# Map Physical Coordinates → Reference Coordinates
# Domain: [0,1] → [-1,1]
# ==========================================================
def map_to_reference_domain(points):

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    xi = 2.0 * (x - xmin) / (xmax - xmin) - 1.0
    yi = 2.0 * (y - ymin) / (ymax - ymin) - 1.0
    zi = 2.0 * (z - zmin) / (zmax - zmin) - 1.0

    return xi, yi, zi


# Reference coordinates
xi_int, yi_int, zi_int = map_to_reference_domain(interior_pts)
xi_b, yi_b, zi_b = map_to_reference_domain(boundary_pts)

# ==========================================================
# Build 3D Legendre Basis Matrices
# ==========================================================
def build_legendre_matrices(xi, yi, zi, N):

    # ------------------------------------------------------
    # Build 1D Legendre basis and second derivatives
    # ------------------------------------------------------
    def build_1d_legendre(pts, N, scale_d2):

        basis_list = []
        d2_list = []

        for i in range(N):

            coeff = [0]*i + [1]

            d2_coeff = legder(legder(coeff))

            basis_list.append(
                legval(pts, coeff)
            )

            d2_list.append(
                legval(pts, d2_coeff) * scale_d2
            )

        return basis_list, d2_list

    # 1D matrices
    Px, D2x = build_1d_legendre(xi, N, scale_d2x)
    Py, D2y = build_1d_legendre(yi, N, scale_d2y)
    Pz, D2z = build_1d_legendre(zi, N, scale_d2z)

    # Containers
    cols_P = []
    cols_dxx = []
    cols_dyy = []
    cols_dzz = []

    # ------------------------------------------------------
    # Tensor-product basis construction
    # ------------------------------------------------------
    for i in range(N):
        for j in range(N):
            for k in range(N):

                cols_P.append(
                    Px[i] * Py[j] * Pz[k]
                )

                cols_dxx.append(
                    D2x[i] * Py[j] * Pz[k]
                )

                cols_dyy.append(
                    Px[i] * D2y[j] * Pz[k]
                )

                cols_dzz.append(
                    Px[i] * Py[j] * D2z[k]
                )

    # Convert to matrices
    P = np.vstack(cols_P).T
    dxx = np.vstack(cols_dxx).T
    dyy = np.vstack(cols_dyy).T
    dzz = np.vstack(cols_dzz).T

    return P, dxx, dyy, dzz


# ==========================================================
# Construct Legendre Matrices
# ==========================================================
start_stage1 = time.time()

P_int, dxx_int, dyy_int, dzz_int = build_legendre_matrices(
    xi_int,
    yi_int,
    zi_int,
    N_basis
)

P_b, _, _, _ = build_legendre_matrices(
    xi_b,
    yi_b,
    zi_b,
    N_basis
)

# ==========================================================
# Exact Solution
# ==========================================================
def u_exact(x, y, z):

    return (
        np.exp(x) * (y**2)
        + (z**2 + 2.0) * np.sin(y)
    )


# ==========================================================
# PDE Right-Hand Side
# ==========================================================
def forcing_term(x, y, z):

    u_ex = u_exact(x, y, z)

    return (
        (2.0 + y**2) * np.exp(x)
        - (z**2) * np.sin(y)
        - u_ex**2
    )


# Interior forcing values
A_int = forcing_term(x_int, y_int, z_int)

# Exact boundary values
U_b_exact = u_exact(x_b, y_b, z_b)

# ==========================================================
# Initial Linear Solve
# Weak Boundary Condition Enforcement
# ==========================================================
A_linear = np.vstack([
    dxx_int + dyy_int + dzz_int,
    P_b
])

R_linear = np.concatenate([
    A_int,
    U_b_exact
])

beta = np.linalg.lstsq(
    A_linear,
    R_linear,
    rcond=None
)[0]

# Residual history
res_hist_stage1 = []
res_hist_stage2 = []

# ==========================================================
# Gauss--Newton Iteration
# ==========================================================
print("\nGauss-Newton Iteration")

for it in range(max_iter):

    # ------------------------------------------------------
    # Compute solution and derivatives
    # ------------------------------------------------------
    u_int = P_int @ beta

    uxx = dxx_int @ beta
    uyy = dyy_int @ beta
    uzz = dzz_int @ beta

    # ------------------------------------------------------
    # Residuals
    # ------------------------------------------------------
    R_int = (
        uxx + uyy + uzz
        - u_int**2
        - A_int
    )

    R_b = P_b @ beta - U_b_exact

    R = np.concatenate([R_int, R_b])

    # Residual norm
    residual_norm = np.linalg.norm(R)

    res_hist_stage1.append(residual_norm)

    print(
        f"Iter {it}: ||R|| = {residual_norm:.2e}"
    )

    # ------------------------------------------------------
    # Jacobian Matrix
    # ------------------------------------------------------
    J_int = (
        dxx_int + dyy_int + dzz_int
        - (2.0 * u_int)[:, None] * P_int
    )

    J = np.vstack([J_int, P_b])

    # ------------------------------------------------------
    # Newton Update
    # ------------------------------------------------------
    delta = np.linalg.lstsq(
        J,
        -R,
        rcond=None
    )[0]

    beta += delta

# ==========================================================
# Global Evaluation
# ==========================================================
P_all = np.vstack([P_int, P_b])

x_all = np.concatenate([x_int, x_b])
y_all = np.concatenate([y_int, y_b])
z_all = np.concatenate([z_int, z_b])

u_exact_all = u_exact(
    x_all,
    y_all,
    z_all
)

u_before = P_all @ beta

# ==========================================================
# Stage 1 Diagnostics
# ==========================================================
end_stage1 = time.time()

print("\nStage 1 Time:",
      np.round(end_stage1 - start_stage1, 2),
      "s")

abs_error_before = np.abs(
    u_before - u_exact_all
)

L2_before = np.sqrt(
    np.mean(abs_error_before**2)
)

Linf_before = np.max(abs_error_before)

print("\nErrors BEFORE Exact BC Enforcement")
print(f"RMS error   = {L2_before:.2e}")
print(f"Linf error = {Linf_before:.2e}")

# ==========================================================
# STEP 2
# Exact Boundary Condition Enforcement Using Gaussian RBF
# ==========================================================
print("\n------STAGE 2------")

start_stage2 = time.time()

# Gaussian shape parameter
lam = 6500.0

# ==========================================================
# Gaussian RBF Matrix
# ==========================================================
def gaussian_rbf_matrix(x, y, z, cx, cy, cz):

    dx = x[:, None] - cx[None, :]
    dy = y[:, None] - cy[None, :]
    dz = z[:, None] - cz[None, :]

    return np.exp(
        -lam * (dx**2 + dy**2 + dz**2)
    )


# ==========================================================
# RBF Centers = Boundary Points
# ==========================================================
cx = x_b.copy()
cy = y_b.copy()
cz = z_b.copy()

# Boundary interpolation matrix
A_bc = gaussian_rbf_matrix(
    x_b,
    y_b,
    z_b,
    cx,
    cy,
    cz
)

# LU factorization
LU, piv = lu_factor(A_bc)

# Useful precomputation
AinvPb = lu_solve((LU, piv), P_b)

# ==========================================================
# Interior RBF Derivatives
# ==========================================================
dx = x_int[:, None] - cx[None, :]
dy = y_int[:, None] - cy[None, :]
dz = z_int[:, None] - cz[None, :]

r2 = dx**2 + dy**2 + dz**2

Phi_int = np.exp(-lam * r2)

Phi_xx = (
    (4.0 * lam**2 * dx**2 - 2.0 * lam)
    * Phi_int
)

Phi_yy = (
    (4.0 * lam**2 * dy**2 - 2.0 * lam)
    * Phi_int
)

Phi_zz = (
    (4.0 * lam**2 * dz**2 - 2.0 * lam)
    * Phi_int
)

# ==========================================================
# Reduced Residual Function
# ==========================================================
def reduced_residual(beta):

    # Boundary prediction
    u_b_pred = P_b @ beta

    # RBF correction coefficients
    q = lu_solve(
        (LU, piv),
        U_b_exact - u_b_pred
    )

    # Corrected solution
    u = P_int @ beta + Phi_int @ q

    # Second derivatives
    uxx = dxx_int @ beta + Phi_xx @ q
    uyy = dyy_int @ beta + Phi_yy @ q
    uzz = dzz_int @ beta + Phi_zz @ q

    # PDE residual
    R = (
        uxx + uyy + uzz
        - u**2
        - A_int
    )

    return R, q, u


print("\nReduced Newton Iteration")

# ==========================================================
# Precompute Reduced Operators
# ==========================================================
effective_P = P_int - Phi_int @ AinvPb

L_reduced = (
    dxx_int + dyy_int + dzz_int
    - Phi_xx @ AinvPb
    - Phi_yy @ AinvPb
    - Phi_zz @ AinvPb
)

# ==========================================================
# Reduced Newton Iteration
# ==========================================================
for it in range(2):

    R, q, u = reduced_residual(beta)

    residual_norm = np.linalg.norm(R)

    res_hist_stage2.append(residual_norm)

    print(
        f"Reduced Iter {it}: ||R|| = {residual_norm:.2e}"
    )

    if residual_norm < tol:
        break

    # Jacobian
    J = (
        L_reduced
        - (2.0 * u)[:, None] * effective_P
    )

    # Newton update
    delta = np.linalg.lstsq(
        J,
        -R,
        rcond=None
    )[0]

    beta += delta

# ==========================================================
# Final Solution
# ==========================================================
u_b_pred = P_b @ beta

q = lu_solve(
    (LU, piv),
    U_b_exact - u_b_pred
)

Phi_all = gaussian_rbf_matrix(
    x_all,
    y_all,
    z_all,
    cx,
    cy,
    cz
)

u_all = P_all @ beta + Phi_all @ q

# ==========================================================
# Final Diagnostics
# ==========================================================
end_stage2 = time.time()

print("\nStage 2 Time:",
      np.round(end_stage2 - start_stage2, 2),
      "s")

# Boundary condition error
psi_boundary = u_b_pred + A_bc @ q

dirichlet_error = np.abs(
    psi_boundary - U_b_exact
)

print(f"\nMax BC Error  = {np.max(dirichlet_error):.2e}")
print(f"Mean BC Error = {np.mean(dirichlet_error):.2e}")

# Final global errors
abs_error_after = np.abs(
    u_all - u_exact_all
)

L2_after = np.sqrt(
    np.mean(abs_error_after**2)
)

Linf_after = np.max(abs_error_after)

print(f"\nRMS error   = {L2_after:.2e}")
print(f"Linf error = {Linf_after:.2e}")


# ==========================================================
# Absolute Error Plots (Separate Figures)
# ==========================================================

# ----------------------------------------------------------
# Error arrays
# ----------------------------------------------------------
err_before = abs_error_before
err_after  = abs_error_after


# ==========================================================
# Figure 1 : Error BEFORE Exact BC Enforcement
# ==========================================================
fig = plt.figure(figsize=(6, 6))

ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    x_all,
    y_all,
    z_all,
    c=err_before,
    cmap="viridis",
    s=0.5,
    vmin=0.0,
    vmax=np.max(err_before)
)

# ----------------------------------------------------------
# Axis labels
# ----------------------------------------------------------
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$z$")

# Equal aspect ratio
ax.set_box_aspect((1, 1, 1))

# ----------------------------------------------------------
# Colorbar
# ----------------------------------------------------------
cbar = fig.colorbar(
    scatter,
    ax=ax,
    shrink=0.6,
    pad=0.03
)

cbar.set_label("Absolute error")

plt.tight_layout()

# ----------------------------------------------------------
# Save figure
# ----------------------------------------------------------
plt.savefig(
    "fig4d_1.pdf",dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ==========================================================
# Figure 2 : Error AFTER Exact BC Enforcement
# ==========================================================
fig = plt.figure(figsize=(6, 6))

ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    x_all,
    y_all,
    z_all,
    c=err_after,
    cmap="viridis",
    s=0.5,
    vmin=0.0,
    vmax=np.max(err_after)
)

# ----------------------------------------------------------
# Axis labels
# ----------------------------------------------------------
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$z$")

# Equal aspect ratio
ax.set_box_aspect((1, 1, 1))

# ----------------------------------------------------------
# Colorbar
# ----------------------------------------------------------
cbar = fig.colorbar(
    scatter,
    ax=ax,
    shrink=0.6,
    pad=0.03
)

cbar.set_label("Absolute error")

plt.tight_layout()

# ----------------------------------------------------------
# Save figure
# ----------------------------------------------------------
plt.savefig(
    "fig4d_2.pdf",dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ==========================================================
# Convergence Plot
# ==========================================================
iters1 = np.arange(len(res_hist_stage1))
iters2 = np.arange(len(res_hist_stage2))

# Maximum iteration number
max_iter = max(
    len(res_hist_stage1),
    len(res_hist_stage2)
)

# ==========================================================
# Create Figure
# ==========================================================
plt.figure(figsize=(5, 4))

# ----------------------------------------------------------
# Stage 1 Residual History
# ----------------------------------------------------------
plt.semilogy(
    iters1,
    res_hist_stage1,
    marker="o",
    markersize=4,
    markerfacecolor="white",
    markeredgewidth=1,
    linewidth=1,
    color="black",
    label="Stage 1"
)

# ----------------------------------------------------------
# Stage 2 Residual History
# ----------------------------------------------------------
plt.semilogy(
    iters2,
    res_hist_stage2,
    marker="s",
    markersize=4,
    markerfacecolor="white",
    markeredgewidth=1,
    linewidth=1,
    color="red",
    label="Stage 2"
)

# ----------------------------------------------------------
# Labels
# ----------------------------------------------------------
plt.xlabel("Gauss-Newton Iteration")

plt.ylabel(r"$\|R\|$")

# ----------------------------------------------------------
# Integer iteration ticks
# ----------------------------------------------------------
plt.xticks(np.arange(0, max_iter))

# ----------------------------------------------------------
# Grid and Legend
# ----------------------------------------------------------
plt.grid(
    True,
    linestyle="-",
    linewidth=0.5,
    alpha=0.5
)

plt.legend(frameon=False)

plt.tight_layout()

# ----------------------------------------------------------
# Save Figures
# ----------------------------------------------------------
plt.savefig(
    "fig4d_res.pdf",dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[ ]:




