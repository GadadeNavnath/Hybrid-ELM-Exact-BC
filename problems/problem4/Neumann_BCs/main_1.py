#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from numpy.polynomial.legendre import legval, legder
import time
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt

print("")
print("------STAGE 1------")
# ==========================================================
# Basis & Newton settings
# ==========================================================
N_basis = 9

NB = N_basis ** 3

max_iter = 11

tol = 1e-12

# ==========================================================
# Domain = [0,1]^3
# ==========================================================
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0
zmin, zmax = 0.0, 1.0

scale_dx = 2 / (xmax - xmin)
scale_d2x = scale_dx ** 2
scale_dy = 2 / (ymax - ymin)
scale_d2y = scale_dy ** 2
scale_dz = 2 / (zmax - zmin)
scale_d2z = scale_dz ** 2

# ==========================================================
# Load points
# ==========================================================
interior_pts = np.load("data/shell_interior_points.npy")
boundary_pts = np.load("data/shell_boundary_points.npy")
boundary_normals = np.load("data/shell_boundary_normals.npy")
K = np.load("data/K.npy")

LU_K, piv_K = lu_factor(K)
# ==========================================================
# Load boundary normals (OUTWARD)
# ==========================================================

nx_b = boundary_normals[:,0]
ny_b = boundary_normals[:,1]
nz_b = boundary_normals[:,2]

N_int = interior_pts.shape[0]
N_bnd = boundary_pts.shape[0]
print("Interior pts:", N_int)
print("Boundary pts:", N_bnd)
print("Total unknowns:",NB)
print("")

# ==========================================================
# Mapping to Legendre coordinates
# ==========================================================
def map_to_leg_coords(X):
    x, y, z = X[:,0], X[:,1], X[:,2]
    xi = 2*(x - xmin)/(xmax - xmin) - 1
    yi = 2*(y - ymin)/(ymax - ymin) - 1
    zi = 2*(z - zmin)/(zmax - zmin) - 1
    return xi, yi, zi

xi_int, yi_int, zi_int = map_to_leg_coords(interior_pts)
xi_b, yi_b, zi_b = map_to_leg_coords(boundary_pts)

# ==========================================================
# Build 3D Legendre matrices
# ==========================================================
def leg_matrices_3d(xi, yi, zi, N):

    def leg_1d_all(pts, N, scale1, scale2):
        P_list, D_list, D2_list = [], [], []
        for i in range(N):
            ci = [0]*i + [1]
            dci = legder(ci)
            d2ci = legder(dci)

            P_list.append(legval(pts, ci))
            D_list.append(legval(pts, dci) * scale1)
            D2_list.append(legval(pts, d2ci) * scale2)

        return P_list, D_list, D2_list

    Px, Dx, D2x = leg_1d_all(xi, N, scale_dx, scale_d2x)
    Py, Dy, D2y = leg_1d_all(yi, N, scale_dy, scale_d2y)
    Pz, Dz, D2z = leg_1d_all(zi, N, scale_dz, scale_d2z)

    cols_P, cols_dx, cols_dy, cols_dz = [], [], [], []
    cols_dxx, cols_dyy, cols_dzz = [], [], []

    for i in range(N):
        for j in range(N):
            for k in range(N):
                cols_P.append(Px[i]*Py[j]*Pz[k])
                cols_dx.append(Dx[i]*Py[j]*Pz[k])
                cols_dy.append(Px[i]*Dy[j]*Pz[k])
                cols_dz.append(Px[i]*Py[j]*Dz[k])
                cols_dxx.append(D2x[i]*Py[j]*Pz[k])
                cols_dyy.append(Px[i]*D2y[j]*Pz[k])
                cols_dzz.append(Px[i]*Py[j]*D2z[k])

    return (
        np.vstack(cols_P).T,
        np.vstack(cols_dx).T,
        np.vstack(cols_dy).T,
        np.vstack(cols_dz).T,
        np.vstack(cols_dxx).T,
        np.vstack(cols_dyy).T,
        np.vstack(cols_dzz).T,
    )

start1 = time.time()

P_int, Dx_int, Dy_int, Dz_int, dxx_int, dyy_int, dzz_int = \
    leg_matrices_3d(xi_int, yi_int, zi_int, N_basis)

P_b, Dx_b, Dy_b, Dz_b, _, _, _ = \
    leg_matrices_3d(xi_b, yi_b, zi_b, N_basis)

# ==========================================================
# PDE definitions
# ==========================================================
def u_exact(x,y,z):
    return np.exp(x)*(y**2) + (z**2 + 2.0)*np.sin(y)

def A_xyz(x,y,z):
    u_ex = u_exact(x,y,z)
    return (2+y**2)*np.exp(x) - (z**2)*np.sin(y) - u_ex**2

# ==========================================================
# Data
# ==========================================================
x_int, y_int, z_int = interior_pts[:,0], interior_pts[:,1], interior_pts[:,2]
x_b, y_b, z_b = boundary_pts[:,0], boundary_pts[:,1], boundary_pts[:,2]

# ==========================================================
# Exact gradient for Neumann BC
# ==========================================================
def grad_u_exact(x,y,z):
    ux = np.exp(x)*(y**2)
    uy = 2*y*np.exp(x) + (z**2 + 2.0)*np.cos(y)
    uz = 2*z*np.sin(y)
    return ux, uy, uz

ux_b, uy_b, uz_b = grad_u_exact(x_b, y_b, z_b)
g_b = ux_b*nx_b + uy_b*ny_b + uz_b*nz_b

A_int = A_xyz(x_int, y_int, z_int)
U_b_exact = u_exact(x_b, y_b, z_b)

# ==========================================================
# Initial linear solve (PURE NEUMANN CONSISTENT)
# ==========================================================
# --- build normal derivative matrix at boundary
Pbn = nx_b[:,None]*Dx_b + ny_b[:,None]*Dy_b + nz_b[:,None]*Dz_b

# --- stacked linear system
A_lin = np.vstack([dxx_int + dyy_int + dzz_int , Pbn])
R_lin = np.concatenate([A_int, g_b])

beta = np.linalg.lstsq(A_lin, R_lin, rcond=None)[0]

res_hist_stage1 = []
res_hist_stage2 = []

# ==========================================================
# Gauss-Newton iteration
# ==========================================================
print("Gauss-Newton Iteration")

for it in range(max_iter):

    u_int = P_int @ beta
    uxx = dxx_int @ beta
    uyy = dyy_int @ beta
    uzz = dzz_int @ beta

    # interior residual
    R_int = uxx + uyy + uzz - u_int**2 - A_int

    # Neumann boundary residual
    R_b = Pbn @ beta - g_b
    R = np.concatenate([R_int, R_b])

    res=np.linalg.norm(R)
    res_hist_stage1.append(res)

    print(f"Iter {it}: ||R||={res:.2e}")

    # Jacobian
    J_int = dxx_int + dyy_int + dzz_int - (2*u_int)[:,None]*P_int
    J = np.vstack([J_int, Pbn])

    delta = np.linalg.lstsq(J, -R, rcond=None)[0]
    beta += delta

# ==========================================================
# GLOBAL STACK
# ==========================================================
P_all = np.vstack([P_int, P_b])

x_all = np.concatenate([x_int, x_b])
y_all = np.concatenate([y_int, y_b])
z_all = np.concatenate([z_int, z_b])

u_exact_all = u_exact(x_all, y_all, z_all)
u_before = P_all @ beta

end1 = time.time()
print("")
print("Stage 1 time:",np.round((end1-start1),2),"s")
print("")

# ==========================================================
# Stage 1 Neumann BC error
# ==========================================================
ux_stage1 = Dx_b @ beta
uy_stage1 = Dy_b @ beta
uz_stage1 = Dz_b @ beta

flux_stage1 = (
    ux_stage1 * nx_b +
    uy_stage1 * ny_b +
    uz_stage1 * nz_b
)

neumann_error_stage1 = np.abs(flux_stage1 - g_b)

# ==========================================================
# Save Stage 1 Neumann boundary error
# ==========================================================
stage1_neumann_data = np.column_stack((
    x_b,
    y_b,
    z_b,
    neumann_error_stage1
))

np.savetxt(
    "data/stage1_neumann_error.dat",
    stage1_neumann_data,
    fmt="%.16e",
    header="x y z neumann_error"
)

# ==========================================================
# ERROR BEFORE BC
# ==========================================================
abs_error_before = np.abs(u_before - u_exact_all)

print("\nGlobal Errors BEFORE BC enforcement")
print(f"RMS error = {np.sqrt(np.mean(abs_error_before**2)):.2e}")
print(f"Linf error = {np.max(abs_error_before):.2e}")

# ==========================================================
# STEP 2
# Exact Boundary Condition Enforcement Using Gaussian RBF
#===========================================================
print("")
print("------STAGE 2------")
# ==========================================================
# RBF MATRIX (ASYMMETRIC — MUST MATCH K) STEP 2
# ==========================================================
start2 = time.time()

lam = 6500.0
alpha = 0.9999
hx, hy, hz = 1e-5, 1e-5, 1e-5

def rbf_matrix(x, y, z, cx, cy, cz, lam, alpha, hx, hy, hz):

    dx = x[:, None] - alpha * cx[None, :] + hx
    dy = y[:, None] - alpha * cy[None, :] + hy
    dz = z[:, None] - alpha * cz[None, :] + hz

    return np.exp(-lam * (dx**2 + dy**2 + dz**2))

# ----------------------------------------------------------
# RBF centers = PHYSICAL boundary points
# ----------------------------------------------------------
cx, cy, cz = x_b.copy(), y_b.copy(), z_b.copy()

# ----------------------------------------------------------
# Interior RBF pieces (ASYMMETRIC — must match K)
# ----------------------------------------------------------
dx = x_int[:, None] - alpha * cx[None, :] + hx
dy = y_int[:, None] - alpha * cy[None, :] + hy
dz = z_int[:, None] - alpha * cz[None, :] + hz

r2 = dx**2 + dy**2 + dz**2
Phi = np.exp(-lam * r2)

Phi_int = Phi

Phi_xx = (4*lam**2 * dx**2 - 2*lam) * Phi
Phi_yy = (4*lam**2 * dy**2 - 2*lam) * Phi
Phi_zz = (4*lam**2 * dz**2 - 2*lam) * Phi

# ==========================================================
# SCHUR REDUCTION
# ==========================================================
KinvB = lu_solve((LU_K, piv_K), Pbn)

effective_P = P_int - Phi_int @ KinvB

L_reduced = (
    dxx_int + dyy_int + dzz_int
    - Phi_xx @ KinvB
    - Phi_yy @ KinvB
    - Phi_zz @ KinvB
)

# ==========================================================
#  Gauss–Newton Iteration
# ==========================================================
def residual(beta):

    ux_poly = Dx_b @ beta
    uy_poly = Dy_b @ beta
    uz_poly = Dz_b @ beta

    flux_poly = (
        ux_poly * boundary_normals[:,0] +
        uy_poly * boundary_normals[:,1] +
        uz_poly * boundary_normals[:,2]
    )

    rhs_neu = g_b - flux_poly

    q = lu_solve((LU_K, piv_K), rhs_neu)

    u = P_int @ beta + Phi_int @ q

    uxx = dxx_int @ beta + Phi_xx @ q
    uyy = dyy_int @ beta + Phi_yy @ q
    uzz = dzz_int @ beta + Phi_zz @ q

    R = uxx + uyy + uzz - u**2 - A_int

    return R, q, u


print("\nReduced Newton\n")

for it in range(2):

    R, q, u = residual(beta)

    res=np.linalg.norm(R)
    res_hist_stage2.append(res)
    print(f"Reduced Iter {it}: ||R||={res:.2e}")

    if res < tol:
        break

    J = L_reduced - (2*u)[:,None] * effective_P

    delta = np.linalg.lstsq(J, -R, rcond=None)[0]

    beta += delta

# ==========================================================
# Final q evaluation
# ==========================================================
ux_poly = Dx_b @ beta
uy_poly = Dy_b @ beta
uz_poly = Dz_b @ beta

flux_poly = (
    ux_poly * boundary_normals[:,0] +
    uy_poly * boundary_normals[:,1] +
    uz_poly * boundary_normals[:,2]
)

rhs_neu = g_b - flux_poly

q = lu_solve((LU_K, piv_K), rhs_neu)

# ==========================================================
# GLOBAL ERROR AFTER EXACT BC
# ==========================================================
Phi_all = rbf_matrix(
    x_all, y_all, z_all,
    cx, cy, cz,
    lam, alpha, hx, hy, hz
)

u_all = P_all @ beta + Phi_all @ q

end2 = time.time()
print("")
print("Stage 2 time:",np.round((end2-start2),2),"s")
print("")

abs_error_after = np.abs(u_all - u_exact_all)

print(f"RMS error = {np.sqrt(np.mean(abs_error_after**2)):.2e}")
print(f"Linf error = {np.max(abs_error_after):.2e}")

# ==========================================================
# Neumann BC error
# ==========================================================
flux_rbf = K @ q

flux_total = flux_poly + flux_rbf

neumann_error = np.abs(flux_total - g_b)

print(f"\nMax Neumann error  = {np.max(neumann_error):.2e}")
print(f"Mean Neumann error = {np.mean(neumann_error):.2e}")

# ==========================================================
# Save Stage 2 Neumann boundary error
# ==========================================================
stage2_neumann = np.column_stack((
    x_b,
    y_b,
    z_b,
    neumann_error
))

np.savetxt(
    "data/stage2_neumann_error.dat",
    stage2_neumann,
    fmt="%.16e",
    header="x y z neumann_error"
)

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
    "fig4n_1.pdf",dpi=300,
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
    "fig4n_2.pdf",dpi=300,
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
    "fig4n_res.pdf",dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[ ]:




