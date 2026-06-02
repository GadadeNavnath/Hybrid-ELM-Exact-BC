#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
N_basis = 10

NB = N_basis ** 3

max_iter = 4

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
# Exact solution and gradient (DEFINED EARLY)
# ==========================================================
def u_exact(x,y,z):
    return np.exp(x)*(y**2) + (z**2 + 2.0)*np.sin(y)

def grad_u_exact(x,y,z):
    ux = np.exp(x)*(y**2)
    uy = 2*y*np.exp(x) + (z**2 + 2.0)*np.cos(y)
    uz = 2*z*np.sin(y)
    return ux, uy, uz

def A_xyz(x,y,z):
    u_ex = u_exact(x,y,z)
    return (2+y**2)*np.exp(x) - (z**2)*np.sin(y) - u_ex**2

# ==========================================================
# Load points
# ==========================================================
interior_pts = np.load("data/shell_interior_points.npy")
boundary_pts = np.load("data/shell_boundary_points.npy")
boundary_normals = np.load("data/shell_boundary_normals.npy")


print("Interior points:", interior_pts.shape[0])
print("Boundary points:", boundary_pts.shape[0])
print("Boundary normals:", boundary_normals.shape[0])
print("Total unknowns:",NB)
print("")

# ==========================================================
# Boundary classification (mixed BC)
# ==========================================================
tol_b = 1e-12
x_b, y_b, z_b = boundary_pts[:,0], boundary_pts[:,1], boundary_pts[:,2]

is_dirichlet = (
    (np.abs(x_b - xmin) < tol_b) |
    (np.abs(y_b - ymin) < tol_b) |
    (np.abs(z_b - zmin) < tol_b)
)

is_neumann = ~is_dirichlet

bnd_D = boundary_pts[is_dirichlet]

bnd_N = boundary_pts[is_neumann]

normals_N = boundary_normals[is_neumann]

print("Dirichlet pts:", bnd_D.shape[0])
print("Neumann pts :", bnd_N.shape[0])
print("")

# ==========================================================
# Mapping
# ==========================================================
def map_to_leg_coords(X):
    x, y, z = X[:,0], X[:,1], X[:,2]
    xi = 2*(x - xmin)/(xmax - xmin) - 1
    yi = 2*(y - ymin)/(ymax - ymin) - 1
    zi = 2*(z - zmin)/(zmax - zmin) - 1
    return xi, yi, zi

xi_int, yi_int, zi_int = map_to_leg_coords(interior_pts)
xi_D, yi_D, zi_D = map_to_leg_coords(bnd_D)
xi_N, yi_N, zi_N = map_to_leg_coords(bnd_N)

# ==========================================================
# Legendre matrices
# ==========================================================
def leg_matrices_3d(xi, yi, zi, N):

    def leg_1d_all(pts, N, s1, s2):
        P_list, D_list, D2_list = [], [], []
        for i in range(N):
            ci = [0]*i + [1]
            dci = legder(ci)
            d2ci = legder(dci)
            P_list.append(legval(pts, ci))
            D_list.append(legval(pts, dci) * s1)
            D2_list.append(legval(pts, d2ci) * s2)
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
# build matrices
P_int, Dx_int, Dy_int, Dz_int, dxx_int, dyy_int, dzz_int = \
    leg_matrices_3d(xi_int, yi_int, zi_int, N_basis)

P_D, _, _, _, _, _, _ = \
    leg_matrices_3d(xi_D, yi_D, zi_D, N_basis)

P_N, Dx_N, Dy_N, Dz_N, _, _, _ = \
    leg_matrices_3d(xi_N, yi_N, zi_N, N_basis)

# ==========================================================
# Data
# ==========================================================
x_int, y_int, z_int = interior_pts[:,0], interior_pts[:,1], interior_pts[:,2]
A_int = A_xyz(x_int, y_int, z_int)

# Neumann data
x_N, y_N, z_N = bnd_N[:,0], bnd_N[:,1], bnd_N[:,2]
nx_N, ny_N, nz_N = normals_N[:,0], normals_N[:,1], normals_N[:,2]
ux_N, uy_N, uz_N = grad_u_exact(x_N, y_N, z_N)
g_N = ux_N*nx_N + uy_N*ny_N + uz_N*nz_N

# Dirichlet data
x_D, y_D, z_D = bnd_D[:,0], bnd_D[:,1], bnd_D[:,2]
u_D = u_exact(x_D, y_D, z_D)

# ==========================================================
# Initial mixed solve
# ==========================================================

Pbn = nx_N[:,None]*Dx_N + ny_N[:,None]*Dy_N + nz_N[:,None]*Dz_N

A_lin = np.vstack([dxx_int + dyy_int + dzz_int, Pbn, P_D])
R_lin = np.concatenate([A_int, g_N, u_D])

beta = np.linalg.lstsq(A_lin, R_lin, rcond=None)[0]

res_hist_stage1 = []
res_hist_stage2 = []

# ==========================================================
# Gauss Newton Iteration 
# ==========================================================
print("Gauss-Newton iteration")
for it in range(max_iter):

    u_int = P_int @ beta
    uxx = dxx_int @ beta
    uyy = dyy_int @ beta
    uzz = dzz_int @ beta

    R_int = uxx + uyy + uzz - u_int**2 - A_int
    R_N = Pbn @ beta - g_N
    R_D = P_D @ beta - u_D
    R = np.concatenate([R_int, R_N, R_D])

    res=np.linalg.norm(R)
    res_hist_stage1.append(res)

    print(f"Iter {it}: ||R||={res:.2e}")

    J_int = dxx_int + dyy_int + dzz_int - (2*u_int)[:,None]*P_int
    J = np.vstack([J_int, Pbn, P_D])

    delta = np.linalg.lstsq(J, -R, rcond=None)[0]
    beta += delta

    
# ==========================================================
# GLOBAL STACK (mixed BC analogue of Dirichlet version)
# ==========================================================
P_all = np.vstack([P_int, P_D, P_N])

x_all = np.concatenate([x_int, x_D, x_N])
y_all = np.concatenate([y_int, y_D, y_N])
z_all = np.concatenate([z_int, z_D, z_N])

u_exact_all = u_exact(x_all, y_all, z_all)
u_before = P_all @ beta

end1 = time.time()
print("")
print("Stage 1 time:",np.round((end1-start1),2),"s")
print("")

# ==========================================================
# ERROR BEFOR EEXACT BC (spectral solution)
# ==========================================================
abs_error_before = np.abs(u_before - u_exact_all)

print("\nGlobal Errors BEFORE exact BC enforcement")
print(f"RMS error = {np.sqrt(np.mean(abs_error_before**2)):.2e}")
print(f"Linf error = {np.max(abs_error_before):.2e}")

# ==========================================================
# STEP 2
# Exact Boundary Condition Enforcement Using Gaussian RBF
# ==========================================================
print("")
print("------STAGE 2------")
# ==========================================================
# Gaussian kernel
# ==========================================================
start2 =time.time()

lam = 6500.0
alpha = 0.9999
h = 1e-04
tol = 1e-10


def rbf_matrix_3d_stable(x, y, z, cx, cy, cz, lam):
    dx = x[:, None] - alpha*cx[None, :] + h
    dy = y[:, None] - alpha*cy[None, :] + h
    dz = z[:, None] - alpha*cz[None, :] + h
    return np.exp(-lam * (dx**2 + dy**2 + dz**2))

# ==========================================================
# Centers = ALL boundary points
# ==========================================================
x_b = np.concatenate([x_D, x_N])
y_b = np.concatenate([y_D, y_N])
z_b = np.concatenate([z_D, z_N])

cx, cy, cz = x_b.copy(), y_b.copy(), z_b.copy()

# ==========================================================
# Dirichlet block
# ==========================================================
Phi_D = rbf_matrix_3d_stable(x_D, y_D, z_D, cx, cy, cz, lam)

# ==========================================================
# Neumann block (STABILIZED — IMPORTANT)
# ==========================================================
dxN = x_N[:, None] - alpha*cx[None, :] + h
dyN = y_N[:, None] - alpha*cy[None, :] + h
dzN = z_N[:, None] - alpha*cz[None, :] + h
PhiN = np.exp(-lam*(dxN**2 + dyN**2 + dzN**2))

PhiNx = (-2*lam*dxN) * PhiN
PhiNy = (-2*lam*dyN) * PhiN
PhiNz = (-2*lam*dzN) * PhiN

nx = normals_N[:,0][:,None]
ny = normals_N[:,1][:,None]
nz = normals_N[:,2][:,None]

Phi_Nn = PhiNx*nx + PhiNy*ny + PhiNz*nz

# ==========================================================
# Mixed boundary matrix
# ==========================================================
A_mixed = np.vstack([Phi_D, Phi_Nn])

LU_mix, piv_mix = lu_factor(A_mixed)

# ==========================================================
# Boundary sensitivity wrt beta
# ==========================================================
B_mixed = np.vstack([P_D, Pbn])
AinvB = lu_solve((LU_mix, piv_mix), B_mixed)

# ==========================================================
# Interior RBF matrices (STABILIZED)
# ==========================================================

dx = x_int[:, None] - alpha*cx[None, :] + h
dy = y_int[:, None] - alpha*cy[None, :] + h
dz = z_int[:, None] - alpha*cz[None, :] + h

Phi = np.exp(-lam*(dx**2 + dy**2 + dz**2))
Phi_int = Phi

Phi_xx = (4*lam**2*dx**2 - 2*lam) * Phi
Phi_yy = (4*lam**2*dy**2 - 2*lam) * Phi
Phi_zz = (4*lam**2*dz**2 - 2*lam) * Phi

# ==========================================================
# Effective reduced operators
# ==========================================================
effective_P = P_int - Phi_int @ AinvB

L_reduced = (
    dxx_int + dyy_int + dzz_int
    - Phi_xx @ AinvB
    - Phi_yy @ AinvB
    - Phi_zz @ AinvB
)

# ==========================================================
# Nonlinear residual
# ==========================================================
def residual_mixed(beta):

    rhs_D = u_D - P_D @ beta
    rhs_N = g_N - Pbn @ beta
    rhs = np.concatenate([rhs_D, rhs_N])

    q = lu_solve((LU_mix, piv_mix), rhs)

    u = P_int @ beta + Phi_int @ q
    uxx = dxx_int @ beta + Phi_xx @ q
    uyy = dyy_int @ beta + Phi_yy @ q
    uzz = dzz_int @ beta + Phi_zz @ q

    R = uxx + uyy + uzz - u**2 - A_int
    return R, q, u

# ==========================================================
# Gauss-Newton iteration
# ==========================================================
print("\nMixed BC Newton\n")

for it in range(2):

    R, q, u = residual_mixed(beta)
    res=np.linalg.norm(R)
    res_hist_stage2.append(res)
    print(f"Reduced Iter {it+1}: ||R||={res:.2e}")

    if res < tol:
        break

    # ===== correct nonlinear Jacobian =====
    J = L_reduced - (2*u)[:,None] * effective_P

    delta = np.linalg.lstsq(J, -R, rcond=None)[0]
    beta += delta

    
# --- recompute q for final beta (IMPORTANT)
rhs_D = u_D - P_D @ beta
rhs_N = g_N - Pbn @ beta
rhs = np.concatenate([rhs_D, rhs_N])

q = lu_solve((LU_mix, piv_mix), rhs) 

# ==========================================================
# Final global error
# ==========================================================
P_all = np.vstack([P_int, P_D, P_N])

x_all = np.concatenate([x_int, x_D, x_N])
y_all = np.concatenate([y_int, y_D, y_N])
z_all = np.concatenate([z_int, z_D, z_N])


Phi_all = rbf_matrix_3d_stable(x_all, y_all, z_all, cx, cy, cz, lam)

u_all = P_all @ beta + Phi_all @ q

end2 = time.time()
print("")
print("Stage 2 time:",np.round((end2-start2),2),"s")
print("")

abs_error_after = np.abs(u_all - u_exact_all)

print(f"RMS error   = {np.sqrt(np.mean(abs_error_after**2)):.2e}")
print(f"Linf error = {np.max(abs_error_after):.2e}")

# ==========================================================
# Dirichlet BC error
# ==========================================================
u_D_poly = P_D @ beta

u_D_rbf = Phi_D @ q

u_D_total = u_D_poly + u_D_rbf

dirichlet_error = np.abs(
    u_D_total - u_D
)

print(
    f"\nMax Dirichlet error  = "
    f"{np.max(dirichlet_error):.2e}"
)

print(
    f"Mean Dirichlet error = "
    f"{np.mean(dirichlet_error):.2e}"
)


# ==========================================================
# Neumann BC error
# ==========================================================
flux_poly = Pbn @ beta

flux_rbf = Phi_Nn @ q

flux_total = flux_poly + flux_rbf

neumann_error = np.abs(
    flux_total - g_N
)

print(
    f"\nMax Neumann error  = "
    f"{np.max(neumann_error):.2e}"
)

print(
    f"Mean Neumann error = "
    f"{np.mean(neumann_error):.2e}"
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
    "fig4m_1.pdf",dpi=300,
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
    "fig4m_2.pdf",dpi=300,
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
    "fig4m_res.pdf",dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[ ]:




