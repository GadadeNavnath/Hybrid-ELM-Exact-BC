#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

torch.manual_seed(1234)

# ==========================================================
# Neural Network
# ==========================================================

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(3, 146, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(146, 1, dtype=torch.float64, bias=False)
        )

    def forward(self, x):
        return self.layers(x)


# ==========================================================
# Load collocation points
# ==========================================================

interior_points = torch.from_numpy(
    np.load("data/shell_interior_points.npy")
).to(torch.float64)

boundary_points = torch.from_numpy(
    np.load("data/shell_boundary_points.npy")
).to(torch.float64)

normals = torch.from_numpy(
    np.load("data/shell_boundary_normals.npy")
).to(torch.float64)

K = torch.from_numpy(
    np.load("data/K.npy")
).to(torch.float64)

interior_points.requires_grad_(True)
boundary_points.requires_grad_(True)

print("")
print("Summary:")
print("")
print("Interior points :", interior_points.shape[0])
print("Boundary points :", boundary_points.shape[0])
print("")


# ==========================================================
# Exact solution
# ==========================================================

def u_exact(x, y, z):
    return torch.exp(x) * y**2 + (z**2 + 2.0) * torch.sin(y)


def ux_exact(x, y, z):
    return torch.exp(x) * y**2


def uy_exact(x, y, z):
    return 2.0 * torch.exp(x) * y + (z**2 + 2.0) * torch.cos(y)


def uz_exact(x, y, z):
    return 2.0 * z * torch.sin(y)


# ==========================================================
# Interior quantities
# ==========================================================

x = interior_points[:, 0]
y = interior_points[:, 1]
z = interior_points[:, 2]

f_values = (
    torch.exp(x) * (2.0 + y**2)
    - z**2 * torch.sin(y)
    - (
        torch.exp(x) * y**2
        + (z**2 + 2.0) * torch.sin(y)
    )**2
).detach()

# ==========================================================
# Exact Neumann boundary data
# ==========================================================

xb = boundary_points[:, 0]
yb = boundary_points[:, 1]
zb = boundary_points[:, 2]

grad_exact = torch.stack(
    (
        ux_exact(xb, yb, zb),
        uy_exact(xb, yb, zb),
        uz_exact(xb, yb, zb),
    ),
    dim=1,
)

boundary_flux = (
    torch.sum(grad_exact * normals, dim=1)
    .unsqueeze(1)
    .detach()
)


# ==========================================================
# Initialize model and optimizer
# ==========================================================

model = NN()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
)

adam_start = time.time()

tol_adam = 1e-5
lambda_bc = 100.0

# ==========================================================
# Training loop
# ==========================================================

for epoch in range(20001):

    optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------------
    # Interior prediction
    # ------------------------------------------------------

    N = model(interior_points)

    grad = torch.autograd.grad(
        N,
        interior_points,
        torch.ones_like(N),
        create_graph=True,
    )[0]

    N_x = grad[:, 0]
    N_y = grad[:, 1]
    N_z = grad[:, 2]

    N_xx = torch.autograd.grad(
        N_x,
        interior_points,
        torch.ones_like(N_x),
        create_graph=True,
    )[0][:, 0]

    N_yy = torch.autograd.grad(
        N_y,
        interior_points,
        torch.ones_like(N_y),
        create_graph=True,
    )[0][:, 1]

    N_zz = torch.autograd.grad(
        N_z,
        interior_points,
        torch.ones_like(N_z),
        create_graph=True,
    )[0][:, 2]

    N = N.view(-1)

    # ------------------------------------------------------
    # PDE loss
    # ------------------------------------------------------

    loss_pde = torch.mean(
    (
    N_xx
    + N_yy
    + N_zz
    - N**2
    - f_values
    )**2
    )
    # ------------------------------------------------------
    # Neumann boundary loss
    # ------------------------------------------------------

    N_boundary = model(boundary_points)

    grad_boundary = torch.autograd.grad(
        N_boundary,
        boundary_points,
        torch.ones_like(N_boundary),
        create_graph=True,
    )[0]

    boundary_flux_pred = torch.sum(
        grad_boundary * normals,
        dim=1,
        keepdim=True,
    )

    loss_bc = torch.mean(
        (boundary_flux_pred - boundary_flux) ** 2
    )

    # ------------------------------------------------------
    # Total loss
    # ------------------------------------------------------

    loss = loss_pde + lambda_bc * loss_bc

    loss.backward()

    optimizer.step()

    # ------------------------------------------------------
    # Stopping criterion
    # ------------------------------------------------------

    if loss.item() < tol_adam:

        print(f"Adam converged at epoch {epoch}")
        print(
            f"Loss = {loss.item():.2e} | "
            f"PDE = {loss_pde.item():.2e} | "
            f"BC = {loss_bc.item():.2e}"
        )
        print("")
        break

    # ------------------------------------------------------
    # Print progress
    # ------------------------------------------------------

    if epoch % 5000 == 0:

        print(
            f"Epoch {epoch:5d} | "
            f"Loss = {loss.item():.2e} | "
            f"PDE = {loss_pde.item():.2e} | "
            f"BC = {loss_bc.item():.2e}"
        )
        print("")

# ==========================================================
# Store solution and compute error (Interior + Boundary)
# ==========================================================

all_points = torch.cat((interior_points, boundary_points), dim=0)

with torch.no_grad():
    pred_all = model(all_points).view(-1)

exact_all = u_exact(
    all_points[:, 0],
    all_points[:, 1],
    all_points[:, 2],
).view(-1)

error_all = torch.abs(pred_all - exact_all)

adam_end = time.time()

print(f"Adam Time : {adam_end - adam_start:.2f} sec")
print("")

# ==========================================================
# Stage 1 Neumann Boundary Error
# ==========================================================

N_boundary_stage1 = model(boundary_points)

grad_boundary_stage1 = torch.autograd.grad(
    N_boundary_stage1,
    boundary_points,
    torch.ones_like(N_boundary_stage1),
    create_graph=False,
)[0]

boundary_flux_pred_stage1 = torch.sum(
    grad_boundary_stage1 * normals,
    dim=1,
    keepdim=True,
)

boundary_error_stage1 = torch.abs(
    boundary_flux_pred_stage1 - boundary_flux
)

boundary_data_stage1 = torch.column_stack((
    boundary_points[:, 0].detach(),
    boundary_points[:, 1].detach(),
    boundary_points[:, 2].detach(),
    boundary_flux.detach().view(-1),
    boundary_flux_pred_stage1.detach().view(-1),
    boundary_error_stage1.detach().view(-1),
))

np.savetxt(
    "data/neumann_boundary_error_stage1.dat",
    boundary_data_stage1.cpu().numpy(),
    fmt="%.16e",
    header="x y z exact_flux predicted_flux abs_error",
)

print("***************************************************************************************")
print("Summary (Stage 1 : All Points)")
print("***************************************************************************************")

print(
    f"L2 error norm (all points)     : "
    f"{torch.norm(error_all).item():.2e}"
)

print(
    f"L2 relative error (all points) : "
    f"{(torch.norm(error_all)/torch.norm(exact_all)).item():.2e}"
)

exact_adjusted_all = 1.0 + torch.abs(exact_all)

print(
    f"Percentage relative error      : "
    f"{torch.max(100.0*error_all/exact_adjusted_all).item():.2e} %"
)

print(
    f"Maximum deviation              : "
    f"{torch.max(error_all).item():.2e}"
)

print("")

# ==========================================================
# Save Stage 1 solution (all points)
# ==========================================================

stage1_all = torch.column_stack((
    all_points[:, 0].detach(),
    all_points[:, 1].detach(),
    all_points[:, 2].detach(),
    pred_all.detach(),
))

np.savetxt(
    "data/stage1_solution_all.dat",
    stage1_all.cpu().numpy(),
    fmt="%.16e",
    header="x y z u",
)

print("")
print("Starting Stage 2 (Exact BC + LBFGS)")
print("")

# ==========================================================
# Kernel matrix
# ==========================================================

print("")
print(f"cond(K) = {torch.linalg.cond(K).item():.2e}")
print("")

LU, pivots = torch.linalg.lu_factor(K)

# ==========================================================
# Parameter information
# ==========================================================

params = list(model.parameters())

param_sizes = [
    p.numel()
    for p in params
]

total_params = sum(param_sizes)

N_int = interior_points.shape[0]
N_bd = boundary_points.shape[0]

# ==========================================================
# Kernel parameter
# ==========================================================

lambda_param = 6500.0

alpha = 0.9999

shift = torch.tensor(
    [1e-5, 1e-5, 1e-5],
    dtype=torch.float64,
)


x_i = interior_points.unsqueeze(1)
r_j = boundary_points.unsqueeze(0)

diffs = x_i - alpha * r_j + shift.view(1, 1, 3)

squared_dist = torch.sum(
    diffs**2,
    dim=2,
)

p = torch.exp(
    -lambda_param * squared_dist
)

lambda_term = (
    4.0 * lambda_param**2 * squared_dist
    - 6.0 * lambda_param
)

exp_lambda = p * lambda_term
# ==========================================================
# LBFGS optimizer
# ==========================================================

optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=20,
    tolerance_grad=1e-9,
    tolerance_change=1e-9,
    history_size=500,
    line_search_fn="strong_wolfe",
)

# ==========================================================
# Closure
# ==========================================================

def closure():

    optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------------
    # Boundary prediction
    # ------------------------------------------------------

    N_boundary = model(boundary_points)

    grad_boundary = torch.autograd.grad(
        N_boundary,
        boundary_points,
        torch.ones_like(N_boundary),
        create_graph=True,
    )[0]

    boundary_flux_pred = torch.sum(
        grad_boundary * normals,
        dim=1,
        keepdim=True,
    )

    # ------------------------------------------------------
    # Boundary correction coefficients
    # ------------------------------------------------------

    b = boundary_flux - boundary_flux_pred

    q = torch.linalg.lu_solve(
        LU,
        pivots,
        b,
    )

    # ------------------------------------------------------
    # Interior prediction
    # ------------------------------------------------------

    N_interior = model(interior_points)

    correction = p @ q

    psi_trial = (
        N_interior
        + correction
    ).view(-1)

    # ------------------------------------------------------
    # First derivatives
    # ------------------------------------------------------

    grad = torch.autograd.grad(
        psi_trial,
        interior_points,
        torch.ones_like(psi_trial),
        create_graph=True,
    )[0]

    psi_x = grad[:, 0]
    psi_y = grad[:, 1]
    psi_z = grad[:, 2]

    # ------------------------------------------------------
    # Second derivatives
    # ------------------------------------------------------

    psi_xx = torch.autograd.grad(
        psi_x,
        interior_points,
        torch.ones_like(psi_x),
        create_graph=True,
    )[0][:, 0]

    psi_yy = torch.autograd.grad(
        psi_y,
        interior_points,
        torch.ones_like(psi_y),
        create_graph=True,
    )[0][:, 1]

    psi_zz = torch.autograd.grad(
        psi_z,
        interior_points,
        torch.ones_like(psi_z),
        create_graph=True,
    )[0][:, 2]

    # ------------------------------------------------------
    # PDE residual
    # ------------------------------------------------------

    residual = (
        psi_xx
        + psi_yy
        + psi_zz
        - psi_trial**2
        - f_values
    )

    loss = torch.mean(
        residual**2
    )

    # ======================================================
    # Compute dq/dp
    # ======================================================

    jacobian_boundary = torch.zeros(
        N_bd,
        3,
        total_params,
        dtype=torch.float64,
    )

    for i in range(N_bd):

        point = boundary_points[i].unsqueeze(0)
        point.requires_grad_(True)

        N_i = model(point)

        grad_input = torch.autograd.grad(
            N_i,
            point,
            create_graph=True,
        )[0]

        for j in range(3):

            dNj = grad_input[0, j]

            grad_param = torch.autograd.grad(
                dNj,
                params,
                retain_graph=True,
                create_graph=True,
            )

            grad_param = torch.cat(
                [g.view(-1) for g in grad_param]
            )

            jacobian_boundary[i, j] = grad_param

    jacobian_boundary = torch.sum(
        normals.unsqueeze(-1) * jacobian_boundary,
        dim=1,
    )

    rhs = -jacobian_boundary.detach()

    del_q_del_p = torch.linalg.lu_solve(
        LU,
        pivots,
        rhs,
    )
    # ======================================================
    # Compute dE/dp
    # ======================================================

    error_term = residual.view(-1, 1)

    # ------------------------------------------------------
    # Network derivatives
    # ------------------------------------------------------

    grad_N = torch.autograd.grad(
        N_interior,
        interior_points,
        torch.ones_like(N_interior),
        create_graph=True,
    )[0]

    N_x = grad_N[:, 0]
    N_y = grad_N[:, 1]
    N_z = grad_N[:, 2]

    N_xx = torch.autograd.grad(
        N_x,
        interior_points,
        torch.ones_like(N_x),
        create_graph=True,
    )[0][:, 0]

    N_yy = torch.autograd.grad(
        N_y,
        interior_points,
        torch.ones_like(N_y),
        create_graph=True,
    )[0][:, 1]

    N_zz = torch.autograd.grad(
        N_z,
        interior_points,
        torch.ones_like(N_z),
        create_graph=True,
    )[0][:, 2]

        # ------------------------------------------------------
    # Jacobian of N_xx
    # ------------------------------------------------------

    jacobian_N_xx = []

    for i in range(N_int):

        grads = torch.autograd.grad(
            N_xx[i],
            params,
            retain_graph=True,
            allow_unused=True,
        )

        grads = [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(grads, params)
        ]

        jacobian_N_xx.append(
            torch.cat(
                [g.reshape(-1) for g in grads]
            )
        )

    jacobian_N_xx = torch.stack(jacobian_N_xx)

    # ------------------------------------------------------
    # Jacobian of N_yy
    # ------------------------------------------------------

    jacobian_N_yy = []

    for i in range(N_int):

        grads = torch.autograd.grad(
            N_yy[i],
            params,
            retain_graph=True,
            allow_unused=True,
        )

        grads = [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(grads, params)
        ]

        jacobian_N_yy.append(
            torch.cat(
                [g.reshape(-1) for g in grads]
            )
        )

    jacobian_N_yy = torch.stack(jacobian_N_yy)

    # ------------------------------------------------------
    # Jacobian of N_zz
    # ------------------------------------------------------

    jacobian_N_zz = []

    for i in range(N_int):

        grads = torch.autograd.grad(
            N_zz[i],
            params,
            retain_graph=True,
            allow_unused=True,
        )

        grads = [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(grads, params)
        ]

        jacobian_N_zz.append(
            torch.cat(
                [g.reshape(-1) for g in grads]
            )
        )

    jacobian_N_zz = torch.stack(jacobian_N_zz)

    # ------------------------------------------------------
    # Jacobian of N
    # ------------------------------------------------------

    jacobian_N = []

    for i in range(N_int):

        grads = torch.autograd.grad(
            N_interior[i],
            params,
            torch.ones_like(N_interior[i]),
            retain_graph=True,
        )

        jacobian_N.append(
            torch.cat(
                [g.reshape(-1) for g in grads]
            )
        )

    jacobian_N = torch.stack(jacobian_N)

        # ------------------------------------------------------
    # Assemble dE/dθ
    # ------------------------------------------------------

    first_sum = (
        jacobian_N_xx
        + jacobian_N_yy
        + jacobian_N_zz
        + exp_lambda @ del_q_del_p
    )

    psi_trial = psi_trial.view(-1, 1)

    second_sum = (
        -2.0
        * psi_trial
        * (
            jacobian_N
            + p @ del_q_del_p
        )
    )

    total_sum = error_term * (
        first_sum
        + second_sum
    )

    delE_delp = (
        2.0
        * total_sum.sum(dim=0)
    ) / N_int

    # ------------------------------------------------------
    # Assign gradients
    # ------------------------------------------------------

    dE_dp_split = torch.split(
        delE_delp,
        param_sizes,
    )

    for param, grad in zip(params, dE_dp_split):

        param.grad = grad.view_as(param)

    return loss

lbfgs_start = time.time()

tol_lbfgs = 1e-3
num_epochs = 10

for epoch in range(num_epochs):

    loss = optimizer.step(closure)

    # ------------------------------------------------------
    # Stopping criterion
    # ------------------------------------------------------

    if loss.item() < tol_lbfgs:

        print(f"LBFGS converged at iteration {epoch}")
        print(f"Loss = {loss.item():.2e}")
        print("")
        break

    # ------------------------------------------------------
    # Print progress
    # ------------------------------------------------------

    if epoch % 3 == 0:

        print(
            f"Iteration {epoch:3d} | "
            f"Loss = {loss.item():.2e}"
        )
        print("")

# ==========================================================
# Final Stage 2 Neumann Boundary Error
# ==========================================================
N_boundary = model(boundary_points)

# ----------------------------------------------------------
# Boundary correction coefficients
# ----------------------------------------------------------
grad_boundary = torch.autograd.grad(
    N_boundary,
    boundary_points,
    grad_outputs=torch.ones_like(N_boundary),
    create_graph=True,
)[0]

boundary_flux_pred = torch.sum(
    grad_boundary * normals,
    dim=1,
    keepdim=True,
)

b_final = boundary_flux - boundary_flux_pred

q_final = torch.linalg.lu_solve(
    LU,
    pivots,
    b_final,
)
q_final = q_final.detach()
# ----------------------------------------------------------
# RBF correction
# ----------------------------------------------------------
x_b = boundary_points.unsqueeze(1)
r_b = boundary_points.detach().unsqueeze(0)

diffs_b = x_b - alpha * r_b + shift.view(1, 1, 3)

squared_dist_b = torch.sum(
    diffs_b**2,
    dim=2,
)

p_boundary = torch.exp(
    -lambda_param * squared_dist_b
)

correction_boundary = p_boundary @ q_final

# ----------------------------------------------------------
# Corrected trial solution
# ----------------------------------------------------------
psi_boundary = N_boundary + correction_boundary

# ----------------------------------------------------------
# Gradient of corrected trial solution
# ----------------------------------------------------------
psi_grad = torch.autograd.grad(
    psi_boundary,
    boundary_points,
    grad_outputs=torch.ones_like(psi_boundary),
    create_graph=False,
)[0]

# ----------------------------------------------------------
# Neumann flux
# ----------------------------------------------------------
boundary_flux_stage2 = torch.sum(
    psi_grad * normals,
    dim=1,
    keepdim=True,
)

# ----------------------------------------------------------
# Boundary error
# ----------------------------------------------------------
boundary_error = torch.abs(
    boundary_flux_stage2 - boundary_flux
)

boundary_data = torch.column_stack((
    boundary_points[:,0].detach(),
    boundary_points[:,1].detach(),
    boundary_points[:,2].detach(),
    boundary_flux.detach().view(-1),
    boundary_flux_stage2.detach().view(-1),
    boundary_error.detach().view(-1),
))

np.savetxt(
    "data/neumann_boundary_error_stage2.dat",
    boundary_data.cpu().numpy(),
    fmt="%.16e",
    header="x y z exact_flux predicted_flux abs_error",
)

# ==========================================================
# Final corrected solution
# ==========================================================

correction_interior = p @ q_final

N_final = model(interior_points)

psi_trial_final = (
    N_final
    + correction_interior
)

# ==========================================================
# Stage 2 solution (All Points)
# ==========================================================

all_points = torch.cat(
    (interior_points, boundary_points),
    dim=0,
)

psi_all = torch.cat(
    (
        psi_trial_final.view(-1),
        psi_boundary.view(-1),
    ),
    dim=0,
)

stage2_solution_all = torch.column_stack((
    all_points[:, 0].detach(),
    all_points[:, 1].detach(),
    all_points[:, 2].detach(),
    psi_all.detach(),
))

np.savetxt(
    "data/stage2_solution_all.dat",
    stage2_solution_all.cpu().numpy(),
    fmt="%.16e",
    header="x y z u",
)

lbfgs_end = time.time()

print(f"LBFGS Time : {lbfgs_end - lbfgs_start:.2f} sec")
print("")
# ==========================================================
# Save Stage 2 model and correction coefficients
# ==========================================================

torch.save(
    model.state_dict(),
    "data/stage2_model.pt",
)

torch.save(
    q_final,
    "data/q_final.pt",
)


# ==========================================================
# Error (All Points)
# ==========================================================

stage2_all = psi_all.view(-1)

exact_all = u_exact(
    all_points[:, 0],
    all_points[:, 1],
    all_points[:, 2],
).view(-1)

error_all = torch.abs(
    stage2_all - exact_all
)

print("***************************************************************************************")
print("Summary (LBFGS Optimizer : All Points)")
print("***************************************************************************************")

print(
    f"L2 error norm (all points)     : "
    f"{torch.norm(error_all).item():.2e}"
)

print(
    f"L2 relative error (all points) : "
    f"{(torch.norm(error_all) / torch.norm(exact_all)).item():.2e}"
)

exact_adjusted = 1.0 + torch.abs(exact_all)

relative_error = (
    100.0 * error_all / exact_adjusted
)

print(
    f"Percentage relative error      : "
    f"{torch.max(relative_error).item():.2e} %"
)

print(
    f"Maximum deviation              : "
    f"{torch.max(error_all).item():.2e}"
)

print("")

