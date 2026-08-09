#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

torch.manual_seed(1234)

# ==========================================================
# Define the neural network
# ==========================================================

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(2, 94, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(94, 1, dtype=torch.float64, bias=False)
        )

    def forward(self, x):
        return self.layers(x)


# ==========================================================
# Load collocation points
# ==========================================================

interior_points = torch.from_numpy(
    np.load("data/sweden_interior_points.npy")
).to(torch.float64)

boundary_points = torch.from_numpy(
    np.load("data/sweden_boundary_points.npy")
).to(torch.float64)

interior_points.requires_grad_(True)
boundary_points.requires_grad_(False)

print("")
print("Summary:")
print("")
print("Interior points :", interior_points.shape[0])
print("Boundary points :", boundary_points.shape[0])
print("")


# ==========================================================
# Exact solution, boundary values and RHS
# ==========================================================

x = interior_points[:, 0]
y = interior_points[:, 1]

boundary_values = (
    torch.sin(torch.pi * boundary_points[:, 0])
    * torch.sin(torch.pi * boundary_points[:, 1])
).unsqueeze(1).detach()

f_values = (
    2.0 * torch.pi**2
    * torch.sin(torch.pi * x)
    * torch.sin(torch.pi * y)
    + torch.sin(torch.pi * x)**3
    * torch.sin(torch.pi * y)**3
).detach()

exact = (
    torch.sin(torch.pi * x)
    * torch.sin(torch.pi * y)
).detach()


# ==========================================================
# Initialize model and optimizer
# ==========================================================

model = NN()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

adam_start = time.time()

tol_adam = 1e-5


# ==========================================================
# Training loop
# ==========================================================

for epoch in range(20001):

    optimizer.zero_grad(set_to_none=True)

    N = model(interior_points)

    grad = torch.autograd.grad(
        N,
        interior_points,
        torch.ones_like(N),
        create_graph=True,
    )[0]

    N_x, N_y = grad[:,0], grad[:,1]

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

    N = N.view(-1)
    N3 = N.pow(3)

    loss_pde = torch.mean(
        (-N_xx - N_yy + N3 - f_values) ** 2
    )

    N_boundary = model(boundary_points)

    loss_bc = torch.mean(
        (N_boundary - boundary_values) ** 2
    )

    lambda_bc = 100.0

    loss = loss_pde + lambda_bc * loss_bc

    loss.backward()

    optimizer.step()

    if loss.item() < tol_adam:
        print(f"Adam converged at epoch {epoch}")
        print(
            f"Loss = {loss.item():.2e} | "
            f"PDE = {loss_pde.item():.2e} | "
            f"BC = {loss_bc.item():.2e}"
        )
        print("")
        break

    if epoch % 5000 == 0:
        print(
            f"Epoch {epoch:5d} | "
            f"Loss = {loss.item():.2e} | "
            f"PDE = {loss_pde.item():.2e} | "
            f"BC = {loss_bc.item():.2e}"
        )
        print("")

# ==========================================================
# Evaluation
# ==========================================================

with torch.no_grad():
    N_eval = model(interior_points).view(-1)

exact = exact.view(-1)

# ==========================================================
# Store solution and compute error (Interior + Boundary)
# ==========================================================

all_points = torch.cat(
    (interior_points, boundary_points),
    dim=0
)

with torch.no_grad():
    pred_all = model(all_points).view(-1)

exact_all = (
    torch.sin(torch.pi * all_points[:, 0])
    * torch.sin(torch.pi * all_points[:, 1])
).view(-1)

error_all = torch.abs(
    pred_all - exact_all
)

adam_end = time.time()

print(f"Adam Time : {adam_end - adam_start:.2f} sec")
print("")

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
    f"{torch.max(100.0 * error_all / exact_adjusted_all).item():.2e} %"
)

print(
    f"Maximum deviation              : "
    f"{torch.max(error_all).item():.2e}"
)

print("")

# ==========================================================
# Store Stage 1 solution (all points)
# ==========================================================

stage1_all = torch.column_stack((
    all_points[:, 0].detach(),
    all_points[:, 1].detach(),
    pred_all.detach()
))

np.savetxt(
    "data/stage1_solution.dat",
    stage1_all.numpy(),
    fmt="%.16e",
    header="x y u"
)

print("\nStarting Stage 2 (Exact BC + LBFGS)...\n")

# ==========================================================
# Stage 2 : Exact Boundary Condition Enforcement
# ==========================================================
# Boundary kernel matrix
# ==========================================================

lambda_param = 1000.0

diff_bc = boundary_points.unsqueeze(1) - boundary_points.unsqueeze(0)
dist2_bc = torch.sum(diff_bc**2, dim=2)

A = torch.exp(-lambda_param * dist2_bc)

print("")
print(f"cond(A) = {torch.linalg.cond(A).item():.2e}")
print("")

LU, pivots = torch.linalg.lu_factor(A)

# ==========================================================
# Interior kernel matrix
# ==========================================================

diff_int = interior_points.unsqueeze(1) - boundary_points.unsqueeze(0)
dist2_int = torch.sum(diff_int**2, dim=2)

p = torch.exp(-lambda_param * dist2_int)

lambda_term = (
    4.0 * lambda_param**2 * dist2_int
    - 4.0 * lambda_param
)

exp_into_lambda = p * lambda_term
# ==========================================================
# Parameter information
# ==========================================================


params = list(model.parameters())
param_sizes = [p.numel() for p in params]

N_int = interior_points.shape[0]

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
    # Boundary correction coefficients
    # ------------------------------------------------------

    N_boundary = model(boundary_points)

    b = boundary_values - N_boundary

    q = torch.linalg.lu_solve(
        LU,
        pivots,
        b,
    )

    # ------------------------------------------------------
    # Trial solution
    # ------------------------------------------------------

    N_interior = model(interior_points)

    correction = p @ q

    psi_trial = (
        N_interior + correction
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

    # ------------------------------------------------------
    # PDE residual
    # ------------------------------------------------------

    residual = (
        -psi_xx
        -psi_yy
        + psi_trial.pow(3)
        - f_values
    )

    loss = torch.mean(residual**2)

    # ======================================================
    # Compute dq/dp
    # ======================================================

    jacobian_boundary = []

    for i in range(N_boundary.shape[0]):

        grads = torch.autograd.grad(
            N_boundary[i],
            params,
            torch.ones_like(N_boundary[i]),
            retain_graph=True,
        )

        jacobian_boundary.append(
            torch.cat(
                [g.reshape(-1) for g in grads]
            )
        )

    jacobian_boundary = torch.stack(jacobian_boundary)

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
    # N_xx Jacobian
    # ------------------------------------------------------

    grad_N = torch.autograd.grad(
        N_interior,
        interior_points,
        torch.ones_like(N_interior),
        create_graph=True,
    )[0]

    N_x = grad_N[:, 0]
    N_y = grad_N[:, 1]

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

    jacobian_N_xx = []

    for i in range(N_xx.shape[0]):

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
            torch.cat([g.reshape(-1) for g in grads])
        )

    jacobian_N_xx = torch.stack(jacobian_N_xx)

    # ------------------------------------------------------
    # N_yy Jacobian
    # ------------------------------------------------------

    jacobian_N_yy = []

    for i in range(N_yy.shape[0]):

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
            torch.cat([g.reshape(-1) for g in grads])
        )

    jacobian_N_yy = torch.stack(jacobian_N_yy)
    # ------------------------------------------------------
    # N Jacobian
    # ------------------------------------------------------
    jacobian_N = []
    for i in range(N_interior.shape[0]):  
        grad_i = torch.autograd.grad(N_interior[i], params, 
                                     torch.ones_like(N_interior[i]), retain_graph = True)
        grad_i = torch.cat([g.view(-1) for g in grad_i])  # Flatten gradients
        jacobian_N.append(grad_i)

    jacobian_N = torch.stack(jacobian_N)    

    first_sum_second_term = - torch.matmul(exp_into_lambda , del_q_del_p)



    first_sum = (- jacobian_N_xx - jacobian_N_yy) + first_sum_second_term


    psi_trial = psi_trial.view(-1,1)
    squared_psi_trial = psi_trial**2
    second_sum = 3 * squared_psi_trial * (jacobian_N + torch.matmul(p, del_q_del_p))    

    total_sum = error_term * (first_sum + second_sum)
    total_sum_reduced = total_sum.sum(dim=0, keepdim=True)


    delE_delp = (2 * total_sum_reduced)/N_int



    dE_dp = delE_delp.view(-1)  

    # Assign computed gradients to model parameters
    dE_dp_split = torch.split(dE_dp, param_sizes)


    # print(dE_dp_split)
    for param, grad in zip(params, dE_dp_split):
        param.grad = grad.view(param.shape)


    return loss

lbfgs_start = time.time()

tol_lbfgs = 1e-3
num_epochs = 10

for epoch in range(num_epochs):

    loss = optimizer.step(closure)

    # Stopping criterion
    if loss.item() < tol_lbfgs:
        print(f"LBFGS converged at iteration {epoch}")
        print(f"Loss = {loss.item():.2e}")
        print("")
        break

    # Print progress
    if epoch % 3 == 0:
        print(
            f"Iteration {epoch:3d} | "
            f"Loss = {loss.item():.2e}"
        )
        print("")

with torch.no_grad():

    N_boundary_final = model(boundary_points)
    b_final = boundary_values - N_boundary_final
    q_final = torch.linalg.lu_solve(LU, pivots, b_final)

    # Interior
    correction_interior = p @ q_final
    psi_trial_final = model(interior_points) + correction_interior

    # Boundary
    correction_boundary = A @ q_final
    psi_boundary_final = N_boundary_final + correction_boundary

lbfgs_end = time.time()
print(f"LBFGS Time : {lbfgs_end-lbfgs_start:.2f} sec")
print("")

# ==========================================================
# Save Stage 2 model and correction coefficients
# ==========================================================

torch.save(model.state_dict(), "data/stage2_model.pt")
torch.save(q_final, "data/q_final.pt")
# ==========================================================
# Save Stage 2 solution (all points)
# ==========================================================

all_points = torch.cat((interior_points, boundary_points), dim=0)

stage2_solution = torch.column_stack((
    all_points[:, 0].detach(),
    all_points[:, 1].detach(),
    torch.cat((
        psi_trial_final.view(-1),
        psi_boundary_final.view(-1)
    )).detach()
))

np.savetxt(
    "data/stage2_solution.dat",
    stage2_solution.numpy(),
    fmt="%.16e",
    header="x y u"
)

# ==========================================================
# Save boundary correction (all points)
# ==========================================================

boundary_correction = torch.column_stack((
    all_points[:, 0].detach(),
    all_points[:, 1].detach(),
    torch.cat((
        correction_interior.view(-1),
        correction_boundary.view(-1)
    )).detach()
))

np.savetxt(
    "data/boundary_correction.dat",
    boundary_correction.numpy(),
    fmt="%.16e",
    header="x y correction"
)

# ==========================================================
# Error on all points
# ==========================================================

stage2_all = torch.cat((
    psi_trial_final.view(-1),
    psi_boundary_final.view(-1)
))

exact_all = (
    torch.sin(torch.pi * all_points[:, 0])
    * torch.sin(torch.pi * all_points[:, 1])
).view(-1)

error_all = torch.abs(stage2_all - exact_all)

print("***************************************************************************************")
print("Summary (LBFGS Optimizer)")
print("***************************************************************************************")

print(f"L2 error norm (all points)     : {torch.norm(error_all).item():.2e}")
print(f"L2 relative error (all points) : {(torch.norm(error_all)/torch.norm(exact_all)).item():.2e}")

exact_adjusted = 1.0 + torch.abs(exact_all)
relative_error = 100.0 * error_all / exact_adjusted

print(f"Percentage relative error      : {torch.max(relative_error).item():.2e} %")
print(f"Maximum deviation              : {torch.max(error_all).item():.2e}")
print("")


# In[ ]:




