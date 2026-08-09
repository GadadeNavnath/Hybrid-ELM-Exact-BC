#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

torch.manual_seed(12)

# Define the neural network
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 143, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(143, 1, dtype=torch.float64, bias=False)
        )

    def forward(self, x):
        return self.layers(x)


# ===========================
# Load collocation points
# ===========================
interior_points = torch.from_numpy(
    np.load("data/cardioid_interior_points.npy")
).to(torch.float64)

boundary_points = torch.from_numpy(
    np.load("data/cardioid_boundary_points.npy")
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
    torch.log(
        1 + boundary_points[:, 0]**2 + boundary_points[:, 1]**2
    )
    .unsqueeze(1)
    .detach()
)

f_values = (
    1
    + x**2
    + y**2
    + 4.0 / (1 + x**2 + y**2)**2
).detach()

exact = (
    torch.log(1 + x**2 + y**2)
).detach()

# ==========================================================
# Initialize model and optimizer
# ==========================================================

model = NN()

optimizer = optim.Adam(model.parameters(), lr=0.001)

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

    expN = torch.exp(N).view(-1)

    loss_pde = torch.mean(
        (N_xx + N_yy + expN - f_values) ** 2
    )

    # Boundary loss
    N_boundary = model(boundary_points)

    loss_bc = torch.mean(
    (N_boundary-boundary_values)**2)

    lambda_bc = 100.0

    loss = loss_pde + lambda_bc * loss_bc

    loss.backward()

    optimizer.step()

    # Stopping criterion
    if loss.item() < tol_adam:
        print(f"Adam converged at epoch {epoch}")
        print(
            f"Loss = {loss.item():.2e} | "
            f"PDE = {loss_pde.item():.2e} | "
            f"BC = {loss_bc.item():.2e}"
        )
        print("")
        break

    # Print progress
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

all_points = torch.cat((interior_points, boundary_points), dim=0)

with torch.no_grad():
    pred_all = model(all_points).view(-1)

exact_all = torch.log(
    1 + all_points[:, 0]**2 + all_points[:, 1]**2
).view(-1)

error_all = torch.abs(pred_all - exact_all)

adam_end = time.time()

print(f"Adam Time : {adam_end - adam_start:.2f} sec")
print("")

print("***************************************************************************************")
print("Summary (Stage 1 : All Points)")
print("***************************************************************************************")

print(f"L2 error norm (all points)     : {torch.norm(error_all).item():.2e}")
print(f"L2 relative error (all points) : {(torch.norm(error_all)/torch.norm(exact_all)).item():.2e}")

exact_adjusted_all = 1.0 + torch.abs(exact_all)

print(f"Percentage relative error      : {torch.max(100.0*error_all/exact_adjusted_all).item():.2e} %")
print(f"Maximum deviation              : {torch.max(error_all).item():.2e}")
print("")

# ==========================================================
# Save Stage 1 solution (all points)
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

# Define A matrix 
lambda_param = 1500.0

diffs = boundary_points.unsqueeze(1) - boundary_points.unsqueeze(0)  # Shape: (N, N, 2)
squared_dist = torch.sum(diffs ** 2, dim=2)  # Shape: (N, N)

A = torch.exp(-lambda_param * squared_dist)

# Print matrix property
cond = torch.linalg.cond(A)
print("")
print(f"cond(A) = {cond.item():.2e}")
print("")


# Define optimizer
optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, 
                              tolerance_grad=1e-9, tolerance_change=1e-9, 
                              history_size=500, line_search_fn="strong_wolfe")

LU, pivots = torch.linalg.lu_factor(A)


diffs_int = (interior_points.unsqueeze(1) - boundary_points.unsqueeze(0))
squared_dist_int = torch.sum(diffs_int ** 2, dim=2)
p = torch.exp(-lambda_param * squared_dist_int)

lambda_term = (4 * lambda_param**2) * squared_dist_int - 4 * lambda_param


exp_lambda = p * lambda_term 


def closure():

    optimizer.zero_grad(set_to_none=True)
    params = list(model.parameters())

    N_boundary = model(boundary_points)
    b = boundary_values - N_boundary
    q = torch.linalg.lu_solve(LU, pivots, b)

    N_interior = model(interior_points)


    correction = p@q
    psi_trial = N_interior + correction
    psi_trial = psi_trial.view(-1)
    exp_psi_trial = torch.exp(psi_trial)
    # First and second derivatives
    psi_trial_grad = torch.autograd.grad(psi_trial, interior_points,
                               grad_outputs=torch.ones_like(psi_trial),
                                         create_graph=True)[0]
    psi_trial_x = psi_trial_grad[:, 0]
    psi_trial_y = psi_trial_grad[:, 1]

    psi_trial_xx = torch.autograd.grad(psi_trial_x, interior_points,
                                grad_outputs=torch.ones_like(psi_trial_x),
                               create_graph=True)[0][:, 0]
    psi_trial_yy = torch.autograd.grad(psi_trial_y, interior_points,
                                grad_outputs=torch.ones_like(psi_trial_y),
                               create_graph=True)[0][:, 1]


    loss = torch.mean((psi_trial_xx + psi_trial_yy + exp_psi_trial - f_values) ** 2)

#***************************************************************************************************
    # Compute del_q_del_p
    jacobian1 = []
    for i in range(N_boundary.shape[0]):
        grad_i = torch.autograd.grad(N_boundary[i],params, 
                torch.ones_like(N_boundary[i]), retain_graph=True)
        grad_i = torch.cat([g.view(-1) for g in grad_i])
        jacobian1.append(grad_i)

    jacobian1 = torch.stack(jacobian1)

#*****************************************************************************************************
    num_boundary_points = boundary_points.shape[0]
    total_params = jacobian1.shape[1]  # Number of parameters
    del_q_del_p = torch.zeros((num_boundary_points, 
                    total_params), dtype=torch.float64) 

    del_q_del_p = torch.linalg.lu_solve(LU, pivots, -jacobian1)

#***********************************************************************************************  
    # Compute del(E)/del(p)

    error_term = (psi_trial_xx + psi_trial_yy + exp_psi_trial - f_values).view(-1,1)



    grad_N_x = torch.autograd.grad(N_interior, interior_points, 
                grad_outputs = torch.ones_like(N_interior), 
                            create_graph=True)[0][:, 0]  # Select dx component
    # Compute second derivative N_xx
    grad_N_xx = torch.autograd.grad(grad_N_x, interior_points, 
                        grad_outputs=torch.ones_like(grad_N_x), 
                            create_graph=True)[0][:, 0]  # Select dx component

    # Compute Jacobian w.r.t. all model parameters
    jacobian_N_xx = []

    for i in range(grad_N_xx.shape[0]):  # Loop over 10 interior points
        grad_i = torch.autograd.grad(grad_N_xx[i], params, 
                                     torch.ones_like(grad_N_xx[i]), retain_graph = True)
        grad_i = torch.cat([g.view(-1) for g in grad_i])  # Flatten gradients
        jacobian_N_xx.append(grad_i)

    jacobian_N_xx = torch.stack(jacobian_N_xx)  

#*****************************************************************************************
    # Compute first derivative N_y
    grad_N_y = torch.autograd.grad(N_interior, interior_points,
                    grad_outputs=torch.ones_like(N_interior), 
                        create_graph=True)[0][:, 1]  # Select dy component

    # Compute second derivative N_yy
    grad_N_yy = torch.autograd.grad(grad_N_y, interior_points, 
                        grad_outputs = torch.ones_like(grad_N_y), 
                        create_graph=True)[0][:, 1]  # Select dy component

    # Compute Jacobian w.r.t. all model parameters
    jacobian_N_yy = []    
    for i in range(grad_N_yy.shape[0]): 
        grad_i = torch.autograd.grad(grad_N_yy[i], params, 
                                retain_graph=True)
        grad_i = torch.cat([g.view(-1) for g in grad_i])  # Flatten gradients
        jacobian_N_yy.append(grad_i)

    jacobian_N_yy = torch.stack(jacobian_N_yy)  

#************************************************************************************************

    jacobian_N = []
    for i in range(N_interior.shape[0]):  
        grad_i = torch.autograd.grad(N_interior[i], params, 
                                     torch.ones_like(N_interior[i]), retain_graph = True)
        grad_i = torch.cat([g.view(-1) for g in grad_i])  # Flatten gradients
        jacobian_N.append(grad_i)

    jacobian_N = torch.stack(jacobian_N)    

#************************************************************************************************

    first_sum_second_term = exp_lambda @ del_q_del_p


    first_sum = (jacobian_N_xx + jacobian_N_yy) + first_sum_second_term


    exp_psi_trial_col = exp_psi_trial.view(-1,1)
    second_sum = exp_psi_trial_col * (jacobian_N + p @ del_q_del_p)

    total_sum = error_term * (first_sum + second_sum)
    total_sum_reduced = total_sum.sum(dim=0, keepdim=True)


    N_int = interior_points.shape[0]

    delE_delp = (2.0 / N_int) * total_sum_reduced

    dE_dp = delE_delp.view(-1)  # Ensure it's a 1D tensor

    dE_dp_split = torch.split(dE_dp, [p.numel() for p in model.parameters()])


    for param, grad in zip(model.parameters(), dE_dp_split):
        param.grad = grad.view(param.shape)  # Reshape to match param shape

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
    if epoch % 2 == 0:
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

exact_all = torch.log(
    1 + all_points[:, 0]**2 + all_points[:, 1]**2
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




