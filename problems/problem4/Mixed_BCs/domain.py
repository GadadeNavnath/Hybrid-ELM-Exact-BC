#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
import os

os.makedirs("data", exist_ok=True)

# ==========================================================
# Shell Parameters
# ==========================================================
R_INNER = 0.5
R_OUTER = 1.0

# ==========================================================
# Remove Near-Duplicate Points
# ==========================================================
def remove_near_duplicates(points, tol=1e-12):
    """
    Remove points that are closer than 'tol'.
    """

    tree = cKDTree(points)

    keep = np.ones(len(points), dtype=bool)

    for i, p in enumerate(points):

        if not keep[i]:
            continue

        idx = tree.query_ball_point(p, tol)

        if i in idx:
            idx.remove(i)

        keep[idx] = False

    return points[keep]


# ==========================================================
# Spherical Surface Points (Positive Octant)
# ==========================================================
def spherical_layer_points_positive_octant(
    radius,
    n_theta,
    n_phi,
    pole_tol=1e-10
):
    """
    Generate spherical points only in the positive octant.
    """

    pts = []

    # ------------------------------------------------------
    # Cell-centered angular sampling
    # Prevents singularity at poles
    # ------------------------------------------------------
    theta = (
        np.arange(n_theta) + 0.5
    ) * (0.5 * np.pi / n_theta)

    phi = (
        np.arange(n_phi) + 0.5
    ) * (0.5 * np.pi / n_phi)

    for ph in phi:

        sin_ph = np.sin(ph)

        # --------------------------------------------------
        # Collapse near-pole region to one point
        # --------------------------------------------------
        if sin_ph < pole_tol:

            x = 0.0
            y = 0.0
            z = radius * np.cos(ph)

            pts.append([x, y, z])

            continue

        for th in theta:

            x = radius * sin_ph * np.cos(th)
            y = radius * sin_ph * np.sin(th)
            z = radius * np.cos(ph)

            pts.append([x, y, z])

    return pts


# ==========================================================
# Outer Radial Boundary Layers
# ==========================================================
def radial_layers_outer(
    r_inner,
    r_outer,
    n_layers,
    base_spacing,
    growth,
    n_theta,
    n_phi
):
    """
    Generate layers progressing inward from outer sphere.
    """

    pts = []

    spacing = base_spacing
    radius = r_outer

    for _ in range(n_layers):

        radius -= spacing

        if radius <= r_inner:
            break

        pts += spherical_layer_points_positive_octant(
            radius,
            n_theta,
            n_phi
        )

        spacing *= growth

    return pts, radius


# ==========================================================
# Inner Radial Boundary Layers
# ==========================================================
def radial_layers_inner(
    r_inner,
    r_outer,
    n_layers,
    base_spacing,
    growth,
    n_theta,
    n_phi
):
    """
    Generate layers progressing outward from inner sphere.
    """

    pts = []

    spacing = base_spacing
    radius = r_inner

    for _ in range(n_layers):

        radius += spacing

        if radius >= r_outer:
            break

        pts += spherical_layer_points_positive_octant(
            radius,
            n_theta,
            n_phi
        )

        spacing *= growth

    return pts, radius


# ==========================================================
# Planar Boundary Layers
# ==========================================================
def planar_boundary_layers(
    axis,
    n_layers,
    base_spacing,
    growth,
    n_plane
):
    """
    Generate planar layers near x=0, y=0, or z=0 planes.
    """

    pts = []

    spacing = base_spacing

    grid = np.linspace(0.0, R_OUTER, n_plane)

    for _ in range(n_layers):

        if axis == "x":

            for y in grid:
                for z in grid:

                    radius = np.sqrt(spacing**2 + y**2 + z**2)

                    if R_INNER <= radius <= R_OUTER:
                        pts.append([spacing, y, z])

        elif axis == "y":

            for x in grid:
                for z in grid:

                    radius = np.sqrt(x**2 + spacing**2 + z**2)

                    if R_INNER <= radius <= R_OUTER:
                        pts.append([x, spacing, z])

        elif axis == "z":

            for x in grid:
                for y in grid:

                    radius = np.sqrt(x**2 + y**2 + spacing**2)

                    if R_INNER <= radius <= R_OUTER:
                        pts.append([x, y, spacing])

        spacing *= growth

    return pts


# ==========================================================
# Interior Core Fill
# ==========================================================
def core_fill(
    r_inner_core,
    r_outer_core,
    n_r,
    n_theta,
    n_phi
):
    """
    Fill interior shell region with spherical layers.
    """

    pts = []

    radii = np.linspace(
        r_inner_core,
        r_outer_core,
        n_r
    )

    for radius in radii:

        pts += spherical_layer_points_positive_octant(
            radius,
            n_theta,
            n_phi
        )

    return pts


# ==========================================================
# Generate Interior Points
# ==========================================================
outer_pts, r_outer_core = radial_layers_outer(
    R_INNER,
    R_OUTER,
    n_layers=2,
    base_spacing=0.01,
    growth=1.3,
    n_theta=16,
    n_phi=16
)

inner_pts, r_inner_core = radial_layers_inner(
    R_INNER,
    R_OUTER,
    n_layers=2,
    base_spacing=0.01,
    growth=1.3,
    n_theta=8,
    n_phi=8
)

plane_x_pts = planar_boundary_layers(
    axis="x",
    n_layers=2,
    base_spacing=0.01,
    growth=1.3,
    n_plane=16
)

plane_y_pts = planar_boundary_layers(
    axis="y",
    n_layers=2,
    base_spacing=0.01,
    growth=1.3,
    n_plane=16
)

plane_z_pts = planar_boundary_layers(
    axis="z",
    n_layers=2,
    base_spacing=0.01,
    growth=1.3,
    n_plane=16
)

core_pts = core_fill(
    r_inner_core=r_inner_core,
    r_outer_core=r_outer_core,
    n_r=4,
    n_theta=11,
    n_phi=11
)

# ----------------------------------------------------------
# Combine all interior points
# ----------------------------------------------------------
interior_pts = np.array(
    outer_pts
    + inner_pts
    + plane_x_pts
    + plane_y_pts
    + plane_z_pts
    + core_pts,
    dtype=np.float64
)

# ==========================================================
# Generate Boundary Points
# ==========================================================
boundary_pts = []

# ----------------------------------------------------------
# Spherical boundaries
# ----------------------------------------------------------
boundary_pts += spherical_layer_points_positive_octant(
    R_INNER,
    8,
    8
)

boundary_pts += spherical_layer_points_positive_octant(
    R_OUTER,
    16,
    16
)

# ----------------------------------------------------------
# Coordinate planes
# ----------------------------------------------------------
grid = np.linspace(0.0, R_OUTER, 16)

# x = 0 plane
for y in grid:
    for z in grid:

        radius = np.sqrt(y**2 + z**2)

        if R_INNER <= radius <= R_OUTER:
            boundary_pts.append([0.0, y, z])

# y = 0 plane
for x in grid:
    for z in grid:

        radius = np.sqrt(x**2 + z**2)

        if R_INNER <= radius <= R_OUTER:
            boundary_pts.append([x, 0.0, z])

# z = 0 plane
for x in grid:
    for y in grid:

        radius = np.sqrt(x**2 + y**2)

        if R_INNER <= radius <= R_OUTER:
            boundary_pts.append([x, y, 0.0])

boundary_pts = np.array(boundary_pts, dtype=np.float64)

# ==========================================================
# Remove Near-Duplicate Points
# ==========================================================
interior_pts = remove_near_duplicates(
    interior_pts,
    tol=1e-12
)

boundary_pts = remove_near_duplicates(
    boundary_pts,
    tol=1e-12
)

# ==========================================================
# Diagnostics
# ==========================================================
print("Interior points :", interior_pts.shape[0])
print("Boundary points :", boundary_pts.shape[0])

min_dist = pdist(boundary_pts).min()

print("Minimum boundary spacing :", min_dist)

# ==========================================================
# Visualization
# ==========================================================
fig = plt.figure(figsize=(8, 8))

ax = fig.add_subplot(111, projection="3d")

# Interior points
ax.scatter(
    interior_pts[:, 0],
    interior_pts[:, 1],
    interior_pts[:, 2],
    s=1,
    c="blue",
    label="Interior"
)

# Boundary points
ax.scatter(
    boundary_pts[:, 0],
    boundary_pts[:, 1],
    boundary_pts[:, 2],
    s=1,
    c="red",
    label="Boundary"
)

ax.set_box_aspect([1, 1, 1])

ax.set_title(
    "3D Shell: Interior Boundary Layers on All 5 Surfaces"
)

ax.legend()

plt.tight_layout()
plt.show()

# ==========================================================
# Compute Outward Unit Normals
# ==========================================================
def compute_boundary_normals(
    boundary_pts,
    R_inner,
    R_outer,
    tol=1e-10
):
    """
    Compute outward unit normals for the shell boundary.

    Boundary consists of:
        - Outer sphere   : outward radial
        - Inner sphere   : inward radial
        - x = 0 plane    : (-1, 0, 0)
        - y = 0 plane    : (0,-1, 0)
        - z = 0 plane    : (0, 0,-1)
    """

    normals = np.zeros_like(boundary_pts)

    x = boundary_pts[:, 0]
    y = boundary_pts[:, 1]
    z = boundary_pts[:, 2]

    r = np.sqrt(x**2 + y**2 + z**2)

    # ------------------------------------------------------
    # Boundary masks
    # ------------------------------------------------------
    outer_mask = np.abs(r - R_outer) < tol
    inner_mask = np.abs(r - R_inner) < tol

    xplane_mask = np.abs(x) < tol
    yplane_mask = np.abs(y) < tol
    zplane_mask = np.abs(z) < tol

    # ------------------------------------------------------
    # Outer spherical boundary
    # ------------------------------------------------------
    normals[outer_mask, 0] = x[outer_mask] / r[outer_mask]
    normals[outer_mask, 1] = y[outer_mask] / r[outer_mask]
    normals[outer_mask, 2] = z[outer_mask] / r[outer_mask]

    # ------------------------------------------------------
    # Inner spherical boundary
    # ------------------------------------------------------
    normals[inner_mask, 0] = -x[inner_mask] / r[inner_mask]
    normals[inner_mask, 1] = -y[inner_mask] / r[inner_mask]
    normals[inner_mask, 2] = -z[inner_mask] / r[inner_mask]

    # ------------------------------------------------------
    # Coordinate planes
    # ------------------------------------------------------
    plane_mask = ~outer_mask & ~inner_mask

    normals[xplane_mask & plane_mask] = [-1.0, 0.0, 0.0]
    normals[yplane_mask & plane_mask] = [ 0.0,-1.0, 0.0]
    normals[zplane_mask & plane_mask] = [ 0.0, 0.0,-1.0]

    return normals


# ==========================================================
# Compute Boundary Normals
# ==========================================================
boundary_normals = compute_boundary_normals(
    boundary_pts,
    R_INNER,
    R_OUTER
)

# ==========================================================
# Normal Magnitude Check
# ==========================================================
normal_norms = np.linalg.norm(
    boundary_normals,
    axis=1
)

print(
    "Normal magnitude range:",
    normal_norms.min(),
    normal_norms.max()
)

# ==========================================================
# Save Points and Normals
# ==========================================================
np.save(
    "data/shell_interior_points.npy",
    interior_pts
)

np.save(
    "data/shell_boundary_points.npy",
    boundary_pts
)

np.save(
    "data/shell_boundary_normals.npy",
    boundary_normals
)

print("\nSaved:")
print("  data/shell_interior_points.npy")
print("  data/shell_boundary_points.npy")
print("  data/shell_boundary_normals.npy")


# In[ ]:




