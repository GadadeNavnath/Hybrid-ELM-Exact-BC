#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ==========================================================
# Cardioid Point Generation
# ==========================================================
# Generates:
#   1. Boundary points
#   2. Interior points
#   3. Outward unit normals
#
# All arrays are stored in NumPy format
# inside the data/ directory.
# ==========================================================

import os
import numpy as np

from shapely.geometry import (
    Polygon,
    Point,
    LineString,
    MultiPolygon
)


# ==========================================================
# 1. Cardioid Geometry
# ==========================================================
def cardioid_domain(N=2000):
    """
    Create cardioid polygon:
        r = 1 + cos(theta)
    """

    theta = np.linspace(
        0,
        2 * np.pi,
        N,
        endpoint=False
    )

    r = 1 + np.cos(theta)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    coords = np.vstack([x, y]).T

    return Polygon(coords)


# ==========================================================
# 2. Uniform Boundary Points
# ==========================================================
def boundary_points(domain, Nb=500):

    boundary_line = LineString(
        domain.exterior.coords
    )

    s = np.linspace(
        0,
        boundary_line.length,
        Nb,
        endpoint=False
    )

    pts = np.array([
        [
            boundary_line.interpolate(si).x,
            boundary_line.interpolate(si).y
        ]
        for si in s
    ])

    return pts


# ==========================================================
# 3. Boundary-Layer Interior Points
# ==========================================================
def boundary_layer_points(
    domain,
    n_layers=2,
    base_spacing=0.01,
    growth=1.5,
    points_per_layer=300
):

    pts = []

    d = base_spacing

    inner_polygon = domain

    for _ in range(n_layers):

        candidate = inner_polygon.buffer(-d)

        if candidate.is_empty:
            break

        # ==================================================
        # Keep largest connected component if needed
        # ==================================================
        if isinstance(candidate, MultiPolygon):

            candidate = max(
                candidate.geoms,
                key=lambda g: g.area
            )

        ring = LineString(candidate.exterior.coords)

        L = ring.length

        s_vals = np.linspace(
            0,
            L,
            points_per_layer,
            endpoint=False
        )

        for s in s_vals:

            p = ring.interpolate(s)

            pts.append([p.x, p.y])

        inner_polygon = candidate

        d *= growth

    return np.array(pts), inner_polygon


# ==========================================================
# 4. Interior Fill Points
# ==========================================================
def interior_fill(domain, N=20):

    xmin, ymin, xmax, ymax = domain.bounds

    xs = np.linspace(xmin, xmax, N)
    ys = np.linspace(ymin, ymax, N)

    pts = []

    for x in xs:

        for y in ys:

            if domain.contains(Point(x, y)):

                pts.append([x, y])

    return np.array(pts)


# ==========================================================
# 5. Compute Outward Unit Normals
# ==========================================================
def cardioid_normals(boundary_pts):
    """
    Compute outward unit normals
    on the cardioid boundary.
    """

    x = boundary_pts[:, 0]
    y = boundary_pts[:, 1]

    # ======================================================
    # Recover polar angle
    # ======================================================
    theta = np.arctan2(y, x)

    # cardioid radius
    r = 1 + np.cos(theta)

    # radial derivative
    dr_dtheta = -np.sin(theta)

    # ======================================================
    # Parametric derivatives
    # ======================================================
    dx_dtheta = (
        dr_dtheta * np.cos(theta)
        - r * np.sin(theta)
    )

    dy_dtheta = (
        dr_dtheta * np.sin(theta)
        + r * np.cos(theta)
    )

    # tangent vector
    tx = dx_dtheta
    ty = dy_dtheta

    # ======================================================
    # Outward normal
    # ======================================================
    nx = ty
    ny = -tx

    # normalize
    norm = np.sqrt(nx**2 + ny**2)

    nx /= norm
    ny /= norm

    normals = np.vstack([nx, ny]).T

    return normals


# ==========================================================
# 6. Generate Domain Points
# ==========================================================
domain = cardioid_domain()

boundary_pts = boundary_points(
    domain,
    Nb=600
)

boundary_layer_pts, core_domain = boundary_layer_points(
    domain,
    n_layers=3,
    base_spacing=0.01,
    growth=1.3,
    points_per_layer=300
)

interior_fill_pts = interior_fill(
    core_domain,
    N=25
)

interior_pts = np.vstack([
    boundary_layer_pts,
    interior_fill_pts
])


# ==========================================================
# 7. Remove Duplicate Interior Points
# ==========================================================
interior_pts = np.unique(
    interior_pts,
    axis=0
)


# ==========================================================
# 8. Compute Boundary Normals
# ==========================================================
normals = cardioid_normals(
    boundary_pts
)

print("Normals shape :", normals.shape)


# ==========================================================
# 9. Save Arrays in data/ Directory
# ==========================================================
os.makedirs(
    "data",
    exist_ok=True
)

np.save(
    "data/cardioid_boundary_points.npy",
    boundary_pts
)

np.save(
    "data/cardioid_interior_points.npy",
    interior_pts
)

np.save(
    "data/cardioid_boundary_normals.npy",
    normals
)


# ==========================================================
# Print Summary
# ==========================================================
print("Boundary shape :", boundary_pts.shape)
print("Interior shape :", interior_pts.shape)

print("")
print("Saved files:")
print(" - data/cardioid_boundary_points.npy")
print(" - data/cardioid_interior_points.npy")
print(" - data/cardioid_boundary_normals.npy")


# In[ ]:




