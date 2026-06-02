#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import geopandas as gpd

from shapely.geometry import (
    Point,
    Polygon,
    MultiPolygon,
    LineString
)

# ==========================================================
# 1. Create Data Folder
# ==========================================================
DATA_DIR = "data"

os.makedirs(
    DATA_DIR,
    exist_ok=True
)

# ==========================================================
# 2. Load Sweden Geometry
# ==========================================================
url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_SWE_0.json"

gdf = gpd.read_file(url)

geom = gdf.geometry.values[0]

# ==========================================================
# Keep Mainland Only
# ==========================================================
if isinstance(geom, MultiPolygon):

    geom = max(
        geom.geoms,
        key=lambda g: g.area
    )

# ==========================================================
# 3. Compute Bounding Box
# ==========================================================
x0, y0 = geom.exterior.xy

xmin, xmax = (
    min(x0),
    max(x0)
)

ymin, ymax = (
    min(y0),
    max(y0)
)

# ==========================================================
# Aspect Ratio for Embedding
# ==========================================================
HEIGHT = 1.7

# ==========================================================
# 4. Rescale Geometry
#
#        [-1,1] × [-HEIGHT,HEIGHT]
# ==========================================================
def rescale(point):

    x = (
        2.0 * (point[0] - xmin)
        / (xmax - xmin)
        - 1.0
    )

    y = (
        2.0 * HEIGHT * (point[1] - ymin)
        / (ymax - ymin)
        - HEIGHT
    )

    return (x, y)

# ==========================================================
# Construct Computational Polygon
# ==========================================================
domain = Polygon([
    rescale(p)
    for p in geom.exterior.coords
])

# ==========================================================
# 5. Boundary Collocation Points
# ==========================================================
Nb = 600

boundary_line = LineString(
    domain.exterior.coords
)

s_vals = np.linspace(
    0.0,
    boundary_line.length,
    Nb,
    endpoint=False
)

boundary_pts = np.array([
    [
        boundary_line.interpolate(s).x,
        boundary_line.interpolate(s).y
    ]
    for s in s_vals
])

# ==========================================================
# 6. Boundary Layer Points
# ==========================================================
def boundary_layer_points(
    domain,
    n_layers=3,
    base_spacing=0.01,
    growth=1.2,
    points_per_layer=200
):

    pts = []

    d = base_spacing

    inner_polygon = domain

    for _ in range(n_layers):

        # --------------------------------------------------
        # Inward Buffer
        # --------------------------------------------------
        candidate = inner_polygon.buffer(-d)

        if candidate.is_empty:
            break

        # --------------------------------------------------
        # Keep Largest Connected Component
        # --------------------------------------------------
        if isinstance(candidate, MultiPolygon):

            candidate = max(
                candidate.geoms,
                key=lambda g: g.area
            )

        # --------------------------------------------------
        # Create Line Representation
        # --------------------------------------------------
        ring = LineString(
            candidate.exterior.coords
        )

        L = ring.length

        s_vals = np.linspace(
            0.0,
            L,
            points_per_layer,
            endpoint=False
        )

        # --------------------------------------------------
        # Sample Points
        # --------------------------------------------------
        for s in s_vals:

            p = ring.interpolate(s)

            pts.append([p.x, p.y])

        # --------------------------------------------------
        # Update Polygon for Next Layer
        # --------------------------------------------------
        inner_polygon = candidate

        d *= growth

    return np.array(pts), inner_polygon

# ==========================================================
# 7. Interior Fill Points
# ==========================================================
def interior_fill(domain, N=25):

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
# 8. Generate Interior Points
# ==========================================================
boundary_layer_pts, core_domain = boundary_layer_points(
    domain
)

interior_fill_pts = interior_fill(
    core_domain
)

interior_pts = np.vstack([
    boundary_layer_pts,
    interior_fill_pts
])

# ==========================================================
# Point Statistics
# ==========================================================
print("")

print(
    "Boundary points :",
    boundary_pts.shape[0]
)

print(
    "Boundary layer points :",
    boundary_layer_pts.shape[0]
)

print(
    "Core interior points :",
    interior_fill_pts.shape[0]
)

print(
    "Total interior points :",
    interior_pts.shape[0]
)

print(
    "Total points :",
    boundary_pts.shape[0]
    + interior_pts.shape[0]
)

# ==========================================================
# 9. Compute Boundary Normals
# ==========================================================
def compute_boundary_normals(
    boundary_pts,
    domain
):

    N = boundary_pts.shape[0]

    normals = np.zeros_like(boundary_pts)

    # ------------------------------------------------------
    # Interior Reference Point
    # ------------------------------------------------------
    center = np.array(
        domain.centroid.coords[0]
    )

    for i in range(N):

        p_prev = boundary_pts[
            (i - 1) % N
        ]

        p_next = boundary_pts[
            (i + 1) % N
        ]

        # --------------------------------------------------
        # Tangent Vector
        # --------------------------------------------------
        t = p_next - p_prev

        # --------------------------------------------------
        # Candidate Normal
        # --------------------------------------------------
        n = np.array([
            t[1],
            -t[0]
        ])

        n = n / np.linalg.norm(n)

        p = boundary_pts[i]

        # --------------------------------------------------
        # Ensure Outward Direction
        # --------------------------------------------------
        if np.dot(n, p - center) < 0:

            n = -n

        normals[i] = n

    return normals


boundary_normals = compute_boundary_normals(
    boundary_pts,
    domain
)

print(
    "Boundary normals :",
    boundary_normals.shape[0]
)

# ==========================================================
# 10. Prepare Boundary Data
# ==========================================================
boundary_points = boundary_pts.astype(
    np.float64
)

normals_np = boundary_normals.astype(
    np.float64
)

N = boundary_points.shape[0]


# ==========================================================
# 11. Asymmetric Kernel Parameters
# ==========================================================
lambda_param = 600.0

alpha = 0.9999

hx, hy = 1e-4, 1e-4


# ==========================================================
# 12. Pairwise Differences
# ==========================================================
r_i = boundary_points[:, None, :]

r_j = boundary_points[None, :, :]

diffs = r_i - alpha * r_j

diffs[..., 0] += hx

diffs[..., 1] += hy

# ==========================================================
# 13. Dot Products
# ==========================================================
n_i = normals_np[:, None, :]

dot_matrix = np.sum(
    n_i * diffs,
    axis=2
)

# ==========================================================
# 14. Gaussian Kernel Part
# ==========================================================
squared_distances = np.sum(
    diffs**2,
    axis=2
)

A = (
    -2
    * lambda_param
    * np.exp(
        -lambda_param
        * squared_distances
    )
)

# ==========================================================
# 15. Final K Matrix
# ==========================================================
K = A * dot_matrix

# ==========================================================
# Diagnostics
# ==========================================================
cond = np.linalg.cond(K)

print(
    f"\ncond(K) = {cond:.2e}"
)

det = np.linalg.det(K)

print(
    f"\ndet(K) = {det:.2e}"
)

# ==========================================================
# 16. Save Data
# ==========================================================
np.save(
    "data/sweden_boundary_points.npy",
    boundary_pts
)

np.save(
    "data/sweden_interior_points.npy",
    interior_pts
)

np.save(
    "data/sweden_normals.npy",
    boundary_normals
)

np.save(
    "data/K.npy",
    K
)


# In[ ]:




