#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.geometry import (
    Point,
    Polygon,
    MultiPolygon,
    LineString
)

import numpy as np


# ==========================================================
# Create data folder
# ==========================================================
DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)


# ==========================================================
# 1️⃣ Load Sweden geometry
# ==========================================================
url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_SWE_0.json"

gdf = gpd.read_file(url)

geom = gdf.geometry.values[0]


# ==========================================================
# Keep mainland only
# ==========================================================
if isinstance(geom, MultiPolygon):

    geom = max(
        geom.geoms,
        key=lambda g: g.area
    )


# ==========================================================
# 2️⃣ Compute bounding box
# ==========================================================
x0, y0 = geom.exterior.xy

xmin, xmax = min(x0), max(x0)

ymin, ymax = min(y0), max(y0)


# ==========================================================
# Aspect ratio for embedding
# ==========================================================
HEIGHT = 1.7


# ==========================================================
# 3️⃣ Rescale geometry to:
#
#        [-1, 1] × [-HEIGHT, HEIGHT]
# ==========================================================
def rescale(point):
    """
    Rescale a point from geographical coordinates
    into the computational domain.
    """

    x = 2.0 * (point[0] - xmin) / (xmax - xmin) - 1.0

    y = (
        2.0 * HEIGHT * (point[1] - ymin)
        / (ymax - ymin)
        - HEIGHT
    )

    return (x, y)


# ==========================================================
# Create computational polygon
# ==========================================================
domain = Polygon([
    rescale(p)
    for p in geom.exterior.coords
])


# ==========================================================
# 4️⃣ Generate boundary collocation points
# ==========================================================
Nb = 600

boundary_line = LineString(domain.exterior.coords)

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
# 5️⃣ Generate boundary layer points
# ==========================================================
def boundary_layer_points(
    domain,
    n_layers=3,
    base_spacing=0.01,
    growth=1.2,
    points_per_layer=200
):
    """
    Generate interior boundary layers using inward buffers.
    """

    pts = []

    d = base_spacing

    inner_polygon = domain

    for _ in range(n_layers):

        # --------------------------------------------------
        # Inward buffer
        # --------------------------------------------------
        candidate = inner_polygon.buffer(-d)

        if candidate.is_empty:
            break

        # --------------------------------------------------
        # Keep largest connected component
        # --------------------------------------------------
        if isinstance(candidate, MultiPolygon):

            candidate = max(
                candidate.geoms,
                key=lambda g: g.area
            )

        # --------------------------------------------------
        # Create line representation
        # --------------------------------------------------
        ring = LineString(candidate.exterior.coords)

        L = ring.length

        s_vals = np.linspace(
            0.0,
            L,
            points_per_layer,
            endpoint=False
        )

        # --------------------------------------------------
        # Sample points
        # --------------------------------------------------
        for s in s_vals:

            p = ring.interpolate(s)

            pts.append([p.x, p.y])

        # --------------------------------------------------
        # Update polygon for next layer
        # --------------------------------------------------
        inner_polygon = candidate

        d *= growth

    return np.array(pts), inner_polygon


# ==========================================================
# 6️⃣ Generate interior fill points
# ==========================================================
def interior_fill(domain, N=25):
    """
    Generate Cartesian interior points.
    """

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
# 7️⃣ Generate interior points
# ==========================================================
boundary_layer_pts, core_domain = boundary_layer_points(domain)

interior_fill_pts = interior_fill(core_domain)

interior_pts = np.vstack([
    boundary_layer_pts,
    interior_fill_pts
])


# ==========================================================
# 🔍 Print point statistics
# ==========================================================
print("\n--- Point statistics ---\n")

print("Boundary points        :", boundary_pts.shape[0])

print("Boundary layer points  :", boundary_layer_pts.shape[0])

print("Core interior points   :", interior_fill_pts.shape[0])

print("Total interior points  :", interior_pts.shape[0])

print(
    "Total points (all)     :",
    boundary_pts.shape[0] + interior_pts.shape[0]
)


# ==========================================================
# 8️⃣ Save points in .npy format
# ==========================================================
np.save(
    "data/sweden_boundary_points.npy",
    boundary_pts
)

np.save(
    "data/sweden_interior_points.npy",
    interior_pts
)


# ==========================================================
# 9️⃣ Visualization
# ==========================================================
fig, ax = plt.subplots(figsize=(6, 6))


# ==========================================================
# Plot domain boundary
# ==========================================================
bx, by = domain.exterior.xy

ax.plot(
    bx,
    by,
    color="black",
    linewidth=0.6,
    alpha=0.7,
    zorder=3
)


# ==========================================================
# Plot interior points
# ==========================================================
ax.scatter(
    interior_pts[:, 0],
    interior_pts[:, 1],
    s=0.5,
    c="blue",
    label="Interior"
)


# ==========================================================
# Plot boundary points
# ==========================================================
ax.scatter(
    boundary_pts[:, 0],
    boundary_pts[:, 1],
    s=0.5,
    c="red",
    label="Boundary"
)


# ==========================================================
# Axis settings
# ==========================================================
ax.set_xlim(-1.0, 1.0)

ax.set_ylim(-HEIGHT, HEIGHT)

ax.set_aspect(
    "equal",
    adjustable="box"
)

ax.set_xlabel(r"$x$")

ax.set_ylabel(r"$y$")

ax.legend(
    loc="upper left",
    frameon=False,
    markerscale=4
)


# ==========================================================
# Save figure
# ==========================================================
plt.tight_layout()

plt.savefig(
    "fig3_domain.pdf",
    bbox_inches="tight"
)

plt.show()

plt.close(fig)


# In[ ]:




