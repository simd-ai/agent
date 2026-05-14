#!/usr/bin/env python3
"""Render preview screenshots of the 2D and 3D U-shape meshes.

Writes:
  u_2d_preview.png   - top-down view of the 2D mesh (XY plane)
  u_3d_preview.png   - isometric view of the 3D mesh
  u_3d_top.png       - top-down view of the 3D mesh (showing the U outline)

Requires: pyvista, meshio (already in vir/).
"""

import os
import pyvista as pv
import meshio
import numpy as np

pv.OFF_SCREEN = True
HERE = os.path.dirname(os.path.abspath(__file__))


def msh_to_pyvista(path: str) -> pv.UnstructuredGrid:
    """Read a Gmsh .msh and return only the 3D (hex) cells as a PyVista grid."""
    m = meshio.read(path)
    points = m.points
    hex_blocks = [cb.data for cb in m.cells if cb.type == "hexahedron"]
    if not hex_blocks:
        raise RuntimeError(f"No hex cells in {path}")
    hexes = np.vstack(hex_blocks)
    n = hexes.shape[0]
    cells = np.hstack([np.full((n, 1), 8, dtype=np.int64), hexes]).ravel()
    celltypes = np.full(n, pv.CellType.HEXAHEDRON, dtype=np.uint8)
    return pv.UnstructuredGrid(cells, celltypes, points)


def render_2d():
    grid = msh_to_pyvista(os.path.join(HERE, "u_2d.msh"))
    # Slice the single-layer hex grid at its mid-plane to get a flat 2D view.
    z_mid = float(grid.bounds[5] + grid.bounds[4]) / 2.0
    slc = grid.slice(normal="z", origin=(0, 0, z_mid))

    p = pv.Plotter(off_screen=True, window_size=(1400, 1200))
    p.add_mesh(slc, show_edges=True, color="lightsteelblue",
               edge_color="steelblue", line_width=0.5)
    p.view_xy()
    p.camera.zoom(1.3)
    p.add_text("2D U-channel mesh  (29,273 hex)", position="upper_edge",
               font_size=12, color="black")
    p.background_color = "white"
    out = os.path.join(HERE, "u_2d_preview.png")
    p.screenshot(out)
    p.close()
    print(f"Wrote {out}")


def render_3d():
    grid = msh_to_pyvista(os.path.join(HERE, "u_3d.msh"))

    # Surface only (interior hexes are hidden; outer skin is enough)
    surface = grid.extract_surface()

    # Iso view
    p = pv.Plotter(off_screen=True, window_size=(1600, 1200))
    p.add_mesh(surface, color="lightsteelblue", show_edges=False,
               opacity=1.0)
    p.add_mesh(
        surface.extract_feature_edges(feature_angle=30,
                                      boundary_edges=True,
                                      non_manifold_edges=False,
                                      feature_edges=True,
                                      manifold_edges=False),
        color="black", line_width=1.2,
    )
    p.view_isometric()
    p.camera.zoom(1.2)
    p.add_text("3D U-duct mesh  (2,927,300 hex, 100 z-layers)",
               position="upper_edge", font_size=12, color="black")
    p.background_color = "white"
    out_iso = os.path.join(HERE, "u_3d_preview.png")
    p.screenshot(out_iso)
    p.close()
    print(f"Wrote {out_iso}")

    # Top-down (so the U shape is obvious)
    p = pv.Plotter(off_screen=True, window_size=(1400, 1200))
    p.add_mesh(surface, color="lightsteelblue", show_edges=False)
    p.add_mesh(
        surface.extract_feature_edges(feature_angle=30,
                                      boundary_edges=True,
                                      non_manifold_edges=False,
                                      feature_edges=True,
                                      manifold_edges=False),
        color="black", line_width=1.5,
    )
    p.view_xy()
    p.camera.zoom(1.3)
    p.add_text("3D U-duct  -  top view (XY)",
               position="upper_edge", font_size=12, color="black")
    p.background_color = "white"
    out_top = os.path.join(HERE, "u_3d_top.png")
    p.screenshot(out_top)
    p.close()
    print(f"Wrote {out_top}")


if __name__ == "__main__":
    render_2d()
    render_3d()
