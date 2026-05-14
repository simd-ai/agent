#!/usr/bin/env python3
"""Generate a 3D inverted-U PIPE (circular cross-section) for OpenFOAM.

Geometry: a real round pipe bent into an inverted U, plus a small
circular inlet port grafted into the outer wall at the arch apex.

Built with the OpenCASCADE kernel by fusing:
    - left vertical cylinder  (left leg, axis +y)
    - half-torus              (180 deg arch in the XY plane)
    - right vertical cylinder (right leg, axis +y)
    - small vertical cylinder (secondary inlet stub above arch apex)

Patches (assigned by surface centroid after the fuse):
    inlet_main    bottom disc of left leg   (y=0, x=0)
    outlet        bottom disc of right leg  (y=0, x=2*Rc)
    inlet_small   top disc of port stub     (x=Rc, y=L+Rc+r+Lb)
    walls         the rest of the pipe surface

After writing the .msh file the script also renders preview PNGs with
pyvista so the geometry can be visually verified before submission.

Usage:
    python gen_u_3d.py            # writes u_3d.msh + u_3d_preview.png + u_3d_front.png
    gmshToFoam u_3d.msh

Requires: gmsh, meshio, pyvista
"""

import math
import os
import gmsh

# -- Parameters ---------------------------------------------------------------

D    = 0.025    # Main pipe diameter [m]
Ds   = 0.010    # Secondary inlet pipe diameter [m]
Rc   = 0.060    # Arch centerline radius [m]  (must be > D/2)
L    = 0.150    # Leg length [m]
Lb   = 0.020    # Port stub length past the arch outer wall [m]

# Horizontal gap between the inlet_small stub axis and the outlet-leg axis
# [m].  Both pipes are vertical (axes along +y), so this controls how close
# the secondary inlet sits to the outlet leg they run parallel to.
PORT_GAP_FROM_LEG = 0.015           # 15 mm (halved from the previous 30 mm)

# Mesh sizing
lc_wall = 0.0014
lc_bulk = 0.0022

# -- Derived -----------------------------------------------------------------

r  = D  / 2.0
rs = Ds / 2.0

# Port placement: vertical stub PORT_GAP_FROM_LEG away (horizontally) from
# the outlet leg axis (which is at x = 2*Rc).  At z = 0, a vertical line at
# this x intersects the outer torus surface at y_outer_top; the stub starts
# on the bend centreline (inside the tube) so the fuse cuts only at the
# outer wall, and ends Lb above it.
x_stub         = 2 * Rc - PORT_GAP_FROM_LEG
dx_from_center = x_stub - Rc
y_centerline   = L + math.sqrt(Rc ** 2 - dx_from_center ** 2)
y_outer_top    = L + math.sqrt((Rc + r) ** 2 - dx_from_center ** 2)
PORT_THETA     = math.acos(dx_from_center / Rc)   # for reporting only

port_start_x = x_stub
port_start_y = y_centerline
port_end_x   = x_stub
port_end_y   = y_outer_top + Lb
port_dx      = 0.0
port_dy      = port_end_y - port_start_y

# -- Build geometry (OpenCASCADE) --------------------------------------------

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.model.add("u_3d_pipe")
occ = gmsh.model.occ

left_leg  = occ.addCylinder(0,            0,            0, 0,       L,       0, r)
right_leg = occ.addCylinder(2 * Rc,       0,            0, 0,       L,       0, r)
arch      = occ.addTorus(   Rc,           L,            0, Rc, r, angle=math.pi)
port      = occ.addCylinder(port_start_x, port_start_y, 0, port_dx, port_dy, 0, rs)

fused, _ = occ.fuse(
    [(3, left_leg)],
    [(3, right_leg), (3, arch), (3, port)],
)
occ.synchronize()

assert len(fused) == 1, f"Expected one fused volume, got {len(fused)}: {fused}"
vol_tag = fused[0][1]

# -- Identify boundary faces by centroid -------------------------------------

boundary = gmsh.model.getBoundary([(3, vol_tag)], oriented=False, recursive=False)

tol = 1e-3
inlet_main_surfs  = []
outlet_surfs      = []
inlet_small_surfs = []
wall_surfs        = []

for dim, stag in boundary:
    if dim != 2:
        continue
    cx, cy, _ = occ.getCenterOfMass(2, stag)
    if abs(cx - 0.0)          < tol and abs(cy - 0.0)          < tol:
        inlet_main_surfs.append(stag)
    elif abs(cx - 2 * Rc)     < tol and abs(cy - 0.0)          < tol:
        outlet_surfs.append(stag)
    elif abs(cx - port_end_x) < tol and abs(cy - port_end_y)   < tol:
        inlet_small_surfs.append(stag)
    else:
        wall_surfs.append(stag)

assert inlet_main_surfs,  "Did not find inlet_main face"
assert outlet_surfs,      "Did not find outlet face"
assert inlet_small_surfs, "Did not find inlet_small face"
assert wall_surfs,        "Did not find wall surfaces"

print(f"  inlet_main:  {inlet_main_surfs}")
print(f"  outlet:      {outlet_surfs}")
print(f"  inlet_small: {inlet_small_surfs}")
print(f"  walls:       {len(wall_surfs)} surfaces")

gmsh.model.addPhysicalGroup(2, inlet_main_surfs,  name="inlet_main")
gmsh.model.addPhysicalGroup(2, inlet_small_surfs, name="inlet_small")
gmsh.model.addPhysicalGroup(2, outlet_surfs,      name="outlet")
gmsh.model.addPhysicalGroup(2, wall_surfs,        name="walls")
gmsh.model.addPhysicalGroup(3, [vol_tag],         name="internal")

# -- Mesh --------------------------------------------------------------------

gmsh.option.setNumber("Mesh.MeshSizeMin", lc_wall)
gmsh.option.setNumber("Mesh.MeshSizeMax", lc_bulk)
gmsh.option.setNumber("Mesh.Algorithm",   6)   # 2D: Frontal-Delaunay
gmsh.option.setNumber("Mesh.Algorithm3D", 1)   # 3D: Delaunay (tet)
gmsh.option.setNumber("Mesh.ElementOrder", 1)

gmsh.model.mesh.generate(3)

out_msh = os.path.join(os.path.dirname(os.path.abspath(__file__)), "u_3d.msh")
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write(out_msh)

nn = len(gmsh.model.mesh.getNodes()[0])
_, elem_tags, _ = gmsh.model.mesh.getElements(3)
ne = sum(len(t) for t in elem_tags)

gmsh.finalize()

print(f"\nWrote {out_msh}")
print(f"  Geometry: 3D circular-cross-section inverted-U pipe")
print(f"  Main diameter:      {D*1000:.0f} mm")
print(f"  Port diameter:      {Ds*1000:.0f} mm")
print(f"  Arch radius:        {Rc*1000:.0f} mm (centerline)")
print(f"  Leg length:         {L*1000:.0f} mm")
print(f"  Port stub length:   {Lb*1000:.0f} mm  (above outer wall of arch)")
print(f"  Port angle:         {math.degrees(PORT_THETA):.1f} deg (90 = apex; >90 = upstream)")
print(f"  Gap to outlet leg:  {PORT_GAP_FROM_LEG*1000:.1f} mm  (axis-to-axis)")
print(f"  Nodes:  {nn}")
print(f"  Cells:  {ne}")
print(f"  Patches: inlet_main, inlet_small, outlet, walls")
print(f"\n  gmshToFoam {os.path.basename(out_msh)}")


# -- Render preview PNGs -----------------------------------------------------

def render_preview() -> None:
    """Render iso + front views of the pipe geometry as PNGs."""
    try:
        import numpy as np
        import pyvista as pv
        import meshio
    except ImportError as exc:
        print(f"\nSkipping preview render: {exc}")
        return

    pv.OFF_SCREEN = True
    here = os.path.dirname(os.path.abspath(__file__))

    m = meshio.read(out_msh)
    points = m.points

    blocks = []
    for cb in m.cells:
        if cb.type == "tetra":
            n = cb.data.shape[0]
            cells  = np.hstack([np.full((n, 1), 4, np.int64), cb.data]).ravel()
            ctypes = np.full(n, pv.CellType.TETRA, np.uint8)
            blocks.append((cells, ctypes))
        elif cb.type == "hexahedron":
            n = cb.data.shape[0]
            cells  = np.hstack([np.full((n, 1), 8, np.int64), cb.data]).ravel()
            ctypes = np.full(n, pv.CellType.HEXAHEDRON, np.uint8)
            blocks.append((cells, ctypes))

    if not blocks:
        print("Skipping preview render: no 3D cells found in mesh")
        return

    cells  = np.concatenate([b[0] for b in blocks])
    ctypes = np.concatenate([b[1] for b in blocks])
    grid   = pv.UnstructuredGrid(cells, ctypes, points)

    surface = grid.extract_surface()
    edges = surface.extract_feature_edges(
        feature_angle=20,
        boundary_edges=True,
        non_manifold_edges=False,
        feature_edges=True,
        manifold_edges=False,
    )

    # Isometric view
    p = pv.Plotter(off_screen=True, window_size=(1400, 1200))
    p.add_mesh(surface, color="lightsteelblue", opacity=1.0, show_edges=False)
    p.add_mesh(edges, color="black", line_width=1.2)
    p.view_isometric()
    p.reset_camera()
    p.add_text(
        f"3D U-pipe (D={D*1000:.0f}mm, Rc={Rc*1000:.0f}mm, Lb={Lb*1000:.0f}mm)",
        position="upper_edge", font_size=12, color="black",
    )
    p.background_color = "white"
    out_iso = os.path.join(here, "u_3d_preview.png")
    p.screenshot(out_iso)
    p.close()
    print(f"Wrote {out_iso}")

    # Front view (XY plane) — shows the U-shape outline + port placement
    p = pv.Plotter(off_screen=True, window_size=(1400, 1400))
    p.add_mesh(surface, color="lightsteelblue", opacity=0.55, show_edges=False)
    p.add_mesh(edges, color="black", line_width=1.0)
    p.view_xy()
    p.reset_camera()
    p.add_text(
        "3D U-pipe - front view (XY)",
        position="upper_edge", font_size=12, color="black",
    )
    p.background_color = "white"
    out_front = os.path.join(here, "u_3d_front.png")
    p.screenshot(out_front)
    p.close()
    print(f"Wrote {out_front}")


render_preview()
