#!/usr/bin/env python3
"""Generate a 2D inverted-U channel mesh for OpenFOAM.

Geometry (arch / inverted-U with a small inlet port on top of the arch):

                  ___ inlet_small (port from above)
                  |  |
                 _|  |_
                /      \
               /        \
              |          |
              |          |
              |  (legs)  |
              |          |
            inlet_main  outlet

  - Big inlet at bottom of left leg ("inlet_main")
  - Small inlet at the top face of a short vertical port centred above
    the arch apex.  Side stream enters going DOWN and is entrained by
    the main flow passing horizontally underneath at the apex — no
    head-on blockage of the main leg.
  - Outlet at bottom of right leg ("outlet")

Single-cell Z extrusion for 2D OpenFOAM (front/back -> empty).

Patches:
  "inlet_main"   - bottom face of left leg (big inlet)
  "inlet_small"  - top face of the port above the arch apex
  "outlet"       - bottom face of right leg
  "walls"        - all other lateral surfaces
  "back"         - z = 0 (empty)
  "front"        - z = Z (empty)

Usage:
    python gen_u_2d.py            # writes u_2d.msh
    gmshToFoam u_2d.msh
    # Set front/back -> empty in constant/polyMesh/boundary

Requires: gmsh Python API
"""

import math
import os
import gmsh

# -- Parameters ---------------------------------------------------------------

W   = 0.025    # Main channel width [m]  (= main pipe diameter in 3D)
Ws  = 0.010    # Secondary inlet (port) width [m]  (= port diameter in 3D)
Rc  = 0.060    # Arch centerline radius [m]  (must be > W/2)
L   = 0.150    # Leg length below the arch [m]
Lb  = 0.020    # Port length past the outer arch wall [m]

# Horizontal gap between the inlet_small port axis and the outlet-leg axis.
# Both are vertical, so this controls how close the secondary inlet sits to
# the outlet leg (same definition as PORT_GAP_FROM_LEG in gen_u_3d.py).
PORT_GAP_FROM_LEG = 0.015   # 15 mm

Z   = 0.001    # Extrusion depth (single cell) for 2D [m]

# Mesh sizing
lc_wall = 0.0006
lc_bulk = 0.0012

# -- Derived geometry ---------------------------------------------------------

hw  = W  / 2.0
hws = Ws / 2.0
Ri  = Rc - hw
Ro  = Rc + hw

xC, yC = Rc, L

# Vertical port placement: stub axis at x = x_stub, walls parallel to the
# legs (the right wall closer to the outlet leg, the left wall further from
# it).  Each vertical wall meets the outer arch at a point whose y is
# determined by Ro and the wall's x.
x_stub       = 2 * Rc - PORT_GAP_FROM_LEG
port_left_x  = x_stub - hws
port_right_x = x_stub + hws

port_arch_left_xy  = (port_left_x,
                     yC + math.sqrt(Ro ** 2 - (port_left_x  - xC) ** 2))
port_arch_right_xy = (port_right_x,
                     yC + math.sqrt(Ro ** 2 - (port_right_x - xC) ** 2))

# Flat port top, Lb above the outer arch at the stub centreline.
y_outer_top_stub = yC + math.sqrt(Ro ** 2 - (x_stub - xC) ** 2)
port_top_y = y_outer_top_stub + Lb
port_top_left_xy  = (port_left_x,  port_top_y)
port_top_right_xy = (port_right_x, port_top_y)

# -- Build geometry -----------------------------------------------------------

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.model.add("u_2d")
geo = gmsh.model.geo


def pt(x, y, lc=lc_wall):
    return geo.addPoint(x, y, 0, lc)


pC           = pt(xC, yC)
pC_inner_top = pt(xC, yC + Ri)

# Inlet_main face
p_im_L = pt(-hw, 0, lc_bulk)
p_im_R = pt( hw, 0, lc_bulk)

p_iL_top = pt( hw, L)
p_iR_top = pt(2 * Rc - hw, L)

# Outlet face
p_o_L = pt(2 * Rc - hw, 0, lc_bulk)
p_o_R = pt(2 * Rc + hw, 0, lc_bulk)

p_oR_top = pt(2 * Rc + hw, L)
p_oL_top = pt(-hw, L)

# Port corners (on the outer arc and at the top of the port)
p_port_arch_right = pt(port_arch_right_xy[0], port_arch_right_xy[1])
p_port_arch_left  = pt(port_arch_left_xy[0],  port_arch_left_xy[1])
p_port_top_right  = pt(port_top_right_xy[0],  port_top_right_xy[1])
p_port_top_left   = pt(port_top_left_xy[0],   port_top_left_xy[1])

# -- Curves -------------------------------------------------------------------

c_inlet_main = geo.addLine(p_im_L, p_im_R)
c_iL_wall    = geo.addLine(p_im_R, p_iL_top)

c_iA_left  = geo.addCircleArc(p_iL_top,     pC, pC_inner_top)
c_iA_right = geo.addCircleArc(pC_inner_top, pC, p_iR_top)

c_iR_wall = geo.addLine(p_iR_top, p_o_L)
c_outlet  = geo.addLine(p_o_L, p_o_R)
c_oR_wall = geo.addLine(p_o_R, p_oR_top)

# Outer arch is split by the port at the top.
c_oA_right = geo.addCircleArc(p_oR_top,         pC, p_port_arch_right)
c_oA_left  = geo.addCircleArc(p_port_arch_left, pC, p_oL_top)

# Port walls + inlet_small face
c_port_right_wall = geo.addLine(p_port_arch_right, p_port_top_right)
c_inlet_small     = geo.addLine(p_port_top_right,  p_port_top_left)
c_port_left_wall  = geo.addLine(p_port_top_left,   p_port_arch_left)

# Outer-left wall — single straight segment (no side branch)
c_oL_wall = geo.addLine(p_oL_top, p_im_L)

loop = geo.addCurveLoop([
    c_inlet_main,
    c_iL_wall,
    c_iA_left, c_iA_right,
    c_iR_wall,
    c_outlet,
    c_oR_wall,
    c_oA_right,
    c_port_right_wall,
    c_inlet_small,
    c_port_left_wall,
    c_oA_left,
    c_oL_wall,
])

fluid = geo.addPlaneSurface([loop])
geo.synchronize()

# -- Extrude (single cell in Z) ----------------------------------------------

ext = gmsh.model.geo.extrude([(2, fluid)], 0, 0, Z, numElements=[1], recombine=True)
geo.synchronize()

top_surf = ext[0][1]
vol_tag  = ext[1][1]
laterals = [e[1] for e in ext[2:] if e[0] == 2]

lat_inlet_main       = laterals[0]
lat_iL_wall          = laterals[1]
lat_iA_left          = laterals[2]
lat_iA_right         = laterals[3]
lat_iR_wall          = laterals[4]
lat_outlet           = laterals[5]
lat_oR_wall          = laterals[6]
lat_oA_right         = laterals[7]
lat_port_right_wall  = laterals[8]
lat_inlet_small      = laterals[9]
lat_port_left_wall   = laterals[10]
lat_oA_left          = laterals[11]
lat_oL_wall          = laterals[12]

print(f"Lateral surfaces: {len(laterals)} (expect 13)")

walls = [
    lat_iL_wall,
    lat_iA_left, lat_iA_right,
    lat_iR_wall,
    lat_oR_wall,
    lat_oA_right,
    lat_port_right_wall,
    lat_port_left_wall,
    lat_oA_left,
    lat_oL_wall,
]

# -- Physical groups ----------------------------------------------------------

gmsh.model.addPhysicalGroup(2, [lat_inlet_main],  name="inlet_main")
gmsh.model.addPhysicalGroup(2, [lat_inlet_small], name="inlet_small")
gmsh.model.addPhysicalGroup(2, [lat_outlet],      name="outlet")
gmsh.model.addPhysicalGroup(2, walls,             name="walls")
gmsh.model.addPhysicalGroup(2, [fluid],           name="back")
gmsh.model.addPhysicalGroup(2, [top_surf],        name="front")
gmsh.model.addPhysicalGroup(3, [vol_tag],         name="internal")

# -- Mesh ---------------------------------------------------------------------

gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.MeshSizeMin", lc_wall * 0.5)
gmsh.option.setNumber("Mesh.MeshSizeMax", lc_bulk * 1.5)
gmsh.option.setNumber("Mesh.ElementOrder", 1)

gmsh.model.mesh.generate(3)

# -- Save ---------------------------------------------------------------------

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "u_2d.msh")
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write(out_path)

nn = len(gmsh.model.mesh.getNodes()[0])
ne = sum(len(e) for e in gmsh.model.mesh.getElements(3)[1])

gmsh.finalize()

print(f"\nWrote {out_path}")
print(f"  Geometry: 2D inverted-U channel with vertical inlet_small port near outlet leg")
print(f"  Main width:         {W*1000:.0f} mm")
print(f"  Port opening width: {Ws*1000:.0f} mm")
print(f"  Arch radius:        {Rc*1000:.0f} mm (centerline)")
print(f"  Leg length:         {L*1000:.0f} mm")
print(f"  Port stub length:   {Lb*1000:.0f} mm")
print(f"  Gap to outlet leg:  {PORT_GAP_FROM_LEG*1000:.1f} mm  (axis-to-axis)")
print(f"  Nodes:  {nn}")
print(f"  Cells:  {ne}")
print(f"  Patches: inlet_main, inlet_small, outlet, walls, front, back")
print(f"\n  gmshToFoam {os.path.basename(out_path)}")
print(f"  # Set front/back -> empty in constant/polyMesh/boundary")


# -- Render preview PNG ------------------------------------------------------

def render_preview() -> None:
    """Slice the single-layer hex grid at z mid-plane and render as PNG."""
    try:
        import numpy as np
        import pyvista as pv
        import meshio
    except ImportError as exc:
        print(f"\nSkipping preview render: {exc}")
        return

    pv.OFF_SCREEN = True
    here = os.path.dirname(os.path.abspath(__file__))

    m = meshio.read(out_path)
    points = m.points
    hex_blocks = [cb.data for cb in m.cells if cb.type == "hexahedron"]
    if not hex_blocks:
        print("Skipping preview render: no hex cells in mesh")
        return
    hexes = np.vstack(hex_blocks)
    n_cells = hexes.shape[0]
    cells = np.hstack([np.full((n_cells, 1), 8, np.int64), hexes]).ravel()
    ctypes = np.full(n_cells, pv.CellType.HEXAHEDRON, np.uint8)
    grid = pv.UnstructuredGrid(cells, ctypes, points)

    z_mid = float(grid.bounds[5] + grid.bounds[4]) / 2.0
    slc = grid.slice(normal="z", origin=(0, 0, z_mid))

    p = pv.Plotter(off_screen=True, window_size=(1600, 1600))
    p.add_mesh(slc, show_edges=True, color="lightsteelblue",
               edge_color="#406090", line_width=0.5)
    p.view_xy()
    p.reset_camera()
    p.add_text(f"2D U-channel mesh ({n_cells:,} hex cells)",
               position="upper_edge", font_size=14, color="black")
    p.background_color = "white"
    out_png = os.path.join(here, "u_2d_preview.png")
    p.screenshot(out_png)
    p.close()
    print(f"Wrote {out_png}")


render_preview()
