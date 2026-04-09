"""
test_viz.py — High-quality 3D terrain mesh visualization test script.

Tests multiple approaches for visualizing a CadMapper OBJ terrain mesh
with elevation coloring, good lighting, and interactive controls.

Usage:
    python test_viz.py [method]

Methods:
    1  - trimesh with elevation vertex colors (pyglet window)
    2  - PyVista standalone window with elevation coloring
    3  - PyVista with enhanced lighting and edge highlighting
    4  - Plotly interactive HTML (opens in browser)
    all - run all methods sequentially (default)

Each method opens an interactive 3D window you can rotate/zoom/pan.
Close the window to proceed to the next method.
"""

import sys
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────
OBJ_PATH = r"D:\Documents\SFU\Code\Python_Code\flood_simulator\files\Cooum\cadmapper-Cooum\cadmapper-chennai-tamil-nadu-in.obj"

# Colormap choices (matplotlib names). Try: terrain, viridis, inferno, plasma, gist_earth
COLORMAP = "terrain"


def load_with_trimesh():
    """Load the OBJ mesh using trimesh and return it."""
    import trimesh
    print(f"Loading mesh from: {OBJ_PATH}")
    mesh = trimesh.load_mesh(OBJ_PATH)
    v = mesh.vertices
    print(f"  Vertices: {len(v):,}  |  Faces: {len(mesh.faces):,}")
    print(f"  X range: {v[:,0].min():.1f} – {v[:,0].max():.1f}")
    print(f"  Y range: {v[:,1].min():.1f} – {v[:,1].max():.1f}")
    print(f"  Z range: {v[:,2].min():.1f} – {v[:,2].max():.1f}")
    return mesh


# ── Method 1: Trimesh + elevation vertex colors ───────────────────────
def method_trimesh_colored(mesh=None):
    """
    Apply per-vertex elevation coloring to the trimesh, then show
    with trimesh's built-in pyglet viewer. This gives you the same
    interactive viewer as mesh.show() but with color.
    """
    import trimesh
    import matplotlib.pyplot as plt

    if mesh is None:
        mesh = load_with_trimesh()

    print("\n[Method 1] Trimesh with elevation vertex colors")
    print("  Applying colormap to vertices by Z height...")

    z = mesh.vertices[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min())  # normalize 0–1

    # Get RGBA colors from matplotlib colormap
    cmap = plt.get_cmap(COLORMAP)
    colors = (cmap(z_norm) * 255).astype(np.uint8)  # shape (N, 4)

    # Assign per-vertex colors
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        vertex_colors=colors,
    )

    print("  Opening interactive viewer (close window to continue)...")
    print("  Controls: left-drag=rotate, right-drag=pan, scroll=zoom")

    # Create a scene with better lighting
    scene = trimesh.Scene(mesh)

    # Add lights — a directional light from above and one from the side
    from trimesh.scene.lighting import DirectionalLight
    scene.lights = [
        DirectionalLight(color=[255, 255, 255, 255], intensity=3.0,
                         direction=[0, 0, -1]),
        DirectionalLight(color=[200, 200, 255, 255], intensity=2.0,
                         direction=[1, 1, -0.5]),
        DirectionalLight(color=[255, 220, 200, 255], intensity=1.5,
                         direction=[-1, -1, -0.5]),
    ]

    scene.show()


# ── Method 2: PyVista basic elevation coloring ────────────────────────
def method_pyvista_basic(mesh=None):
    """
    Load OBJ with PyVista, color by Z elevation, standalone window.
    """
    import pyvista as pv

    print("\n[Method 2] PyVista — elevation coloring, standalone window")

    # Load directly with PyVista (reads OBJ natively)
    print(f"  Loading mesh with PyVista...")
    pv_mesh = pv.read(OBJ_PATH)
    print(f"  Points: {pv_mesh.n_points:,}  |  Cells: {pv_mesh.n_cells:,}")

    # Extract Z-coordinates as elevation scalar
    elevation = pv_mesh.points[:, 2]
    pv_mesh["Elevation"] = elevation

    # Create plotter — notebook=False forces standalone window
    p = pv.Plotter(notebook=False, title="Terrain — PyVista Elevation")
    p.add_mesh(
        pv_mesh,
        scalars="Elevation",
        cmap=COLORMAP,
        show_scalar_bar=True,
        scalar_bar_args={"title": "Elevation (m)", "n_labels": 5},
        smooth_shading=True,
        split_sharp_edges=True,
    )

    # Better camera angle — look from above at an angle
    p.camera_position = "xy"
    p.camera.elevation = 30
    p.camera.azimuth = -45

    p.add_axes()
    p.enable_anti_aliasing("ssaa")

    print("  Opening PyVista window...")
    print("  Controls: left-drag=rotate, middle-drag=pan, scroll=zoom")
    print("  Press 'q' to close.")
    p.show()


# ── Method 3: PyVista enhanced (lighting + edges) ─────────────────────
def method_pyvista_enhanced(mesh=None):
    """
    PyVista with multiple lights, edge highlighting for buildings,
    and a clipped elevation range to bring out terrain detail.
    """
    import pyvista as pv

    print("\n[Method 3] PyVista — enhanced lighting + building edges")

    pv_mesh = pv.read(OBJ_PATH)
    elevation = pv_mesh.points[:, 2]
    pv_mesh["Elevation"] = elevation

    # Compute elevation statistics for smart clipping
    z_med = np.median(elevation)
    z_std = np.std(elevation)
    z_p5 = np.percentile(elevation, 5)
    z_p95 = np.percentile(elevation, 95)
    print(f"  Elevation stats: median={z_med:.0f}, std={z_std:.0f}")
    print(f"  P5={z_p5:.0f}, P95={z_p95:.0f}")

    p = pv.Plotter(notebook=False, title="Terrain — PyVista Enhanced")

    # Main mesh with elevation coloring
    p.add_mesh(
        pv_mesh,
        scalars="Elevation",
        cmap=COLORMAP,
        clim=[z_p5, z_p95],          # clip to 5th–95th percentile
        show_scalar_bar=True,
        scalar_bar_args={
            "title": "Elevation (m)",
            "n_labels": 6,
            "fmt": "%.0f",
        },
        smooth_shading=True,
        split_sharp_edges=True,
        edge_color="gray",
        opacity=1.0,
    )

    # Remove default lighting and add custom
    p.remove_all_lights()
    # Key light — from upper right
    p.add_light(pv.Light(
        position=(1, 1, 2), focal_point=(0, 0, 0),
        color="white", intensity=0.8
    ))
    # Fill light — softer, from upper left
    p.add_light(pv.Light(
        position=(-1, 1, 1), focal_point=(0, 0, 0),
        color=[0.85, 0.85, 1.0], intensity=0.4
    ))
    # Rim light — from behind/below
    p.add_light(pv.Light(
        position=(0, -1, 0.5), focal_point=(0, 0, 0),
        color=[1.0, 0.95, 0.8], intensity=0.3
    ))

    p.camera_position = "xy"
    p.camera.elevation = 40
    p.camera.azimuth = -30
    p.add_axes()
    p.enable_anti_aliasing("ssaa")
    p.set_background("white", top="lightblue")

    print("  Opening PyVista enhanced window...")
    print("  Controls: left-drag=rotate, middle-drag=pan, scroll=zoom")
    print("  Tip: use 'v' to toggle between perspective and parallel projection")
    print("  Press 'q' to close.")
    p.show()


# ── Method 4: Plotly interactive HTML ─────────────────────────────────
def method_plotly(mesh=None):
    """
    Convert trimesh to Plotly Mesh3d trace and open in browser.
    Fully interactive, works without any special backend.
    Note: 250k faces can be slow in the browser — we decimate if needed.
    """
    import trimesh
    import plotly.graph_objects as go

    if mesh is None:
        mesh = load_with_trimesh()

    print("\n[Method 4] Plotly — interactive HTML in browser")

    # Optionally decimate for performance (Plotly handles ~100k faces well)
    target_faces = 100_000
    if len(mesh.faces) > target_faces:
        print(f"  Decimating {len(mesh.faces):,} -> ~{target_faces:,} faces for Plotly...")
        mesh_plot = mesh.simplify_quadric_decimation(target_faces)
        print(f"  After decimation: {len(mesh_plot.faces):,} faces")
    else:
        mesh_plot = mesh

    verts = mesh_plot.vertices
    faces = mesh_plot.faces
    z = verts[:, 2]

    print("  Building Plotly figure...")
    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=z,
            colorscale="Earth",  # Plotly colorscale name
            colorbar=dict(title="Elevation (m)", thickness=20),
            lighting=dict(
                ambient=0.3,
                diffuse=0.7,
                specular=0.2,
                roughness=0.5,
                fresnel=0.1,
            ),
            lightposition=dict(x=1000, y=1000, z=5000),
            flatshading=False,
        )
    ])

    fig.update_layout(
        title="Chennai Terrain — Plotly Interactive",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Elevation",
            aspectmode="data",  # preserve real proportions
            camera=dict(
                eye=dict(x=0.8, y=-1.2, z=1.0),
            ),
        ),
        width=1200,
        height=800,
    )

    print("  Opening in browser...")
    print("  Controls: left-drag=rotate, right-drag=pan, scroll=zoom")
    fig.show()


# ── Main ──────────────────────────────────────────────────────────────
METHODS = {
    "1": ("Trimesh elevation colors", method_trimesh_colored),
    "2": ("PyVista basic elevation", method_pyvista_basic),
    "3": ("PyVista enhanced lighting", method_pyvista_enhanced),
    "4": ("Plotly HTML", method_plotly),
}


def main():
    choice = sys.argv[1] if len(sys.argv) > 1 else "all"

    if choice == "all":
        mesh = load_with_trimesh()
        for key in sorted(METHODS):
            label, fn = METHODS[key]
            print(f"\n{'='*60}")
            print(f"  Running: {label}")
            print(f"{'='*60}")
            try:
                fn(mesh)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
    elif choice in METHODS:
        label, fn = METHODS[choice]
        print(f"Running: {label}")
        fn()
    else:
        print(f"Unknown method: {choice}")
        print("Usage: python test_viz.py [1|2|3|4|all]")
        sys.exit(1)


if __name__ == "__main__":
    main()
