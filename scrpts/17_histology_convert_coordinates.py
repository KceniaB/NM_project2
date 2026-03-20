#%% 
"""
KB 11-August-2025
SW and MF feedback 

""" 
import numpy as np
import pandas as pd

from iblatlas.atlas import AllenAtlas

# ba = AllenAtlas()
# ba.xyz2i([value in metres)]
# ba.i2xyz([value in index])

#%%
ba = AllenAtlas()

# %%
"""
CONVERT XYZ um INTO ALLEN CCF COORDINATES
NOTE: 
    XYZ is in ML AP DV 
"""
# Add input table of the coordinates in xyz um 
data = [
    [0, -6540, 4140],
    [0, -6640, 3900],
    [1500, -4360, 3900],
    [0, -6540, 3950],
    [0, -6540, 3585],
    [1500, -4360, 4550],
    [500, -3100, 4280],
    [1180, -3120, 4280],
    [1500, -2920, 4015],
    [-1500, -2920, 4015],
    [500, -3100, 4210],
    [1500, -3120, 3950],
    [-500, -3100, 4200],
    [1350, -3120, 4100],
    [0, -6540, 3585],
    [1750, -1940, 3000],
    [-2000, -3160, 3000],
    [850, -5450, 3900],
    [850, -5450, 3600],
    [-1250, 740, 3900],
    [0, -6640, 4050],
    [0, -6640, 3850],
    [0, -6640, 3800],
    [0, -6600, 3900],
    [-1700, -650, 4285],
    [850, -5450, 3600],
    [850, -5450, 3600],
    [850, -5450, 3600],
    [-1750, -5450, 3550],
    [850, -5450, 3600],
    [850, -5450, 3600],
    [-1750, -5450, 3550],
    [1750, -700, 4150],
    [-1220, -200, 4750],
    [1760, -730, 4150],
    [1760, -730, 4150],
]

# Convert to DataFrame 
df = pd.DataFrame(data, columns=['x_um', 'y_um', 'z_um'])

# Convert micrometers to meters
coords_m = df[['x_um', 'y_um', 'z_um']].values / 1e6

# Flip z to negative
coords_m[:, 2] *= -1

# Apply xyz2ccf function row by row
results = np.array([ba.xyz2ccf(row) for row in coords_m])

# Add results back to DataFrame
df[['x_ccf', 'y_ccf', 'z_ccf']] = results 

df


# %%
"""
CONVERT ALLEN CCF COORDINATES INTO XYZ 
NOTE: 
    XYZ is in ML AP DV 
    THESE ALLEN CCF ARE TAKEN FROM LASAGNA, SO I ALREADY CHANGED THE COLUMNS TO BE ML AP DV, BECAUSE LASAGNA GIVES ML DV AP
""" 
data_ccf = [
    [111, 157, 228],
    [143, 228, 243],
    [129, 159, 228],
    [117, 145, 228],
    [126, 222, 226],
    [191, 188, 204],
    [190, 190, 145],
    [187, 178, 210],
    [190, 192, 168],
    [195, 198, 176],
    [192, 196, 242],
    [124, 137, 228],
    [123, 131, 227],
    [123, 137, 228],
    [88, 155, 190],
    [95, 175, 190],
    [88, 165, 271],
    [99, 171, 168],
    [102, 155, 196],
    [97, 163, 267],
    [279, 196, 164],
    [283, 230, 266],
    [268, 182, 164],
    [269, 167, 156],
]


# Convert to DataFrame
df_ccf = pd.DataFrame(data_ccf, columns=['x_ccf', 'y_ccf', 'z_ccf'])

# Apply ba.ccf2xyz row-by-row
xyz_results = np.array([ba.ccf2xyz(row) for row in df_ccf.values])

# Add results to DataFrame
df_ccf[['x', 'y', 'z']] = xyz_results 

df_ccf

#%%
# Convert meters → micrometers
df_ccf[['x_um', 'y_um', 'z_um']] = df_ccf[['x', 'y', 'z']] * 1e6

# Prepare final table: only micrometer columns
df_um = df_ccf[['x_um', 'y_um', 'z_um']]
# %%
ba.bc.xyz2i([0,0,0]) #test bregma id 











#%% 

""" 
TO VISUALIZE - 3D WITH LINES, 2 examples 
"""

# pip install iblatlas plotly scikit-image numpy
import numpy as np
from iblatlas.atlas import AllenAtlas
from skimage.measure import marching_cubes
import plotly.graph_objects as go
import plotly.io as pio

# ---- POP-OUT WINDOW ----
pio.renderers.default = "browser"

# ---- SETTINGS ----
RES_UM = 25
FLIP_DV = True   # plot with dorsal up

# ---- LOAD ATLAS & MESH ----
ba = AllenAtlas(res_um=RES_UM)   # label shape ~ (AP, ML, DV)
vol = (ba.label > 0).astype(np.uint8)
res_mm = ba.res_um / 1000.0
verts, faces, _, _ = marching_cubes(vol, level=0.5, spacing=(res_mm, res_mm, res_mm), step_size=2)

dv_extent_mm = vol.shape[2] * res_mm
z_vals = dv_extent_mm - verts[:, 2] if FLIP_DV else verts[:, 2]

mesh = go.Mesh3d(
    x=verts[:, 1], y=verts[:, 0], z=z_vals,  # x=ML, y=AP, z=DV
    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
    opacity=0.12, flatshading=True, name="Brain"
)

# ---- HELPERS ----
def _inside_brain(xyz_m):
    i_ml, i_ap, i_dv = ba.bc.xyz2i(np.asarray(xyz_m)[None, :], mode="clip").astype(int)[0]
    if not (0 <= i_ap < vol.shape[0] and 0 <= i_ml < vol.shape[1] and 0 <= i_dv < vol.shape[2]):
        return False
    return vol[i_ap, i_ml, i_dv] > 0

def _xyz_m_to_plot_mm(xyz_m):
    i_ml, i_ap, i_dv = ba.bc.xyz2i(np.asarray(xyz_m)[None, :], mode="clip").astype(float)[0]
    x_mm, y_mm, z_mm = i_ml * res_mm, i_ap * res_mm, i_dv * res_mm
    if FLIP_DV:
        z_mm = dv_extent_mm - z_mm
    return float(x_mm), float(y_mm), float(z_mm)

def points_trace(xyz_list_m, name="Points", size=6):
    xs, ys, zs = zip(*(_xyz_m_to_plot_mm(p) for p in xyz_list_m))
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="markers", marker=dict(size=size), name=name)

def line_trace(p1_m, p2_m, name="Line", width=6):
    x1,y1,z1 = _xyz_m_to_plot_mm(p1_m)
    x2,y2,z2 = _xyz_m_to_plot_mm(p2_m)
    return go.Scatter3d(x=[x1,x2], y=[y1,y2], z=[z1,z2], mode="lines", line=dict(width=width), name=name)

def segment_length_mm(p1_m, p2_m):
    x1,y1,z1 = _xyz_m_to_plot_mm(p1_m)
    x2,y2,z2 = _xyz_m_to_plot_mm(p2_m)
    return float(np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2))

def vertical_to_dorsal_surface(start_m, max_steps=20000, bisect_steps=14):
    """
    Move strictly along +DV (purely dorsal) from start_m until leaving the brain.
    Return the last in-brain point (approximate dorsal surface) in meters.
    """
    start_m = np.asarray(start_m, float)
    step_m = ba.res_um * 1e-6          # one voxel in meters
    dir_m  = np.array([0.0, 0.0, 1.0]) # +DV only (vertical)
    prev = start_m.copy()
    curr = prev.copy()
    # If start is outside, step ventral until inside (rare)
    if not _inside_brain(curr):
        for _ in range(max_steps):
            curr = prev - dir_m * step_m
            if _inside_brain(curr):
                prev = curr
                break
            prev = curr
    # March dorsal until we exit
    for _ in range(max_steps):
        curr = prev + dir_m * step_m
        if not _inside_brain(curr):
            lo, hi = prev.copy(), curr.copy()
            for _ in range(bisect_steps):
                mid = (lo + hi) / 2.0
                if _inside_brain(mid):
                    lo = mid
                else:
                    hi = mid
            return lo
        prev = curr
    return curr

# ---- YOUR POINTS (meters, ML/AP/DV) ----
A = [ 0.000000, -0.006540, -0.003950 ]   # perfect
C = [ 0.000000, -0.006540, -0.003585 ]   # perfect

# Compute dorsal-most points by going straight up (pure DV)
B_top = vertical_to_dorsal_surface(A)
D_top = vertical_to_dorsal_surface(C)

# Report lengths
len_AB = segment_length_mm(A, B_top)
len_CD = segment_length_mm(C, D_top)
print("B (dorsal-most from A):", B_top, " | AB length:", f"{len_AB:.3f} mm")
print("D (dorsal-most from C):", D_top, " | CD length:", f"{len_CD:.3f} mm")

# ---- FIGURE ----
fig = go.Figure([
    mesh,
    points_trace([A, B_top], name="A & B"),
    line_trace(A, B_top, name=f"A–B (vertical, {len_AB:.2f} mm)"),
    points_trace([C, D_top], name="C & D"),
    line_trace(C, D_top, name=f"C–D (vertical, {len_CD:.2f} mm)"),
])

fig.update_layout(
    scene=dict(
        xaxis_title="ML (mm, +right)",
        yaxis_title="AP (mm, +anterior)",
        zaxis_title="DV (mm, dorsal up)" if FLIP_DV else "DV (mm)",
        aspectmode="data"
    ),
    title=f"Allen Mouse Brain ({RES_UM} µm) — vertical (pure DV) to dorsal surface"
)
fig.show()











#%% 
""" 2 dots for DR but they are too posterior and DR represented in a 3D brain """
# %%
# pip install iblatlas plotly scikit-image numpy
import numpy as np
from iblatlas.atlas import AllenAtlas
from skimage.measure import marching_cubes
import plotly.graph_objects as go

# --- SETTINGS ---
RES_UM = 25
FLIP_DV = True
PURPLE = "#803896"
POINT_SIZE = 8

# --- LOAD ATLAS & WHOLE-BRAIN MESH ---
ba = AllenAtlas(res_um=RES_UM)                # label shape ~ (AP, ML, DV)
vol = (ba.label > 0).astype(np.uint8)
res_mm = ba.res_um / 1000.0

verts, faces, _, _ = marching_cubes(
    vol, level=0.5, spacing=(res_mm, res_mm, res_mm), step_size=2
)  # verts are (AP_mm, ML_mm, DV_mm)

dv_extent_mm = vol.shape[2] * res_mm
z_vals = dv_extent_mm - verts[:, 2] if FLIP_DV else verts[:, 2]

brain_mesh = go.Mesh3d(
    x=verts[:, 1], y=verts[:, 0], z=z_vals,     # x=ML, y=AP, z=DV
    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
    opacity=0.06, flatshading=True, name="Brain", hoverinfo="skip"
)

# --- HELPERS ---
def _xyz_m_to_plot_mm(xyz_m):
    """
    xyz_m: (N,3) in meters, order (ML, AP, DV), bregma-referenced (IBL).
    Returns x_mm, y_mm, z_mm for plotly (x=ML, y=AP, z=DV).
    """
    xyz_m = np.asarray(xyz_m, dtype=float)
    ijk = ba.bc.xyz2i(xyz_m, mode='clip').astype(float)  # (ML, AP, DV)
    x_mm, y_mm, z_mm = ijk[:, 0]*res_mm, ijk[:, 1]*res_mm, ijk[:, 2]*res_mm
    if FLIP_DV:
        z_mm = dv_extent_mm - z_mm
    return x_mm, y_mm, z_mm

def points_with_labels(points_named, color=PURPLE, size=POINT_SIZE):
    labels = [p[0] for p in points_named]
    coords = np.array([p[1] for p in points_named], float)
    x, y, z = _xyz_m_to_plot_mm(coords)
    return go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers+text",
        marker=dict(size=size, color=PURPLE),
        text=labels, textposition="top center",
        textfont=dict(color=PURPLE, size=12),
        name="Points"
    )

def mesh_from_mask(mask_bool, color=PURPLE, name="Region", opacity=0.45, step=1):
    """Create a Mesh3d from a boolean mask in (AP,ML,DV)."""
    if mask_bool.dtype != np.uint8:
        mask_bool = mask_bool.astype(np.uint8)
    v, f, _, _ = marching_cubes(mask_bool, level=0.5,
                                spacing=(res_mm, res_mm, res_mm),
                                step_size=step)
    z = dv_extent_mm - v[:, 2] if FLIP_DV else v[:, 2]
    return go.Mesh3d(
        x=v[:, 1], y=v[:, 0], z=z,
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color=color, opacity=opacity, flatshading=True, name=name
    )

# --- Build DR mask (union of DR and its subregions) ---
# Many atlases label DR subparts (DRD, DRV, DRL, etc.). Take all acronyms starting with 'DR'.
acros = ba.regions.acronym
dr_indices = [i for i, a in enumerate(acros) if isinstance(a, str) and a.startswith("DR")]
label_idx = np.abs(ba.label)  # hemisphere sign removed; 0=outside brain
mask_DR = np.isin(label_idx, dr_indices)

DR_mesh = mesh_from_mask(mask_DR, color=PURPLE, name="Dorsal Raphe (DR)", opacity=0.5, step=1)

# ==== EDIT HERE: your points (ML, AP, DV) in METERS ====
points_named = [
    ("A", [0.000000, -0.006540, -0.003950]),
    ("C", [0.000000, -0.006540, -0.003585]),
    # add more here...
]
# =======================================================

fig = go.Figure([brain_mesh, DR_mesh, points_with_labels(points_named)])
fig.update_layout(
    scene=dict(
        xaxis_title="ML (mm, +right)",
        yaxis_title="AP (mm, +anterior)",
        zaxis_title="DV (mm, dorsal up)",
        aspectmode="data"
    ),
    title=f"Allen Mouse Brain ({RES_UM} µm) — DR (purple) + points"
)
fig.show()

# %%
#%% 
""" 2 dots for DR but they are too posterior and DR represented in a 3D brain """
# %%
# pip install iblatlas plotly scikit-image numpy
import numpy as np
from iblatlas.atlas import AllenAtlas
from skimage.measure import marching_cubes
import plotly.graph_objects as go

# --- SETTINGS ---
RES_UM = 25
FLIP_DV = True
PURPLE = "#803896"
POINT_SIZE = 8

# --- LOAD ATLAS & WHOLE-BRAIN MESH ---
ba = AllenAtlas(res_um=RES_UM)                # label shape ~ (AP, ML, DV)
vol = (ba.label > 0).astype(np.uint8)
res_mm = ba.res_um / 1000.0

verts, faces, _, _ = marching_cubes(
    vol, level=0.5, spacing=(res_mm, res_mm, res_mm), step_size=2
)  # verts are (AP_mm, ML_mm, DV_mm)

dv_extent_mm = vol.shape[2] * res_mm
z_vals = dv_extent_mm - verts[:, 2] if FLIP_DV else verts[:, 2]

brain_mesh = go.Mesh3d(
    x=verts[:, 1], y=verts[:, 0], z=z_vals,     # x=ML, y=AP, z=DV
    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
    opacity=0.06, flatshading=True, name="Brain", hoverinfo="skip"
)

def mesh_from_mask(mask_bool, color=PURPLE, name="Region", opacity=0.45, step=1):
    """Create a Mesh3d from a boolean mask in (AP,ML,DV)."""
    if mask_bool.dtype != np.uint8:
        mask_bool = mask_bool.astype(np.uint8)
    v, f, _, _ = marching_cubes(mask_bool, level=0.5,
                                spacing=(res_mm, res_mm, res_mm),
                                step_size=step)
    z = dv_extent_mm - v[:, 2] if FLIP_DV else v[:, 2]
    return go.Mesh3d(
        x=v[:, 1], y=v[:, 0], z=z,
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color=color, opacity=opacity, flatshading=True, name=name
    )

# --- Build DR mask (union of DR and its subregions) ---
# Many atlases label DR subparts (DRD, DRV, DRL, etc.). Take all acronyms starting with 'DR'.
acros = ba.regions.acronym
dr_indices = [i for i, a in enumerate(acros) if isinstance(a, str) and a.startswith("DR")]
label_idx = np.abs(ba.label)  # hemisphere sign removed; 0=outside brain
mask_DR = np.isin(label_idx, dr_indices)

DR_mesh = mesh_from_mask(mask_DR, color=PURPLE, name="Dorsal Raphe (DR)", opacity=0.5, step=1)


fig = go.Figure([brain_mesh, DR_mesh])
fig.update_layout(
    scene=dict(
        xaxis_title="ML (mm, +right)",
        yaxis_title="AP (mm, +anterior)",
        zaxis_title="DV (mm, dorsal up)",
        aspectmode="data"
    ),
    title=f"Allen Mouse Brain ({RES_UM} µm) — DR (purple) + points"
)
fig.show()





#%% 
# %%
from pathlib import Path
import numpy as np
from iblatlas import atlas
from iblatlas.atlas import AllenAtlas
from skimage.measure import marching_cubes
import plotly.graph_objects as go

file_track = '/home/kceniabougrova/Downloads/ManualLine_pts.csv'

brain_atlas = locals().get('brain_atlas') or atlas.AllenAtlas(25)  # reuse if exists
ixiyiz = np.loadtxt(Path(file_track), delimiter=',')[:, [1, 0, 2]]  # keep order as in CSV
ixiyiz[:, 1] = 527 - ixiyiz[:, 1]            # flip AP for 25 µm histology
# DO NOT SORT -> preserve original row order
xyz = brain_atlas.bc.i2xyz(ixiyiz)           # meters, (ML, AP, DV)

RES_UM = 25
FLIP_DV = True
POINT_SIZE = 5  # smaller dots

PURPLE = "#803896"
RED    = "#E74C3C"
BLUE   = "#1F77B4"
GREEN  = "#2CA02C"

ba = brain_atlas if isinstance(brain_atlas, AllenAtlas) else AllenAtlas(res_um=RES_UM)

vol = (ba.label > 0).astype(np.uint8)
res_mm = ba.res_um / 1000.0
verts, faces, _, _ = marching_cubes(vol, level=0.5, spacing=(res_mm, res_mm, res_mm), step_size=2)
dv_extent_mm = vol.shape[2] * res_mm
z_vals = dv_extent_mm - verts[:, 2] if FLIP_DV else verts[:, 2]

brain_mesh = go.Mesh3d(
    x=verts[:, 1], y=verts[:, 0], z=z_vals,  # x=ML, y=AP, z=DV
    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
    opacity=0.08, flatshading=True, name="Brain", hoverinfo="skip"
)

def _xyz_m_to_plot_mm(xyz_m):
    """(N,3) meters (ML,AP,DV) -> plotly mm coords (x=ML, y=AP, z=DV)."""
    ijk = ba.bc.xyz2i(np.asarray(xyz_m), mode='clip').astype(float)   # (ML,AP,DV)
    x_mm, y_mm, z_mm = ijk[:, 0]*res_mm, ijk[:, 1]*res_mm, ijk[:, 2]*res_mm
    if FLIP_DV:
        z_mm = dv_extent_mm - z_mm
    return x_mm, y_mm, z_mm

# ---- Build per-point colors in the SAME order as CSV rows - manual edit ----
n = xyz.shape[0]
group_lengths = [5, 6, 3, 6, 4]
palette = [PURPLE, RED, PURPLE, BLUE, GREEN]
colors = []
for L, col in zip(group_lengths, palette):
    colors.extend([col] * L)
# Clip (or assert) to your actual row count
colors = colors[:n]

# Single trace with per-point colors (preserves order exactly)
x_mm, y_mm, z_mm = _xyz_m_to_plot_mm(xyz)
points_trace = go.Scatter3d(
    x=x_mm, y=y_mm, z=z_mm,
    mode="markers",
    marker=dict(size=POINT_SIZE, color=colors),
    name="Track points"
)

fig = go.Figure([brain_mesh, points_trace])
fig.update_layout(
    scene=dict(
        xaxis_title="ML (mm, +right)",
        yaxis_title="AP (mm, +anterior)",
        zaxis_title="DV (mm, dorsal up)" if FLIP_DV else "DV (mm)",
        aspectmode="data"
    ),
    title=f"Allen Mouse Brain ({RES_UM} µm) — 24 track points (CSV order colors)"
)
fig.show()




#%%
# %%
from pathlib import Path
import numpy as np
from iblatlas import atlas
from iblatlas.atlas import AllenAtlas
from skimage.measure import marching_cubes
import plotly.graph_objects as go

# ---------- LOAD & CONVERT (PRESERVE CSV ORDER) ----------
file_track = '/home/kceniabougrova/Downloads/ManualLine_pts.csv'

brain_atlas = locals().get('brain_atlas') or atlas.AllenAtlas(25)  # reuse if exists
ixiyiz = np.loadtxt(Path(file_track), delimiter=',')[:, [1, 0, 2]]  # keep order as in CSV
ixiyiz[:, 1] = 527 - ixiyiz[:, 1]            # flip AP for 25 µm histology
# DO NOT SORT -> preserve original row order
xyz = brain_atlas.bc.i2xyz(ixiyiz)           # meters, (ML, AP, DV)

# ---------- PLOTTING ----------
RES_UM = 25
FLIP_DV = True
POINT_SIZE = 5  # smaller dots

PURPLE = "#803896"
RED    = "#E74C3C"
BLUE   = "#1F77B4"
GREEN  = "#2CA02C"

ba = brain_atlas if isinstance(brain_atlas, AllenAtlas) else AllenAtlas(res_um=RES_UM)

# Whole-brain mesh
vol = (ba.label > 0).astype(np.uint8)
res_mm = ba.res_um / 1000.0
verts, faces, _, _ = marching_cubes(vol, level=0.5, spacing=(res_mm, res_mm, res_mm), step_size=2)
dv_extent_mm = vol.shape[2] * res_mm
z_vals = dv_extent_mm - verts[:, 2] if FLIP_DV else verts[:, 2]

brain_mesh = go.Mesh3d(
    x=verts[:, 1], y=verts[:, 0], z=z_vals,  # x=ML, y=AP, z=DV
    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
    opacity=0.08, flatshading=True, name="Brain", hoverinfo="skip"
)

def _xyz_m_to_plot_mm(xyz_m):
    """(N,3) meters (ML,AP,DV) -> plotly mm coords (x=ML, y=AP, z=DV)."""
    ijk = ba.bc.xyz2i(np.asarray(xyz_m), mode='clip').astype(float)   # (ML,AP,DV)
    x_mm, y_mm, z_mm = ijk[:, 0]*res_mm, ijk[:, 1]*res_mm, ijk[:, 2]*res_mm
    if FLIP_DV:
        z_mm = dv_extent_mm - z_mm
    return x_mm, y_mm, z_mm

# ---- Build per-point colors in the SAME order as CSV rows ----
n = xyz.shape[0]
group_lengths = [5, 6, 3, 6, 4]
palette = [PURPLE, RED, PURPLE, BLUE, GREEN]
colors = []
for L, col in zip(group_lengths, palette):
    colors.extend([col] * L)
# Clip (or assert) to your actual row count
colors = colors[:n]

# Single trace with per-point colors (preserves order exactly)
x_mm, y_mm, z_mm = _xyz_m_to_plot_mm(xyz)
points_trace = go.Scatter3d(
    x=x_mm, y=y_mm, z=z_mm,
    mode="markers",
    marker=dict(size=POINT_SIZE, color=colors),
    name="Track points"
)

fig = go.Figure([brain_mesh, points_trace])
fig.update_layout(
    scene=dict(
        xaxis_title="ML (mm, +right)",
        yaxis_title="AP (mm, +anterior)",
        zaxis_title="DV (mm, dorsal up)" if FLIP_DV else "DV (mm)",
        aspectmode="data"
    ),
    title=f"Allen Mouse Brain ({RES_UM} µm) — 24 track points (CSV order colors)"
)
fig.show()