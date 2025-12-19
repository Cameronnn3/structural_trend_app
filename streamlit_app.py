import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplstereonet
import io
from scipy.spatial import cKDTree
import numba as nb

# ----------------------------
# Core functionality (same outputs)
# ----------------------------

def _build_adjacency_from_pairs(n, pairs):
    """
    pairs: (m,2) int array of undirected edges (i<j)
    Returns CSR-like adjacency: offsets (n+1), neigh (2m)
    """
    deg = np.zeros(n, dtype=np.int64)
    i = pairs[:, 0]; j = pairs[:, 1]
    np.add.at(deg, i, 1)
    np.add.at(deg, j, 1)

    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(deg, out=offsets[1:])

    neigh = np.empty(offsets[-1], dtype=np.int32)
    cur = offsets[:-1].copy()

    # fill
    for a, b in pairs:
        neigh[cur[a]] = b; cur[a] += 1
        neigh[cur[b]] = a; cur[b] += 1

    # IMPORTANT: sort each neighbor list so we can do fast intersections
    for v in range(n):
        s = offsets[v]; e = offsets[v+1]
        neigh[s:e].sort()

    return offsets, neigh


@nb.njit
def _intersect_count(a, a0, a1, b, b0, b1, min_k):
    # count intersection of sorted a[a0:a1] and b[b0:b1] with k > min_k
    ia = a0
    ib = b0
    c = 0
    while ia < a1 and ib < b1:
        va = a[ia]; vb = b[ib]
        if va == vb:
            if va > min_k:
                c += 1
            ia += 1; ib += 1
        elif va < vb:
            ia += 1
        else:
            ib += 1
    return c


@nb.njit(parallel=True)
def _count_planes(pts, offsets, neigh):
    n = pts.shape[0]
    total = 0
    for i in nb.prange(n):
        si = offsets[i]; ei = offsets[i+1]
        for idx_j in range(si, ei):
            j = neigh[idx_j]
            if j <= i:
                continue
            sj = offsets[j]; ej = offsets[j+1]
            # triangles i-j-k where k in N(i)∩N(j) and k>j
            total += _intersect_count(neigh, si, ei, neigh, sj, ej, j)
    return total


@nb.njit(parallel=True)
def _fill_planes(pts, offsets, neigh, out_normals, out_is_colinear):
    n = pts.shape[0]
    write_pos = np.zeros(n, dtype=np.int64)

    # first pass: per-i counts to compute write offsets
    counts = np.zeros(n, dtype=np.int64)
    for i in nb.prange(n):
        si = offsets[i]; ei = offsets[i+1]
        c = 0
        for idx_j in range(si, ei):
            j = neigh[idx_j]
            if j <= i:
                continue
            sj = offsets[j]; ej = offsets[j+1]
            c += _intersect_count(neigh, si, ei, neigh, sj, ej, j)
        counts[i] = c

    # prefix sum counts -> write_pos
    total = 0
    for i in range(n):
        write_pos[i] = total
        total += counts[i]

    # second pass: actually write normals
    for i in nb.prange(n):
        si = offsets[i]; ei = offsets[i+1]
        w = write_pos[i]
        for idx_j in range(si, ei):
            j = neigh[idx_j]
            if j <= i:
                continue
            sj = offsets[j]; ej = offsets[j+1]
            ia = si
            ib = sj
            # intersect N(i) and N(j)
            while ia < ei and ib < ej:
                a = neigh[ia]; b = neigh[ib]
                if a == b:
                    k = a
                    if k > j:
                        v1x = pts[j,0] - pts[i,0]; v1y = pts[j,1] - pts[i,1]; v1z = pts[j,2] - pts[i,2]
                        v2x = pts[k,0] - pts[i,0]; v2y = pts[k,1] - pts[i,1]; v2z = pts[k,2] - pts[i,2]
                        nx = v1y*v2z - v1z*v2y
                        ny = v1z*v2x - v1x*v2z
                        nz = v1x*v2y - v1y*v2x
                        norm = (nx*nx + ny*ny + nz*nz) ** 0.5
                        if norm == 0.0:
                            out_is_colinear[w] = True
                            out_normals[w,0] = 0.0
                            out_normals[w,1] = 0.0
                            out_normals[w,2] = 0.0
                        else:
                            out_is_colinear[w] = False
                            out_normals[w,0] = nx / norm
                            out_normals[w,1] = ny / norm
                            out_normals[w,2] = nz / norm
                        w += 1
                    ia += 1; ib += 1
                elif a < b:
                    ia += 1
                else:
                    ib += 1


@st.cache_data(show_spinner=False)
def calculate_planes_cached(points: np.ndarray, separation_limit: float):
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if n < 3:
        return np.empty((0,3), dtype=np.float32), []

    tree = cKDTree(pts)
    # edges where distance <= separation_limit
    pairs = np.array(list(tree.query_pairs(r=float(separation_limit))), dtype=np.int32)
    if pairs.size == 0:
        return np.empty((0,3), dtype=np.float32), []

    offsets, neigh = _build_adjacency_from_pairs(n, pairs)

    # count triangles first so we can allocate exactly (no change in plane count)
    total = _count_planes(pts, offsets, neigh)

    normals = np.empty((total, 3), dtype=np.float32)
    is_col = np.empty(total, dtype=np.bool_)

    _fill_planes(pts, offsets, neigh, normals, is_col)

    # build the same "colinear list" type as you had (but beware: huge lists can be massive)
    # If this gets too big, you can instead return just a count without changing plane normals.
    colinear = []
    # We only know which triangle indices are colinear here; rebuilding the original triplets
    # would require also writing (i,j,k). If you truly need the actual point triplets, we can
    # add arrays for i,j,k (still fast, but more memory).
    # For now: preserve your "colinear count" behavior by storing placeholders:
    # (If you need actual triplets, say so and I’ll give the exact i/j/k output version.)
    colinear_count = int(is_col.sum())
    if colinear_count:
        colinear = [None] * colinear_count  # keeps len(colinear) identical to original intent

    # IMPORTANT: planes returned include all normals; colinear triangles are not included as normals in your old code.
    # In your old code, colinear triangles were excluded from planes list. So we must filter them out:
    planes = normals[~is_col]

    return planes, colinear


@st.cache_data(show_spinner=False)
def extract_strike_dip_cached(planes: np.ndarray):
    strikes, dips = [], []
    for nx, ny, nz in planes:
        if nz < 0:
            nx, ny, nz = -nx, -ny, -nz
        dip = np.degrees(np.arctan2(np.hypot(nx, ny), nz))
        dipdir = np.degrees(np.arctan2(nx, ny)) % 360.0
        strike = (dipdir - 90.0) % 360.0
        strikes.append(strike)
        dips.append(dip)
    return np.array(strikes), np.array(dips)


@st.cache_data(show_spinner=False)
def plot_stereonets_cached(dens_strikes, dens_dips, plot_strikes, plot_dips, method, sigma):
    """
    Cached figure generation so changing unrelated widgets doesn't
    re-run the heavy mplstereonet density work.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 10),
        subplot_kw={'projection': 'stereonet'}
    )

    ax1.pole(plot_strikes, plot_dips, 'k.', ms=2)
    ax1.grid(True)

    dens = ax2.density_contourf(
        dens_strikes, dens_dips,
        measurement='poles',
        method=method,
        sigma=sigma
    )
    ax2.pole(plot_strikes, plot_dips, 'wo', ms=1, alpha=0.15)
    fig.colorbar(dens, ax=ax2, label='Pole density', pad=0.125)
    ax2.grid(True)

    dgx, dgy, dgz = mplstereonet.density_grid(
        dens_strikes, dens_dips,
        measurement='poles',
        method=method,
        sigma=sigma
    )
    i_max, j_max = np.unravel_index(np.nanargmax(dgz), dgz.shape)
    max_x, max_y = float(dgx[i_max, j_max]), float(dgy[i_max, j_max])
    max_strike, max_dip = mplstereonet.geographic2pole(max_x, max_y)
    max_strike, max_dip = float(max_strike), float(max_dip)

    for ax in (ax1, ax2):
        ax.pole(max_strike, max_dip, 'ro', ms=3)
        ax.plane(max_strike, max_dip, 'r')

    return fig, max_strike, max_dip


# ----------------------------
# Streamlit UI
# ----------------------------

st.title('Structural Trend Application')

uploaded = st.file_uploader('Upload CSV file', type='csv')
if uploaded is None:
    st.stop()

# Read CSV (cached by Streamlit automatically via uploaded file bytes changing)
skip = st.number_input('Header rows to skip', min_value=0, step=1, value=0)

try:
    df = pd.read_csv(uploaded, skiprows=int(skip))
except Exception as e:
    st.error(f"Error reading CSV: {e}")
    st.stop()

if df.shape[1] < 3:
    st.error("Need at least 3 columns of point data.")
    st.stop()

cols = df.columns.tolist()

# ---------- FORM 1: plane calculation ----------
with st.form("plane_calc_form", clear_on_submit=False):
    x_col = st.selectbox('X column', cols, index=0)
    y_col = st.selectbox('Y column', cols, index=1 if len(cols) > 1 else 0)
    z_col = st.selectbox('Z column', cols, index=2 if len(cols) > 2 else 0)
    sep_lim = st.number_input('Separation-distance limit', min_value=0.0, step=0.1, value=1.0)
    calc_submit = st.form_submit_button('Calculate planes')

if calc_submit:
    pts = df[[x_col, y_col, z_col]].to_numpy()
    with st.spinner("Calculating planes..."):
        planes, collinear = calculate_planes_cached(pts, float(sep_lim))
    st.session_state['planes'] = planes
    st.session_state['collinear'] = collinear
    # reset downstream outputs when planes change
    st.session_state.pop('fig', None)
    st.session_state.pop('max_strike', None)
    st.session_state.pop('max_dip', None)

if 'planes' not in st.session_state:
    st.info("Click **Calculate planes** to continue.")
    st.stop()

planes = st.session_state['planes']
colinear_count = len(st.session_state.get('collinear', []))
st.write(f"Valid planes: {len(planes)}, colinear: {colinear_count}")

# Strike/dip for *all planes* (cached)
strikes_all, dips_all = extract_strike_dip_cached(planes)


use_all = st.radio('Use all planes for density?', ['Yes', 'No'])

# make randomness stable across reruns unless user changes it
if 'subset_seed' not in st.session_state:
    st.session_state['subset_seed'] = 0

if use_all == 'Yes':
    method_choice = None
    size = None
    start = None
    end = None
    seed = None
else:
    method_choice = st.selectbox('Subset method', ['Random subset', 'Slice'])
    if method_choice == 'Random subset':
        size = st.number_input(
            'Subset size', min_value=1, max_value=len(strikes_all),
            value=min(500, len(strikes_all))
        )
        seed = st.number_input('Random seed', min_value=0, step=1, value=int(st.session_state['subset_seed']))
    else:
        start = st.number_input('Slice start index', min_value=0, max_value=len(strikes_all)-1, value=0)
        end = st.number_input('Slice end index', min_value=1, max_value=len(strikes_all), value=len(strikes_all))
        seed = None
        size = None

method = st.selectbox('Density method', ['exponential_kamb', 'linear_kamb', 'kamb', 'schmidt'])
sigma = None
if method in ['exponential_kamb', 'linear_kamb', 'kamb']:
    sigma = st.number_input('Sigma (Kamb)', min_value=0.1, step=0.1, value=3.0)

# plotting subset (for speed)
max_plot = st.number_input('Max poles to plot', min_value=1, max_value=len(strikes_all), value=min(len(strikes_all), 2000))
out_name = st.text_input('Output filename (with extension .png/.jpg)', 'stereonet.png')

gen_submit = st.form_submit_button('Generate stereonets')

if gen_submit:
    # choose density subset
    if use_all == 'Yes':
        strikes_sub, dips_sub = strikes_all, dips_all
    else:
        if method_choice == 'Random subset':
            st.session_state['subset_seed'] = int(seed)
            rng = np.random.default_rng(int(seed))
            idx = rng.choice(len(strikes_all), int(size), replace=False)
            strikes_sub, dips_sub = strikes_all[idx], dips_all[idx]
        else:
            strikes_sub, dips_sub = strikes_all[int(start):int(end)], dips_all[int(start):int(end)]

    # choose poles to plot (separate from density inputs, same as your intent)
    if max_plot < len(strikes_sub):
        rng_plot = np.random.default_rng(0)  # stable plotting selection
        idx_plot = rng_plot.choice(len(strikes_sub), int(max_plot), replace=False)
        plot_strikes, plot_dips = strikes_sub[idx_plot], dips_sub[idx_plot]
    else:
        plot_strikes, plot_dips = strikes_sub, dips_sub

    with st.spinner('Generating stereonets...'):
        fig, max_strike, max_dip = plot_stereonets_cached(
            strikes_sub, dips_sub, plot_strikes, plot_dips, method, sigma
        )

    st.session_state['fig'] = fig
    st.session_state['max_strike'] = max_strike
    st.session_state['max_dip'] = max_dip
    st.session_state['out_name'] = out_name

# ---------- display/download ----------
if 'fig' in st.session_state:
    max_strike = st.session_state['max_strike']
    max_dip = st.session_state['max_dip']
    out_name = st.session_state.get('out_name', 'stereonet.png')

    st.success(f"Max-density pole: strike={max_strike:.1f}, dip={max_dip:.1f}")
    st.pyplot(st.session_state['fig'])

    buf = io.BytesIO()
    fmt = 'png' if out_name.lower().endswith('.png') else 'jpg'
    st.session_state['fig'].savefig(buf, format=fmt)
    buf.seek(0)
    st.download_button('Download plot', data=buf, file_name=out_name, mime=f"image/{fmt}")

    # NOTE: saving to server is usually pointless on Streamlit Cloud,
    # but keeping it because your original code did it.
    st.session_state['fig'].savefig(out_name, dpi=300)
    st.info(f"Also saved on server as '{out_name}'")
