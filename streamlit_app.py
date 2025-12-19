import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplstereonet
import io
from scipy.spatial import cKDTree

# ----------------------------
# Core functionality (same outputs)
# ----------------------------
@st.cache_data(show_spinner=False)
def calculate_planes(points, separation_limit):
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if n < 3:
        return np.empty((0, 3)), []

    # --- fast neighbor search ---
    tree = cKDTree(pts)

    # slight padding so we don't miss borderline candidates
    r = float(separation_limit) * (1.0 + 1e-12)
    pairs = np.array(list(tree.query_pairs(r=r)), dtype=np.int32)
    
    if pairs.size == 0:
        return np.empty((0, 3)), []
    
    # EXACT same criterion as your original code
    d = np.linalg.norm(pts[pairs[:, 0]] - pts[pairs[:, 1]], axis=1)
    pairs = pairs[d <= float(separation_limit)]

    # --- build sorted adjacency lists ---
    neighbors = [[] for _ in range(n)]
    for i, j in pairs:
        neighbors[i].append(j)
        neighbors[j].append(i)

    for i in range(n):
        neighbors[i].sort()

    planes = []
    colinear = []

    # --- triangle enumeration via sorted intersections ---
    for i in range(n):
        Ni = neighbors[i]
        for j in Ni:
            if j <= i:
                continue
            Nj = neighbors[j]

            # intersect Ni and Nj with k > j
            pi = pj = 0
            while pi < len(Ni) and pj < len(Nj):
                ki = Ni[pi]
                kj = Nj[pj]
                if ki == kj:
                    k = ki
                    if k > j:
                        v1 = pts[j] - pts[i]
                        v2 = pts[k] - pts[i]
                        normal = np.cross(v1, v2)
                        norm = np.linalg.norm(normal)
                        if norm == 0:
                            colinear.append((pts[i], pts[j], pts[k]))
                        else:
                            planes.append(normal / norm)
                    pi += 1
                    pj += 1
                elif ki < kj:
                    pi += 1
                else:
                    pj += 1

    return np.asarray(planes), colinear



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
        planes, collinear = calculate_planes(pts, float(sep_lim))
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

gen_submit = st.button('Generate stereonets')

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
