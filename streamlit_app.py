"""# Updated streamlit_app.py
# - Adds a progress indicator for the plane calculation (real-time progress bar)
# - Adds an optional voxel-grid downsampling (fast pre-filter) before plane extraction
# - Uses a simple file-based cache (per input hash) so repeated UI interactions are instant
# - Keeps high-resolution PNG download (300 dpi)
#
# Note: For best performance, install scipy (pip install scipy) to get cKDTree.
#"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplstereonet
import io
import os
import hashlib
import threading
import time

# Try fast KD-tree implementation; fall back to naive if not available
try:
    from scipy.spatial import cKDTree as KDTree
    _have_kdtree = True
except Exception:
    KDTree = None
    _have_kdtree = False

CACHE_DIR = ".plane_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def hash_inputs(points_bytes: bytes, shape: tuple, separation_limit: float, downsample_size: float):
    m = hashlib.sha256()
    m.update(points_bytes)
    m.update(str(shape).encode())
    m.update(str(separation_limit).encode())
    m.update(str(downsample_size).encode())
    return m.hexdigest()


def cache_path_for_hash(h):
    return os.path.join(CACHE_DIR, f"{h}.npz")


def voxel_downsample(pts: np.ndarray, voxel_size: float):
    """
    Simple voxel-grid downsampling: group points into cubes of side `voxel_size`
    and keep one point per occupied voxel (the first encountered).
    Returns the downsampled points and the number kept.
    """
    if voxel_size <= 0:
        return pts
    coords = np.floor(pts / float(voxel_size)).astype(np.int64)
    # Use a dict to pick the first point in each voxel
    seen = {}
    indices = []
    for i, key in enumerate(map(tuple, coords)):
        if key not in seen:
            seen[key] = i
            indices.append(i)
    return pts[indices]


def _compute_planes_to_cache(pts: np.ndarray, separation_limit: float, cache_file: str, progress_dict: dict):
    """
    Compute planes and save compressed npz to cache_file.
    progress_dict is a shared dict with keys 'total','current','done' (used only for progress UI).
    """
    pts = np.asarray(pts, dtype=np.float32)
    n = len(pts)
    if n < 3:
        np.savez_compressed(cache_file, planes=np.empty((0, 3), dtype=np.float32), colinear=np.empty((0, 3), dtype=np.int32))
        progress_dict['total'] = n
        progress_dict['current'] = n
        progress_dict['done'] = True
        return

    if _have_kdtree:
        tree = KDTree(pts)
    else:
        tree = None

    planes_list = []
    colinear = []
    progress_dict['total'] = n
    progress_dict['current'] = 0
    progress_dict['done'] = False

    eps = 1e-8

    for i in range(n):
        # neighbors within separation_limit (including i)
        if tree is not None:
            neigh = tree.query_ball_point(pts[i], separation_limit)
        else:
            dists = np.linalg.norm(pts - pts[i], axis=1)
            neigh = np.where(dists <= separation_limit)[0].tolist()

        # keep only neighbors with index > i to avoid duplicate combinations
        neigh = [j for j in neigh if j > i]
        m = len(neigh)
        if m >= 2:
            neigh_arr = np.array(neigh, dtype=np.int32)
            idx_j, idx_k = np.triu_indices(m, k=1)
            js = neigh_arr[idx_j]
            ks = neigh_arr[idx_k]
            v1 = pts[js] - pts[i]
            v2 = pts[ks] - pts[i]
            normals = np.cross(v1, v2)
            norms = np.linalg.norm(normals, axis=1)
            nonzero_mask = norms > eps

            if np.any(~nonzero_mask):
                for bad_j, bad_k in zip(js[~nonzero_mask], ks[~nonzero_mask]):
                    colinear.append((int(i), int(bad_j), int(bad_k)))

            if np.any(nonzero_mask):
                normals_nonzero = normals[nonzero_mask] / norms[nonzero_mask][:, None]
                planes_list.append(normals_nonzero.astype(np.float32))

        progress_dict['current'] += 1

    if len(planes_list) > 0:
        planes = np.vstack(planes_list)
    else:
        planes = np.empty((0, 3), dtype=np.float32)

    # Save to compressed file
    np.savez_compressed(cache_file, planes=planes, colinear=np.array(colinear, dtype=np.int32))
    progress_dict['done'] = True


def load_cached_planes(cache_file: str):
    data = np.load(cache_file)
    planes = data['planes']
    colinear = data['colinear']
    return planes, colinear.tolist() if colinear.size else []


def extract_strike_dip(planes):
    if planes.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    nx = planes[:, 0].astype(np.float64)
    ny = planes[:, 1].astype(np.float64)
    nz = planes[:, 2].astype(np.float64)

    flip_mask = nz < 0
    nx[flip_mask] *= -1
    ny[flip_mask] *= -1
    nz[flip_mask] *= -1

    dip = np.degrees(np.arctan2(np.hypot(nx, ny), nz))
    dipdir = np.degrees(np.arctan2(nx, ny)) % 360.0
    strike = (dipdir - 90.0) % 360.0
    return strike.astype(np.float32), dip.astype(np.float32)


def plot_stereonets(dens_strikes, dens_dips, plot_strikes, plot_dips, method, sigma):
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

    # high-resolution PNG bytes for download & display
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    png_bytes = buf.getvalue()

    return fig, max_strike, max_dip, png_bytes


# --- Streamlit UI ---
st.title('Structural Trend Application')

uploaded = st.file_uploader('Upload CSV file', type='csv')
if uploaded is not None:
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
    x_col = st.selectbox('X column', cols, index=0)
    y_col = st.selectbox('Y column', cols, index=1 if len(cols) > 1 else 0)
    z_col = st.selectbox('Z column', cols, index=2 if len(cols) > 2 else 0)
    sep_lim = st.number_input('Separation-distance limit', min_value=0.0, step=0.1, value=1.0)

    # Optional voxel downsampling to speed up plane extraction
    do_downsample = st.checkbox('Downsample points (voxel grid) before plane extraction', value=False)
    downsample_size = 0.0
    if do_downsample:
        downsample_size = st.number_input('Voxel size for downsampling', min_value=0.001, step=0.001, value=0.5)

    pts = df[[x_col, y_col, z_col]].values.astype(np.float32)
    original_count = len(pts)

    # Convert points to bytes for hashing
    pts_bytes = pts.tobytes()
    h = hash_inputs(pts_bytes, pts.shape, float(sep_lim), float(downsample_size))
    cache_file = cache_path_for_hash(h)

    st.markdown(f"Points: {original_count}")

    # If cached file exists, show small info and don't re-run heavy computation
    if os.path.exists(cache_file):
        cached = True
        planes, colinear = load_cached_planes(cache_file)
        st.info(f"Using cached computation for this input (planes: {len(planes)}, colinear: {len(colinear)})")
        st.session_state['planes_hash'] = h
    else:
        cached = False

    if st.button('Calculate planes'):
        if do_downsample and downsample_size > 0:
            pts_ds = voxel_downsample(pts, downsample_size)
            st.write(f"Downsampled: {len(pts_ds)} points (from {original_count})")
        else:
            pts_ds = pts

        # If cache already exists for these exact inputs, load and skip compute
        if os.path.exists(cache_file):
            planes, colinear = load_cached_planes(cache_file)
            st.success(f"Planes loaded from cache: {len(planes)}, colinear: {len(colinear)}")
            st.session_state['planes_hash'] = h
        else:
            # Run compute in a background thread and display progress
            progress = {'total': 0, 'current': 0, 'done': False}
            thread = threading.Thread(target=_compute_planes_to_cache, args=(pts_ds, float(sep_lim), cache_file, progress))
            thread.start()

            progress_bar = st.progress(0.0)
            status_text = st.empty()
            last_update = 0.0
            # Poll progress until done
            while not progress.get('done', False):
                total = progress.get('total', len(pts_ds) if pts_ds is not None else 1)
                current = progress.get('current', 0)
                # Avoid division by zero
                frac = float(current) / float(total) if total > 0 else 0.0
                # Only update UI at ~10Hz to avoid excessive reruns
                now = time.time()
                if now - last_update > 0.08:
                    progress_bar.progress(min(max(frac, 0.0), 1.0))
                    status_text.text(f"Processing point {current}/{total} ({frac*100:.1f}%)")
                    last_update = now
                time.sleep(0.05)
            # Ensure thread finished
            thread.join()
            progress_bar.progress(1.0)
            status_text.text("Done computing planes.")
            # load from saved cache
            planes, colinear = load_cached_planes(cache_file)
            st.success(f"Planes computed: {len(planes)}, colinear: {len(colinear)}")
            st.session_state['planes_hash'] = h

    # If planes were computed (or loaded), continue UI
    if 'planes_hash' in st.session_state and st.session_state.get('planes_hash') == h:
        planes, colinear = load_cached_planes(cache_file)
        st.write(f"Valid planes: {len(planes)}, colinear: {len(colinear)}")

        strikes_all, dips_all = extract_strike_dip(planes)

        use_all = st.radio('Use all planes for density?', ['Yes', 'No'], index=0)
        if use_all == 'Yes':
            strikes_sub, dips_sub = strikes_all, dips_all
        else:
            method_choice = st.selectbox('Subset method', ['Random subset', 'Slice'])
            if method_choice == 'Random subset':
                size = st.number_input('Subset size', min_value=1, max_value=max(1, len(strikes_all)), value=min(500, len(strikes_all)))
                idx = np.random.choice(len(strikes_all), int(size), replace=False)
                strikes_sub, dips_sub = strikes_all[idx], dips_all[idx]
            else:
                start = st.number_input('Slice start index', min_value=0, max_value=max(0, len(strikes_all) - 1), value=0)
                end = st.number_input('Slice end index', min_value=1, max_value=max(1, len(strikes_all)), value=len(strikes_all))
                strikes_sub, dips_sub = strikes_all[int(start):int(end)], dips_all[int(start):int(end)]

        method = st.selectbox('Density method', ['exponential_kamb', 'linear_kamb', 'kamb', 'schmidt'])
        sigma = None
        if method in ['exponential_kamb', 'linear_kamb', 'kamb']:
            sigma = st.number_input('Sigma (Kamb)', min_value=0.1, step=0.1, value=3.0)

        max_plot = st.number_input('Max poles to plot', min_value=1, max_value=max(1, len(strikes_sub)), value=len(strikes_sub))
        if max_plot < len(strikes_sub):
            idx_plot = np.random.choice(len(strikes_sub), int(max_plot), replace=False)
            plot_strikes, plot_dips = strikes_sub[idx_plot], dips_sub[idx_plot]
        else:
            plot_strikes, plot_dips = strikes_sub, dips_sub

        out_name = st.text_input('Output filename (with extension .png/.jpg)', 'stereonet.png')

        if st.button('Generate stereonets'):
            with st.spinner('Generating stereonets...'):
                fig, max_strike, max_dip, png_bytes = plot_stereonets(strikes_sub, dips_sub, plot_strikes, plot_dips, method, sigma)
                st.session_state['fig_png_bytes'] = png_bytes
                st.session_state['max_strike'] = max_strike
                st.session_state['max_dip'] = max_dip
                st.session_state['last_out_name'] = out_name
                # save file server-side if possible
                try:
                    fmt = 'png' if out_name.lower().endswith('.png') else 'jpg'
                    with open(out_name, 'wb') as f:
                        f.write(png_bytes)
                    st.info(f"Also saved on server as '{out_name}'")
                except Exception:
                    pass

        if 'fig_png_bytes' in st.session_state:
            max_strike = st.session_state.get('max_strike', 0.0)
            max_dip = st.session_state.get('max_dip', 0.0)
            st.success(f"Max-density pole: strike={max_strike:.1f}, dip={max_dip:.1f}")

            st.image(st.session_state['fig_png_bytes'], use_column_width=True)

            out_name = st.session_state.get('last_out_name', 'stereonet.png')
            mime = 'image/png' if out_name.lower().endswith('.png') else 'image/jpeg'
            st.download_button('Download plot', data=st.session_state['fig_png_bytes'], file_name=out_name, mime=mime)

    else:
        st.info("No planes computed yet for this dataset & separation limit. Click 'Calculate planes' to start.")
