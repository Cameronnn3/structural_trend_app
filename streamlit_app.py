import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplstereonet
import io

# --- Core functionality ---
def calculate_planes(points, separation_limit):
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    neighbors = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(pts[i] - pts[j]) <= separation_limit:
                neighbors[i].add(j)
                neighbors[j].add(i)
    planes, colinear = [], []
    for i in range(n):
        for j in neighbors[i]:
            for k in neighbors[j]:
                if k <= j or k not in neighbors[i]:
                    continue
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                normal = np.cross(v1, v2)
                norm = np.linalg.norm(normal)
                if norm == 0:
                    colinear.append((pts[i], pts[j], pts[k]))
                else:
                    planes.append(normal / norm)
    return np.array(planes), colinear


def extract_strike_dip(planes):
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


def plot_stereonets(dens_strikes, dens_dips, plot_strikes, plot_dips, method, sigma):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 10),
        subplot_kw={'projection': 'stereonet'}
    )
    fig.subplots_adjust(hspace=0.4)
    
    ax1.pole(plot_strikes, plot_dips, 'k.', ms=2)
    #ax1.set_title('Individual poles', pad=30)
    ax1.grid(True)

    dens = ax2.density_contourf(
        dens_strikes, dens_dips,
        measurement='poles',
        method=method,
        sigma=sigma
    )

    ax2.pole(plot_strikes, plot_dips, 'wo', ms=1, alpha=0.15)
    fig.colorbar(dens, ax=ax2, label='Pole density', pad=0.12)
    ax2.grid(True)
    #ax2.set_title('Density contour', pad=30)

    dgx, dgy, dgz = mplstereonet.density_grid(
        dens_strikes, dens_dips,
        measurement='poles',
        method=method,
        sigma=sigma
    )
    i_max, j_max = np.unravel_index(np.nanargmax(dgz), dgz.shape)
    max_x, max_y = float(dgx[i_max, j_max]), float(dgy[i_max, j_max])
    max_strike, max_dip = mplstereonet.geographic2pole(max_x, max_y)
    # Ensure scalars
    max_strike, max_dip = float(max_strike), float(max_dip)
    for ax in (ax1, ax2):
        ax.pole(max_strike, max_dip, 'ro', ms=3)
        ax.plane(max_strike, max_dip, 'r')

    return fig, max_strike, max_dip

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
    y_col = st.selectbox('Y column', cols, index=1 if len(cols)>1 else 0)
    z_col = st.selectbox('Z column', cols, index=2 if len(cols)>2 else 0)
    sep_lim = st.number_input('Separation-distance limit', min_value=0.0, step=0.1, value=1.0)

    if st.button('Calculate planes'):
        pts = df[[x_col, y_col, z_col]].values
        planes, collinear = calculate_planes(pts, sep_lim)
        st.session_state['planes'] = planes
        st.session_state['collinear'] = collinear

    if 'planes' in st.session_state:
        planes = st.session_state['planes']
        colinear_count = len(st.session_state['collinear'])
        st.write(f"Valid planes: {len(planes)}, colinear: {colinear_count}")
        strikes_all, dips_all = extract_strike_dip(planes)

        use_all = st.radio('Use all planes for density?', ['Yes', 'No'])
        if use_all == 'Yes':
            strikes_sub, dips_sub = strikes_all, dips_all
        else:
            method_choice = st.selectbox('Subset method', ['Random subset', 'Slice'])
            if method_choice == 'Random subset':
                size = st.number_input('Subset size', min_value=1, max_value=len(strikes_all), value=min(500, len(strikes_all)))
                idx = np.random.choice(len(strikes_all), size, replace=False)
                strikes_sub, dips_sub = strikes_all[idx], dips_all[idx]
            else:
                start = st.number_input('Slice start index', min_value=0, max_value=len(strikes_all)-1, value=0)
                end = st.number_input('Slice end index', min_value=1, max_value=len(strikes_all), value=len(strikes_all))
                strikes_sub, dips_sub = strikes_all[start:end], dips_all[start:end]

        method = st.selectbox('Density method', ['exponential_kamb', 'linear_kamb', 'kamb', 'schmidt'])
        sigma = None
        if method in ['exponential_kamb', 'linear_kamb', 'kamb']:
            sigma = st.number_input('Sigma (Kamb)', min_value=0.1, step=0.1, value=3.0)

        max_plot = st.number_input('Max poles to plot', min_value=1, max_value=len(strikes_sub), value=len(strikes_sub))
        if max_plot < len(strikes_sub):
            idx_plot = np.random.choice(len(strikes_sub), max_plot, replace=False)
            plot_strikes, plot_dips = strikes_sub[idx_plot], dips_sub[idx_plot]
        else:
            plot_strikes, plot_dips = strikes_sub, dips_sub

        out_name = st.text_input('Output filename (with extension .png/.jpg)', 'stereonet.png')

        if st.button('Generate stereonets'):
            with st.spinner('Generating stereonets...'):
                fig, max_strike, max_dip = plot_stereonets(strikes_sub, dips_sub, plot_strikes, plot_dips, method, sigma)
                st.session_state['fig']         = fig
                st.session_state['max_strike']  = max_strike
                st.session_state['max_dip']     = max_dip
        
        if 'fig' in st.session_state:
            st.success(f"Max-density pole: strike={max_strike:.1f}, dip={max_dip:.1f}")
            st.pyplot(st.session_state['fig'])
            buf = io.BytesIO()
            fmt = 'png' if out_name.lower().endswith('.png') else 'jpg'
            st.session_state['fig'].savefig(buf, format=fmt)
            buf.seek(0)
            st.download_button('Download plot', data=buf, file_name=out_name, mime=f"image/{fmt}")
            st.session_state['fig'].savefig(out_name, dpi=300)
            st.info(f"Also saved on server as '{out_name}'")
