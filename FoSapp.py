"""
Infinite-slope stability playground
Exercise Earth, Soil, Surface 1 — Universität Bayreuth

Run with:   streamlit run infinite_slope_app.py

Notation follows the lecture slides:
  theta  slope angle            phi    angle of internal friction
  z_s    vertical depth to failure plane (m)
  m      saturated fraction h / z_s (0-1)
  rho_b  bulk density = m*rho_sat + (1-m)*rho_dry
  sigma_N  = rho_b g z_s cos^2(theta)
  u        = rho_w g m z_s cos^2(theta)
  tau      = rho_b g z_s sin(theta) cos(theta)
  SS       = c' + (sigma_N - u) tan(phi)
  FoS      = SS / tau
"""

import io

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import BoundaryNorm, ListedColormap

G = 9.81

st.set_page_config(page_title="Infinite slope FoS", layout="wide")


# ----------------------------------------------------------------------------
# Physics
# ----------------------------------------------------------------------------
def bulk_density(m, rho_sat, rho_dry):
    return m * rho_sat + (1.0 - m) * rho_dry


def infinite_slope(theta_rad, z_s, phi_deg, c_kpa, m, rho_sat, rho_dry, rho_w=1000.0):
    """Return dict of stresses (Pa) and FoS for the infinite-slope model.
    Works element-wise on numpy arrays."""
    rho_b = bulk_density(m, rho_sat, rho_dry)
    cos2 = np.cos(theta_rad) ** 2
    sigma_n = rho_b * G * z_s * cos2
    u = rho_w * G * m * z_s * cos2
    sigma_eff = sigma_n - u
    tau = rho_b * G * z_s * np.sin(theta_rad) * np.cos(theta_rad)
    ss = c_kpa * 1000.0 + sigma_eff * np.tan(np.radians(phi_deg))
    with np.errstate(divide="ignore", invalid="ignore"):
        fos = np.where(tau > 0, ss / tau, np.inf)
    return dict(rho_b=rho_b, sigma_n=sigma_n, u=u, sigma_eff=sigma_eff, tau=tau, ss=ss, fos=fos)


def critical_angle_cohesionless(phi_deg, m, rho_sat, rho_dry, rho_w=1000.0):
    rho_b = bulk_density(m, rho_sat, rho_dry)
    return np.degrees(np.arctan((1 - m * rho_w / rho_b) * np.tan(np.radians(phi_deg))))


# ----------------------------------------------------------------------------
# Terrain helpers
# ----------------------------------------------------------------------------
@st.cache_data
def synthetic_dem(n=160, cell=10.0, relief=320.0, seed=3):
    """A hilly catchment: a main valley, side ridges and some bumps."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n] / n
    z = relief * (0.9 * y + 0.35 * np.abs(x - 0.5) * 2)          # tilted valley
    z += relief * 0.25 * np.sin(3 * np.pi * x) * (0.4 + y)        # side ridges
    for _ in range(14):                                            # hills
        cx, cy = rng.uniform(0, 1, 2)
        r = rng.uniform(0.05, 0.14)
        h = rng.uniform(-0.12, 0.22) * relief
        z += h * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * r**2))
    return z.astype(float), cell


def read_ascii_grid(file_bytes):
    """Read an ESRI ASCII grid (.asc). Returns (dem, cellsize)."""
    text = file_bytes.decode("utf-8", errors="ignore").splitlines()
    header, i = {}, 0
    while i < len(text):
        parts = text[i].split()
        if len(parts) == 2 and parts[0].replace("_", "").isalpha():
            header[parts[0].lower()] = float(parts[1])
            i += 1
        else:
            break
    data = np.loadtxt(io.StringIO("\n".join(text[i:])))
    nodata = header.get("nodata_value")
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)
    return data, header.get("cellsize", 1.0)


def read_geotiff(file_bytes):
    import rasterio  # optional dependency
    from rasterio.io import MemoryFile

    with MemoryFile(file_bytes) as mem, mem.open() as src:
        dem = src.read(1).astype(float)
        if src.nodata is not None:
            dem = np.where(dem == src.nodata, np.nan, dem)
        return dem, abs(src.transform.a)


def slope_deg(dem, cell):
    gy, gx = np.gradient(dem, cell)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def hillshade(dem, cell, az=315, alt=45):
    gy, gx = np.gradient(dem, cell)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(az), np.radians(alt)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)


@st.cache_data
def d8_specific_catchment(dem, cell):
    """D8 flow accumulation -> specific catchment area a (m^2 per m contour)."""
    z = np.where(np.isnan(dem), np.nanmax(dem) + 1e6, dem)
    ny, nx = z.shape
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    pad = np.pad(z, 1, mode="edge")
    best_drop = np.zeros_like(z)
    recv = -np.ones(z.size, dtype=np.int64)
    idx = np.arange(z.size).reshape(z.shape)
    for dy, dx in offs:
        nb = pad[1 + dy : 1 + dy + ny, 1 + dx : 1 + dx + nx]
        drop = (z - nb) / (cell * np.hypot(dy, dx))
        better = drop > best_drop
        best_drop = np.where(better, drop, best_drop)
        ry = np.clip(np.arange(ny)[:, None] + dy, 0, ny - 1)
        rx = np.clip(np.arange(nx)[None, :] + dx, 0, nx - 1)
        recv = np.where(better.ravel(), idx[ry, rx].ravel(), recv)
    area = np.full(z.size, cell * cell)
    for i in np.argsort(z.ravel())[::-1]:  # highest first
        r = recv[i]
        if r >= 0:
            area[r] += area[i]
    return (area / cell).reshape(z.shape)


# ----------------------------------------------------------------------------
# Sidebar: material properties (shared by both tabs)
# ----------------------------------------------------------------------------
st.sidebar.header("Material")
phi = st.sidebar.slider("Friction angle φ (°)", 5.0, 60.0, 30.0, 0.5)
c_kpa = st.sidebar.slider("Effective cohesion c′ (kPa)", 0.0, 100.0, 0.0, 0.5,
                          help="0 = dry sand case from the slides. Root cohesion is often 1–10 kPa.")
rho_dry = st.sidebar.number_input("Dry bulk density ρ_dry (kg/m³)", 300, 3000, 1600, 10)
rho_sat = st.sidebar.number_input("Saturated bulk density ρ_sat (kg/m³)", 800, 3200, 2000, 10)
rho_w = 1000.0
if rho_sat < rho_dry:
    st.sidebar.warning("ρ_sat should be larger than ρ_dry.")

st.sidebar.header("Failure plane & water")
z_s = st.sidebar.slider("Depth to failure plane z_s (m, vertical)", 0.1, 50.0, 1.5, 0.1)

st.title("Infinite-slope stability playground")
st.caption("τ = ρ_b g z_s sinθ cosθ   ·   SS = c′ + (σ_N − u) tanφ   ·   FoS = SS / τ")

tab1, tab2 = st.tabs(["Single slope", "FoS map"])

# ----------------------------------------------------------------------------
# Tab 1: single slope
# ----------------------------------------------------------------------------
with tab1:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        theta = st.slider("Slope angle θ (°)", 0.5, 89.0, 25.0, 0.5)
        m_single = st.slider("Saturated fraction m = h / z_s", 0.0, 1.0, 0.0, 0.05, key="m1")
        r = infinite_slope(np.radians(theta), z_s, phi, c_kpa, m_single, rho_sat, rho_dry, rho_w)
        fos = float(r["fos"])
        st.metric("Factor of safety", f"{fos:.2f}",
                  "stable" if fos >= 1 else "FAILS", delta_color="normal" if fos >= 1 else "inverse")
        st.markdown(
            f"""
| Quantity | Value |
|---|---|
| ρ_b | {float(r['rho_b']):.0f} kg/m³ |
| σ_N (total normal) | {float(r['sigma_n'])/1000:.2f} kPa |
| u (pore pressure) | {float(r['u'])/1000:.2f} kPa |
| σ′_N (effective) | {float(r['sigma_eff'])/1000:.2f} kPa |
| τ (driving) | {float(r['tau'])/1000:.2f} kPa |
| SS (resisting) | {float(r['ss'])/1000:.2f} kPa |
"""
        )
        if c_kpa == 0:
            tc = critical_angle_cohesionless(phi, m_single, rho_sat, rho_dry, rho_w)
            st.info(f"Cohesionless critical angle at this m: **θ_c = {tc:.1f}°**\n\n"
                    f"tanθ_c = (1 − m ρ_w/ρ_b) tanφ")

    with col_out:
        th = np.linspace(0.5, 89, 600)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for mm, ls in [(0.0, "-"), (0.5, "--"), (1.0, ":")]:
            f = infinite_slope(np.radians(th), z_s, phi, c_kpa, mm, rho_sat, rho_dry, rho_w)["fos"]
            ax.plot(th, f, ls, color="0.45", lw=1.2, label=f"m = {mm:g}")
        f = infinite_slope(np.radians(th), z_s, phi, c_kpa, m_single, rho_sat, rho_dry, rho_w)["fos"]
        ax.plot(th, f, color="#D85A30", lw=2.4, label=f"current (m = {m_single:g})")
        ax.axhline(1, color="k", lw=0.8)
        ax.axvline(theta, color="#D85A30", lw=0.8, ls="--")
        ax.plot(theta, fos, "o", color="#D85A30")
        ax.set_ylim(0, max(4, min(fos * 1.3, 20)) if np.isfinite(fos) else 4)
        ax.set_xlabel("Slope angle θ (°)")
        ax.set_ylabel("FoS")
        ax.set_title("FoS vs slope angle")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)

        mm = np.linspace(0, 1, 101)
        if c_kpa == 0:
            fig2, ax2 = plt.subplots(figsize=(7, 3))
            ax2.plot(mm, critical_angle_cohesionless(phi, mm, rho_sat, rho_dry, rho_w), color="#378ADD", lw=2)
            ax2.set_xlabel("Saturated fraction m")
            ax2.set_ylabel("Critical angle θ_c (°)")
            ax2.set_title("How water lowers the critical angle (c′ = 0)")
            ax2.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig2)

# ----------------------------------------------------------------------------
# Tab 2: FoS map
# ----------------------------------------------------------------------------
with tab2:
    left, right = st.columns([1, 3])
    with left:
        st.subheader("Terrain")
        up = st.file_uploader("Upload a DEM (.asc or .tif) — optional", type=["asc", "txt", "tif", "tiff"])
        dem, cell = synthetic_dem()
        source = "Synthetic catchment (160 × 160 cells, 10 m)"
        if up is not None:
            try:
                raw = up.getvalue()
                if up.name.lower().endswith((".tif", ".tiff")):
                    dem, cell = read_geotiff(raw)
                else:
                    dem, cell = read_ascii_grid(raw)
                source = f"{up.name} ({dem.shape[0]} × {dem.shape[1]} cells, {cell:g} m)"
                if dem.size > 600 * 600:
                    st.warning("Large DEM: wetness mode may be slow. Consider cropping/resampling.")
            except ImportError:
                st.error("GeoTIFF needs rasterio (`pip install rasterio`). Use .asc or install it.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read DEM: {e}")
        st.caption(source)

        st.subheader("Water")
        water_mode = st.radio("Saturation", ["Uniform m", "Topographic wetness (SINMAP-style)"])
        if water_mode == "Uniform m":
            m_map = st.slider("Saturated fraction m", 0.0, 1.0, 0.0, 0.05, key="m2")
        else:
            log_rt = st.slider("log₁₀ (R / T)  — wetter →", -6.0, 0.0, -3.5, 0.1,
                               help="R = recharge (m/day), T = transmissivity (m²/day). "
                                    "m = min(1, (R/T) · a / sinθ), a = specific catchment area.")
            st.caption(f"R/T = {10**log_rt:.1e} m⁻¹")
        fos_max = st.slider("Colour scale max FoS", 2.5, 10.0, 3.0, 0.5)

    slope = slope_deg(dem, cell)
    theta_map = np.radians(slope)
    if water_mode == "Uniform m":
        m_grid = np.full_like(dem, m_map)
    else:
        a = d8_specific_catchment(dem, cell)
        with np.errstate(divide="ignore", invalid="ignore"):
            m_grid = np.clip((10**log_rt) * a / np.maximum(np.sin(theta_map), 1e-3), 0, 1)

    res = infinite_slope(theta_map, z_s, phi, c_kpa, m_grid, rho_sat, rho_dry, rho_w)
    fos_map = np.where(np.isnan(dem), np.nan, res["fos"])
    valid = np.isfinite(fos_map) | np.isinf(fos_map)
    unstable_pct = 100 * np.nansum(fos_map < 1) / max(np.sum(~np.isnan(dem)), 1)

    with right:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Unstable area (FoS < 1)", f"{unstable_pct:.1f} %")
        k2.metric("Marginal (1 ≤ FoS < 1.3)", f"{100*np.nansum((fos_map>=1)&(fos_map<1.3))/np.sum(~np.isnan(dem)):.1f} %")
        k3.metric("Median slope", f"{np.nanmedian(slope):.1f}°")
        k4.metric("Max slope", f"{np.nanmax(slope):.1f}°")

        hs = hillshade(np.nan_to_num(dem, nan=np.nanmean(dem)), cell)
        bounds = [0, 1.0, 1.3, 1.5, 2.0, fos_max]
        cmap = ListedColormap(["#C0392B", "#EF9F27", "#FAC775", "#97C459", "#3B6D11"])
        norm = BoundaryNorm(bounds, cmap.N)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
        extent = [0, dem.shape[1] * cell, 0, dem.shape[0] * cell]

        axes[0].imshow(hs, cmap="gray", extent=extent)
        im0 = axes[0].imshow(slope, cmap="magma_r", alpha=0.55, extent=extent)
        axes[0].set_title("Slope (°)")
        fig.colorbar(im0, ax=axes[0], shrink=0.75)

        axes[1].imshow(hs, cmap="gray", extent=extent)
        im1 = axes[1].imshow(m_grid, cmap="Blues", vmin=0, vmax=1, alpha=0.7, extent=extent)
        axes[1].set_title("Saturated fraction m")
        fig.colorbar(im1, ax=axes[1], shrink=0.75)

        axes[2].imshow(hs, cmap="gray", extent=extent)
        im2 = axes[2].imshow(np.clip(fos_map, 0, fos_max - 1e-6), cmap=cmap, norm=norm,
                             alpha=0.75, extent=extent)
        axes[2].set_title("Factor of safety")
        cb = fig.colorbar(im2, ax=axes[2], shrink=0.75, ticks=bounds)
        cb.ax.set_yticklabels(["0", "1.0", "1.3", "1.5", "2.0", f"≥{fos_max:g}"])
        for ax in axes:
            ax.set_xlabel("m")
        fig.tight_layout()
        st.pyplot(fig)

        fig3, ax3 = plt.subplots(figsize=(7, 3.2))
        f_flat = fos_map[np.isfinite(fos_map)]
        ax3.hist(np.clip(f_flat, 0, fos_max), bins=60, color="#7F77DD")
        ax3.axvline(1, color="k", lw=1)
        ax3.set_xlabel("FoS (clipped)")
        ax3.set_ylabel("cells")
        ax3.set_title("FoS distribution")
        ax3.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig3)

        csv = io.StringIO()
        np.savetxt(csv, np.where(np.isfinite(fos_map), fos_map, -9999), fmt="%.3f", delimiter=",")
        st.download_button("Download FoS grid (CSV)", csv.getvalue(), "fos_grid.csv", "text/csv")

st.divider()
st.caption("Infinite-slope model (shallow, translational failure, slope-parallel seepage). "
           "Wetness mode follows the steady-state approach of SINMAP/SHALSTAB. For teaching, not design.")
