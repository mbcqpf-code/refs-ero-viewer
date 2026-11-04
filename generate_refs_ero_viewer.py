import os
import gc
import re
import sys
import json
import shutil
import warnings
import urllib.request
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage import maximum_filter, convolve
from scipy.spatial import cKDTree
import requests
import xarray as xr

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION & DOMAIN GEOMETRY
# ==========================================
OUTPUT_DIR = Path("output")
RUNS_DIR = OUTPUT_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
WORKDIR = Path("./temp_download")
WORKDIR.mkdir(exist_ok=True)

THRESHOLDS = [1.0, 1.5, 2.0]
GRID_RES_KM = 3.0
NEIGHBORHOOD_KM = 40.0
radius_pts = int(NEIGHBORHOOD_KM / GRID_RES_KM)
y_g, x_g = np.ogrid[-radius_pts:radius_pts + 1, -radius_pts:radius_pts + 1]
CIRCULAR_FOOTPRINT = (x_g**2 + y_g**2) <= radius_pts**2
FOOTPRINT_SUM = float(np.sum(CIRCULAR_FOOTPRINT))

# ==========================================
# 2. DYNAMIC CYCLE & PERIOD RESOLVER
# ==========================================
def get_pending_and_available_cycles(max_lookback_hours=36):
    """
    Scans the last 36 hours for cycles available on AWS S3 that have not
    yet completed locally. Returns them sorted chronologically (oldest first).
    """
    now = datetime.now(timezone.utc)
    anchor = now.replace(minute=0, second=0, microsecond=0)
    pending = []

    for h_back in range(0, max_lookback_hours + 1, 6):
        cand = anchor - timedelta(hours=h_back)
        cycle_hour = (cand.hour // 6) * 6
        cand = cand.replace(hour=cycle_hour)
        date_str = cand.strftime("%Y%m%d")
        cycle_id = cand.strftime("%Y%m%d_%H")

        test_url = (
            f"https://noaa-rrfs-ops-pds.s3.amazonaws.com/rrfs.{date_str}/{cycle_hour:02d}/"
            f"rrfs.t{cycle_hour:02d}z.2dfld.3km.f001.conus.grib2.idx"
        )
        try:
            r = requests.head(test_url, timeout=4)
            if r.status_code == 200:
                flag_file = RUNS_DIR / cycle_id / "complete.json"
                if not (flag_file.exists() and flag_file.stat().st_size > 0):
                    pending.append(cand)
        except Exception:
            continue

    pending.sort()
    return pending

def build_synoptic_periods(init_dt):
    """
    Constructs synoptic periods anchored strictly to 12Z-to-12Z valid windows.
    Labels dynamically by the ending valid date/time.
    """
    cycle = init_dt.hour
    periods = []

    def _format_name(end_dt, hours):
        end_str = end_dt.strftime("%a %m/%d")
        if hours < 24:
            return f"Ending 12Z {end_str} ({hours}h)"
        return f"Ending 12Z {end_str}"

    if cycle == 12:
        p_configs = [
            ("p1", 1, 24, 24, init_dt, init_dt + timedelta(hours=24)),
            ("p2", 25, 48, 24, init_dt + timedelta(hours=24), init_dt + timedelta(hours=48)),
            ("p3", 49, 72, 24, init_dt + timedelta(hours=48), init_dt + timedelta(hours=72)),
        ]
    elif cycle == 18:
        p_configs = [
            ("p1", 1, 18, 18, init_dt, init_dt + timedelta(hours=18)),
            ("p2", 19, 42, 24, init_dt + timedelta(hours=18), init_dt + timedelta(hours=42)),
            ("p3", 43, 66, 24, init_dt + timedelta(hours=42), init_dt + timedelta(hours=66)),
        ]
    elif cycle == 0:
        p_configs = [
            ("p1", 1, 12, 12, init_dt, init_dt + timedelta(hours=12)),
            ("p2", 13, 36, 24, init_dt + timedelta(hours=12), init_dt + timedelta(hours=36)),
            ("p3", 37, 60, 24, init_dt + timedelta(hours=36), init_dt + timedelta(hours=60)),
        ]
    elif cycle == 6:
        p_configs = [
            ("p1", 1, 6, 6, init_dt, init_dt + timedelta(hours=6)),
            ("p2", 7, 30, 24, init_dt + timedelta(hours=6), init_dt + timedelta(hours=30)),
            ("p3", 31, 54, 24, init_dt + timedelta(hours=30), init_dt + timedelta(hours=54)),
        ]

    for p_id, f_s, f_e, hrs, v_s, v_e in p_configs:
        periods.append({
            "id": p_id,
            "name": _format_name(v_e, hrs),
            "f_start": f_s,
            "f_end": f_e,
            "hours": hrs,
            "valid_start": v_s,
            "valid_end": v_e
        })

    return periods

# ==========================================
# 3. MEMBER SPECIFICATIONS & S3 FETCHER
# ==========================================
def get_member_catalog(init_dt):
    prev_dt = init_dt - timedelta(hours=6)
    catalog = []

    for i in range(1, 6):
        catalog.append({"id": f"m{i:03d}_t0", "type": f"m{i:03d}", "date": init_dt.strftime("%Y%m%d"),
                        "cycle": init_dt.hour, "max_lead": 60, "offset": 0})
    catalog.append({"id": "rrfs_ctrl_t0", "type": "rrfs_ctrl", "date": init_dt.strftime("%Y%m%d"),
                    "cycle": init_dt.hour, "max_lead": 84, "offset": 0})
    catalog.append({"id": "hrrr_t0", "type": "hrrr", "date": init_dt.strftime("%Y%m%d"),
                    "cycle": init_dt.hour, "max_lead": 48, "offset": 0})

    for i in range(1, 6):
        catalog.append({"id": f"m{i:03d}_lag6", "type": f"m{i:03d}", "date": prev_dt.strftime("%Y%m%d"),
                        "cycle": prev_dt.hour, "max_lead": 54, "offset": 6})
    catalog.append({"id": "rrfs_ctrl_lag6", "type": "rrfs_ctrl", "date": prev_dt.strftime("%Y%m%d"),
                    "cycle": prev_dt.hour, "max_lead": 78, "offset": 6})
    catalog.append({"id": "hrrr_lag6", "type": "hrrr", "date": prev_dt.strftime("%Y%m%d"),
                    "cycle": prev_dt.hour, "max_lead": 42, "offset": 6})

    return catalog

def download_apcp_slice(grib_url, idx_url, dest_path, fxx):
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        return True

    headers = {"User-Agent": "REFS-ERO-Viewer/1.0"}
    try:
        idx_resp = requests.get(idx_url, headers=headers, timeout=10)
        if idx_resp.status_code != 200:
            return False

        lines = idx_resp.text.strip().split("\n")
        start_byte, end_byte = None, None
        target_regex = rf":APCP:surface:{fxx-1}-{fxx} hour acc"

        for i, line in enumerate(lines):
            if re.search(target_regex, line):
                parts = line.split(":")
                start_byte = int(parts[1])
                if i + 1 < len(lines):
                    end_byte = int(lines[i + 1].split(":")[1]) - 1
                break

        if start_byte is None:
            return False

        range_hdr = f"bytes={start_byte}-{end_byte}" if end_byte else f"bytes={start_byte}-"
        grib_resp = requests.get(grib_url, headers={"Range": range_hdr, **headers}, timeout=25)

        if grib_resp.status_code in (200, 206):
            with open(dest_path, "wb") as f:
                f.write(grib_resp.content)
            return True
    except Exception:
        pass
    return False

def get_member_file_info(mem, lead_rel):
    fxx_actual = lead_rel + mem["offset"]
    date_str = mem["date"]
    cycle = mem["cycle"]
    m_type = mem["type"]

    if m_type.startswith("m"):
        mem_num = int(m_type[1:])
        grib_url = (f"https://noaa-rrfs-ops-pds.s3.amazonaws.com/rrfsens.{date_str}/{cycle:02d}/"
                    f"m{mem_num:03d}/rrfs.t{cycle:02d}z.m{mem_num:03d}.2dfldnomads.3km.f{fxx_actual:03d}.conus.grib2")
    elif m_type == "rrfs_ctrl":
        grib_url = (f"https://noaa-rrfs-ops-pds.s3.amazonaws.com/rrfs.{date_str}/{cycle:02d}/"
                    f"rrfs.t{cycle:02d}z.2dfld.3km.f{fxx_actual:03d}.conus.grib2")
    elif m_type == "hrrr":
        grib_url = (f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{date_str}/conus/"
                    f"hrrr.t{cycle:02d}z.wrfsfcf{fxx_actual:02d}.grib2")

    idx_url = f"{grib_url}.idx"
    dest = WORKDIR / f"{mem['id']}_f{lead_rel:03d}.grib2"
    return grib_url, idx_url, dest, fxx_actual

# ==========================================
# 4. FFG RETRIEVAL & CONUS GRID ALIGNMENT
# ==========================================
def get_aligned_ffg(ref_lats, ref_lons, init_dt):
    date_path = init_dt.strftime("%Y/%m/%d")
    date_str = init_dt.strftime("%Y%m%d%H")
    ffg_url = f"https://mesonet.agron.iastate.edu/archive/data/{date_path}/model/ffg/5kmffg_{date_str}.grib2"
    local_ffg = WORKDIR / f"ffg_{date_str}.grib2"

    if not local_ffg.exists():
        urllib.request.urlretrieve(ffg_url, local_ffg)

    ds_1h = xr.open_dataset(local_ffg, engine="cfgrib", backend_kwargs={"filter_by_keys": {"stepRange": "0-1"}})
    ds_3h = xr.open_dataset(local_ffg, engine="cfgrib", backend_kwargs={"filter_by_keys": {"stepRange": "0-3"}})
    ds_6h = xr.open_dataset(local_ffg, engine="cfgrib", backend_kwargs={"filter_by_keys": {"stepRange": "0-6"}})

    v = list(ds_1h.data_vars)[0]
    f1 = (ds_1h[v] / 25.4).values
    f3 = (ds_3h[v] / 25.4).values
    f6 = (ds_6h[v] / 25.4).values

    pts_ffg = np.column_stack((ds_1h.latitude.values.ravel(), ds_1h.longitude.values.ravel()))
    pts_mod = np.column_stack((ref_lats.ravel(), ref_lons.ravel()))

    tree = cKDTree(pts_ffg)
    _, idx = tree.query(pts_mod, workers=-1)

    aligned = {
        "1h": np.where(f1.ravel()[idx].reshape(ref_lats.shape) > 0, f1.ravel()[idx].reshape(ref_lats.shape), np.nan),
        "3h": np.where(f3.ravel()[idx].reshape(ref_lats.shape) > 0, f3.ravel()[idx].reshape(ref_lats.shape), np.nan),
        "6h": np.where(f6.ravel()[idx].reshape(ref_lats.shape) > 0, f6.ravel()[idx].reshape(ref_lats.shape), np.nan),
    }
    return aligned

# ==========================================
# 5. PRE-COMPUTED REFS PPFFG FETCHER
# ==========================================
def extract_precomputed_refs(init_dt, period):
    cycle = init_dt.hour
    date_str = init_dt.strftime("%Y%m%d")
    max_ppffg = None

    def _fetch_fxx(fxx):
        grib_url = f"https://noaa-rrfs-ops-pds.s3.amazonaws.com/refs.{date_str}/{cycle:02d}/ensprod/refs.t{cycle:02d}z.ffri.f{fxx:02d}.conus.grib2"
        idx_url = f"{grib_url}.idx"
        dest = WORKDIR / f"refs_ppffg_f{fxx:02d}.grib2"

        if dest.exists():
            return True
        try:
            r = requests.get(idx_url, timeout=6)
            if r.status_code != 200:
                return False
            lines = r.text.strip().split("\n")
            starts, ends = [], []
            for i, l in enumerate(lines):
                if ":PPFFG:" in l:
                    starts.append(int(l.split(":")[1]))
                    ends.append(int(lines[i+1].split(":")[1]) - 1 if i + 1 < len(lines) else None)
            if not starts:
                return False
            range_hdr = f"bytes={min(starts)}-{max([e for e in ends if e]) if None not in ends else ''}"
            rg = requests.get(grib_url, headers={"Range": range_hdr}, timeout=20)
            if rg.status_code in (200, 206):
                with open(dest, "wb") as f:
                    f.write(rg.content)
                return True
        except Exception:
            return False

    fxx_range = list(range(period["f_start"], period["f_end"] + 1))
    with ThreadPoolExecutor(max_workers=16) as pool:
        pool.map(_fetch_fxx, fxx_range)

    for fxx in fxx_range:
        fpath = WORKDIR / f"refs_ppffg_f{fxx:02d}.grib2"
        if not fpath.exists():
            continue
        for n in [1, 3, 6]:
            s_hr = fxx - n
            if s_hr < period["f_start"] - 1:
                continue
            try:
                ds = xr.open_dataset(fpath, engine="cfgrib", backend_kwargs={"filter_by_keys": {"stepRange": f"{s_hr}-{fxx}"}, "indexpath": ""})
                v = "ppffg" if hasattr(ds, "ppffg") else list(ds.data_vars)[0]
                arr = ds[v].values.astype(np.float32)
                max_ppffg = arr if max_ppffg is None else np.maximum(max_ppffg, arr)
                ds.close()
            except Exception:
                continue
        if fpath.exists():
            fpath.unlink()

    if max_ppffg is not None:
        smoothed = ndimage.gaussian_filter(np.nan_to_num(max_ppffg, nan=0.0), sigma=4.0)
        return np.where(smoothed >= 5.0, smoothed, np.nan)
    return None

# ==========================================
# 6. GRAPHICS RENDERING SYSTEM
# ==========================================
PROJ = ccrs.LambertConformal(central_longitude=-97.5, central_latitude=38.5)

ERO_COLORS = ['#32CD32', '#FFFF00', '#FFA500', '#FF0000', '#A52A2A', '#FF00FF']
CMAP_PROB = ListedColormap(ERO_COLORS)
BOUNDS_PROB = [5, 15, 25, 40, 55, 70, 100]
NORM_PROB = BoundaryNorm(BOUNDS_PROB, CMAP_PROB.N)

COV_COLORS = ['#e0f7fa', '#c8e6c9', '#fff59d', '#ffb74d', '#f44336', '#9c27b0']
CMAP_COV = ListedColormap(COV_COLORS)
BOUNDS_COV = [1, 5, 10, 25, 50, 75, 100]
NORM_COV = BoundaryNorm(BOUNDS_COV, CMAP_COV.N)

FFG_COLORS = [
    '#e04f39', '#f28522', '#fbb034', '#eef641', '#55cc33',
    '#196619', '#7b9ee8', '#00d4ff', '#0011cc', '#ba68c8',
]
CMAP_FFG = ListedColormap(FFG_COLORS)
CMAP_FFG.set_over('#5a187a')
BOUNDS_FFG = [0.01, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
NORM_FFG = BoundaryNorm(BOUNDS_FFG, CMAP_FFG.N)

def render_plot(data, lats, lons, title, subtitle, cbar_label, filename, out_dir, plot_type="prob"):
    fig, ax = plt.subplots(figsize=(13, 8.5), dpi=110, subplot_kw={"projection": PROJ})
    ax.add_feature(cfeature.LAND, facecolor="#f8f9fa")
    ax.add_feature(cfeature.OCEAN, facecolor="#eef2f5")
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor="#333333")
    ax.add_feature(cfeature.BORDERS, linewidth=0.9, edgecolor="#333333")
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="#666666", linestyle=":")
    ax.set_extent([-120, -72, 22, 50], crs=ccrs.PlateCarree())

    if plot_type == "ffg":
        cmap, norm, bounds, extend = CMAP_FFG, NORM_FFG, BOUNDS_FFG, "max"
    elif plot_type == "cov":
        cmap, norm, bounds, extend = CMAP_COV, NORM_COV, BOUNDS_COV, "max"
    else:
        cmap, norm, bounds, extend = CMAP_PROB, NORM_PROB, BOUNDS_PROB, "max"

    if data is not None and np.any(~np.isnan(data)):
        cf = ax.contourf(lons, lats, data, transform=ccrs.PlateCarree(),
                         levels=bounds, cmap=cmap, norm=norm, extend=extend)
    else:
        ax.text(0.5, 0.5, "NO THREAT DETECTED IN THIS WINDOW", transform=ax.transAxes,
                fontsize=18, color="gray", ha="center", va="center", fontweight="bold", alpha=0.6)

    ax.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="bold", loc="left", linespacing=1.3)

    cbar_ax = fig.add_axes([0.18, 0.06, 0.64, 0.025])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks(bounds)
    cbar.ax.set_xticklabels([f"{b}" if plot_type == "ffg" else f"{b}%" for b in bounds], fontsize=10)
    cbar.set_label(cbar_label, fontsize=11, fontweight="bold")

    plt.subplots_adjust(bottom=0.12)
    fig.savefig(out_dir / filename, bbox_inches="tight")
    plt.close(fig)

# ==========================================
# 7. MANIFEST & FRONTEND BUILDER
# ==========================================
def update_manifest_and_html():
    all_runs = sorted([d for d in os.listdir(RUNS_DIR) if (RUNS_DIR / d).is_dir()], reverse=True)
    if len(all_runs) > 10:
        for old in all_runs[10:]:
            shutil.rmtree(RUNS_DIR / old, ignore_errors=True)
    all_runs = all_runs[:10]

    runs_meta = []
    for r in all_runs:
        try:
            r_dt = datetime.strptime(r, "%Y%m%d_%H")
            r_periods = build_synoptic_periods(r_dt)
            runs_meta.append({
                "id": r,
                "label": r_dt.strftime("%Y-%m-%d %Hz"),
                "init": r_dt.strftime("%Y-%m-%dT%H:00:00Z"),
                "periods": [
                    {
                        "id": p["id"],
                        "name": p["name"],
                        "valid_start": p["valid_start"].strftime("%Y-%m-%dT%H:00:00Z"),
                        "valid_end": p["valid_end"].strftime("%Y-%m-%dT%H:00:00Z")
                    }
                    for p in r_periods
                ]
            })
        except Exception:
            continue

    manifest = {
        "latest": all_runs[0] if all_runs else "",
        "runs": runs_meta
    }
    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    build_frontend_html(OUTPUT_DIR / "index.html")

def build_frontend_html(dest_path):
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>REFS Excessive Rainfall & FFG Interactive Viewer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {
      --bg: #090d16;
      --panel: #131b2e;
      --border: #232f48;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --accent: #2563eb;
      --accent-hover: #3b82f6;
    }
    body {
      margin: 0; background: var(--bg); color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      display: flex; flex-direction: column; align-items: center;
    }
    .header { width: 100%; max-width: 1320px; padding: 10px 16px 2px 16px; text-align: center; }
    .header h2 { margin: 0 0 2px 0; font-size: 1.25rem; }
    .header p { margin: 0; color: var(--muted); font-size: 0.8rem; }

    .toolbar {
      width: 100%; max-width: 1320px; background: var(--panel); border: 1px solid var(--border);
      border-radius: 8px; padding: 8px 12px; margin: 6px 0 8px 0; box-sizing: border-box;
      display: flex; flex-direction: column; gap: 6px;
    }
    .row {
      display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;
    }
    .btn-group {
      display: flex; gap: 4px; align-items: center; flex-wrap: wrap;
    }
    .btn-group b {
      font-size: 11.5px; color: #e2e8f0; margin-right: 2px; white-space: nowrap;
    }
    .btn {
      background: #1e293b; color: #cbd5e1; border: 1px solid var(--border);
      padding: 4px 9px; border-radius: 4px; font-size: 11.5px; font-weight: 600; cursor: pointer; transition: 0.15s;
      white-space: nowrap; line-height: 1.3;
    }
    .btn:hover:not(:disabled) { background: #334155; color: #fff; }
    .btn.active { background: var(--accent); color: #fff; border-color: var(--accent-hover); }
    .btn:disabled {
      opacity: 0.35; cursor: not-allowed; border-color: #1e293b; color: #64748b;
    }
    .btn-ffg { background: #1e3a29; border-color: #2e593e; color: #a7f3d0; }
    .btn-ffg:hover:not(:disabled) { background: #264e37; color: #fff; }
    .btn-ffg.active { background: #059669; border-color: #10b981; color: #fff; }
    .btn-help { background: #334155; color: #93c5fd; border-color: #475569; }
    .btn-help:hover { background: #475569; color: #fff; }
    
    .toolbar-divider {
      border-left: 1px solid var(--border);
      margin: 0 4px;
      height: 16px;
      display: inline-block;
    }

    select {
      background: #0f172a; color: #fff; border: 1px solid var(--border);
      padding: 3px 7px; border-radius: 4px; font-size: 11.5px; cursor: pointer;
    }

    .viewer-container {
      width: 100%; max-width: 1320px; display: flex; gap: 12px; justify-content: center;
    }
    .viewer-card {
      flex: 1; background: #000; border-radius: 8px; overflow: hidden;
      box-shadow: 0 10px 25px rgba(0,0,0,0.6); position: relative; border: 1px solid var(--border);
      display: flex; flex-direction: column;
    }
    .pane-toolbar {
      display: none; background: #0f172a; border-bottom: 1px solid var(--border);
      padding: 6px 10px; font-size: 11.5px; align-items: center; justify-content: space-between;
      flex-wrap: wrap; gap: 6px;
    }
    .viewer-card img { width: 100%; height: auto; display: block; }

    /* Modal Help Styling */
    .modal-overlay {
      display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
      background: rgba(0, 0, 0, 0.75); z-index: 9999; justify-content: center; align-items: center;
    }
    .modal-overlay.open { display: flex; }
    .modal-content {
      background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
      width: 90%; max-width: 850px; max-height: 85vh; overflow-y: auto; padding: 20px 24px;
      box-shadow: 0 20px 40px rgba(0,0,0,0.8); color: var(--text);
    }
    .modal-header {
      display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border);
      padding-bottom: 10px; margin-bottom: 14px;
    }
    .modal-header h3 { margin: 0; font-size: 1.2rem; color: #60a5fa; }
    .modal-close {
      background: transparent; border: none; font-size: 24px; color: var(--muted); cursor: pointer;
    }
    .modal-close:hover { color: #fff; }
    .help-grid { display: flex; flex-direction: column; gap: 14px; }
    .help-section {
      background: #0b1120; border: 1px solid var(--border); border-radius: 6px; padding: 12px 14px;
    }
    .help-section h4 { margin: 0 0 6px 0; font-size: 0.95rem; color: #38bdf8; display: flex; align-items: center; gap: 6px; }
    .help-item { margin-bottom: 8px; font-size: 0.85rem; line-height: 1.45; }
    .help-item:last-child { margin-bottom: 0; }
    .help-item b { color: #e2e8f0; }

    .footer { margin: 10px 0 18px 0; color: var(--muted); font-size: 11.5px; }
    kbd { background: #1e293b; border: 1px solid #334155; padding: 2px 5px; border-radius: 3px; }
  </style>
</head>
<body>
  <div class="header">
    <h2>REFS Flash Flood Guidance (FFG) & Excessive Rainfall Viewer</h2>
    <p>Operational 14-Member REFS / HRRR Ensemble Guidance Suite</p>
  </div>

  <div class="toolbar" id="mainToolbar">
    <!-- ROW 1: Temporal Controls (Model Run on left, Valid Period on right) -->
    <div class="row">
      <div class="btn-group">
        <b>Model Run:</b>
        <button class="btn" id="btnNewer" onclick="stepRun(-1)" title="Step to newer run (Arrow Up)">▲ Newer</button>
        <select id="runSelect" onchange="onMainRunChange()"></select>
        <button class="btn" id="btnOlder" onclick="stepRun(1)" title="Step to older run (Arrow Down)">▼ Older</button>
      </div>

      <div class="btn-group" id="periodBox">
        <b>Valid Period:</b>
        <div id="periodButtons" class="btn-group"></div>
      </div>
    </div>

    <!-- ROW 2: Meteorological Products (Left) & Layout / Help Controls (Right) -->
    <div class="row" style="border-top: 1px solid var(--border); padding-top: 6px;">
      <div class="btn-group" id="productGroup">
        <b>Product:</b>
        <button class="btn active" data-prod="prob_1p0x" onclick="setProduct(this)">1.0x FFG</button>
        <button class="btn" data-prod="prob_1p5x" onclick="setProduct(this)">1.5x FFG</button>
        <button class="btn" data-prod="prob_2p0x" onclick="setProduct(this)">2.0x FFG</button>
        <button class="btn" data-prod="refs_precomputed" onclick="setProduct(this)">Official REFS</button>
        <span class="toolbar-divider"></span>
        <button class="btn" data-prod="coverage_min" onclick="setProduct(this)">Cov (Min)</button>
        <button class="btn" data-prod="coverage_avg" onclick="setProduct(this)">Cov (Mean)</button>
        <button class="btn" data-prod="coverage_max" onclick="setProduct(this)">Cov (Max)</button>
        <span class="toolbar-divider"></span>
        <button class="btn btn-ffg" data-prod="ffg_1h" onclick="setProduct(this)">1-hr FFG</button>
        <button class="btn btn-ffg" data-prod="ffg_3h" onclick="setProduct(this)">3-hr FFG</button>
        <button class="btn btn-ffg" data-prod="ffg_6h" onclick="setProduct(this)">6-hr FFG</button>
      </div>

      <div class="btn-group">
        <b>Layout:</b>
        <button class="btn active" id="btnSingle" onclick="setLayout('single')">Single</button>
        <button class="btn" id="btnSplit" onclick="setLayout('split')">Split</button>
        <button class="btn btn-help" onclick="toggleHelpModal(true)" title="View explanation of products (Shortcut: ? or H)">ℹ️ Guide</button>
      </div>
    </div>
  </div>

  <div class="viewer-container" id="viewerContainer">
    <!-- PANE 1 -->
    <div class="viewer-card" id="cardPrimary">
      <div class="pane-toolbar" id="pane1Bar">
        <span style="font-weight:bold; color:#60a5fa;">Pane 1 (Left):</span>
        <div style="display:flex; gap:6px; align-items:center;">
          <select id="pane1Run" onchange="onPane1RunChange()"></select>
          <select id="pane1Period" onchange="updatePane1()"></select>
          <select id="pane1Prod" onchange="updatePane1()"></select>
        </div>
      </div>
      <img id="imgPrimary" src="" alt="Primary REFS Graphic">
    </div>

    <!-- PANE 2 -->
    <div class="viewer-card" id="cardSecondary" style="display:none;">
      <div class="pane-toolbar" id="pane2Bar">
        <span style="font-weight:bold; color:#f59e0b;">Pane 2 (Right):</span>
        <div style="display:flex; gap:6px; align-items:center;">
          <select id="pane2Run" onchange="onPane2RunChange()"></select>
          <select id="pane2Period" onchange="updatePane2()"></select>
          <select id="pane2Prod" onchange="updatePane2()"></select>
        </div>
      </div>
      <img id="imgSecondary" src="" alt="Secondary Graphic">
    </div>
  </div>

  <!-- HELP / PRODUCT GUIDE MODAL -->
  <div id="helpModal" class="modal-overlay" onclick="onModalOverlayClick(event)">
    <div class="modal-content" onclick="event.stopPropagation()">
      <div class="modal-header">
        <h3>REFS Excessive Rainfall Viewer — Product Guide</h3>
        <button class="modal-close" onclick="toggleHelpModal(false)">&times;</button>
      </div>
      <div class="help-grid">
        <div class="help-section">
          <h4>🌧️ Custom FFG Exceedance Probabilities (Rolling Window Swaths)</h4>
          <div class="help-item">
            <b>1.0x FFG:</b> Evaluates 24-hr cumulative threat of flash flooding. Shows the 40-km neighborhood probability that any member's rolling 1-hr, 3-hr, or 6-hr QPF will exceed 100% of antecedent Flash Flood Guidance at least once during the synoptic period.
          </div>
          <div class="help-item">
            <b>1.5x FFG:</b> Probability that rainfall reaches 150% of FFG. Signals moderate-to-severe flash flood risk where runoff easily overwhelms local drainage basins.
          </div>
          <div class="help-item">
            <b>2.0x FFG:</b> Probability of rainfall reaching double (200%) local FFG. Highlighting high risk for significant, life-threatening flash flooding and severe infrastructure impacts.
          </div>
        </div>

        <div class="help-section">
          <h4>🏛️ Official NCEP Product (Instantaneous Comparison)</h4>
          <div class="help-item">
            <b>Official REFS:</b> Plots the pre-computed <code>PPFFG</code> field generated by NCEP in the REFS <code>ffri</code> suite. Represents the <i>peak instantaneous</i> probability observed at any single discrete synoptic hour.
          </div>
        </div>

        <div class="help-section">
          <h4>📏 40-km Neighborhood Areal Coverage Fraction</h4>
          <div class="help-item">
            <b>Coverage (Min):</b> Minimum percentage of the 40-km neighborhood exceeding 1.0x FFG across all 14 members. Identifies the high-confidence core footprint agreed upon by the entire ensemble.
          </div>
          <div class="help-item">
            <b>Coverage (Mean):</b> Ensemble average neighborhood areal coverage fraction. Represents the expected spatial footprint and density of flash flood conditions.
          </div>
          <div class="help-item">
            <b>Coverage (Max):</b> Maximum neighborhood coverage percentage predicted by any individual member. Highlights the reasonable worst-case training or convective blowout scenario.
          </div>
        </div>

        <div class="help-section">
          <h4>🗺️ River Forecast Center (RFC) Antecedent FFG</h4>
          <div class="help-item">
            <b>1-hr, 3-hr, 6-hr FFG:</b> Operational 5-km RFC Flash Flood Guidance mosaics (in inches) ingested at model initialization. Shows rainfall required over 1, 3, or 6 hours to trigger flash flooding.
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="footer">
    Shortcuts: <kbd>←</kbd> / <kbd>→</kbd> Change Product | <kbd>↑</kbd> / <kbd>↓</kbd> Trend Cycles (dprog/dt with Valid Time Lock) | <kbd>1</kbd>–<kbd>0</kbd> Jump | <kbd>S</kbd> Split | <kbd>?</kbd> Help.
  </div>

  <script>
    let manifest = { runs: [] };
    let currentRunId = '';
    let currentProd = 'prob_1p0x';
    let currentPeriod = 'p1';
    let currentValidEnd = '';
    let layoutMode = 'single';
    const preloadedImages = new Map();

    const PRODUCTS = [
      { id: 'prob_1p0x', name: '1.0x FFG' },
      { id: 'prob_1p5x', name: '1.5x FFG' },
      { id: 'prob_2p0x', name: '2.0x FFG' },
      { id: 'refs_precomputed', name: 'Official REFS' },
      { id: 'coverage_min', name: 'Cov (Min)' },
      { id: 'coverage_avg', name: 'Cov (Mean)' },
      { id: 'coverage_max', name: 'Cov (Max)' },
      { id: 'ffg_1h', name: '1-hr FFG' },
      { id: 'ffg_3h', name: '3-hr FFG' },
      { id: 'ffg_6h', name: '6-hr FFG' }
    ];

    async function init() {
      try {
        const res = await fetch('manifest.json?t=' + Date.now());
        manifest = await res.json();
        if (!manifest.runs || manifest.runs.length === 0) return;

        currentRunId = manifest.latest || manifest.runs[0].id;
        const activeRun = manifest.runs.find(r => r.id === currentRunId) || manifest.runs[0];
        
        if (activeRun.periods.length > 0) {
          currentPeriod = activeRun.periods[0].id;
          currentValidEnd = activeRun.periods[0].valid_end;
        }

        const runSel = document.getElementById('runSelect');
        const optionsHtml = manifest.runs.map((r, i) => 
          `<option value="${r.id}">${r.label}${i === 0 ? ' (Latest)' : ''}</option>`
        ).join('');
        runSel.innerHTML = optionsHtml;
        runSel.value = currentRunId;

        ['pane1', 'pane2'].forEach(p => {
          document.getElementById(`${p}Run`).innerHTML = optionsHtml;
          document.getElementById(`${p}Prod`).innerHTML = PRODUCTS.map(pr => 
            `<option value="${pr.id}">${pr.name}</option>`
          ).join('');
        });

        renderPeriodButtons(activeRun);
        initSplitScreen(activeRun);
        preloadRunImages(currentRunId);
        updateSingleViewer();
        updateNavButtons();
      } catch (e) {
        console.error("Manifest load failed.", e);
      }
    }

    function renderPeriodButtons(runObj) {
      const pBox = document.getElementById('periodButtons');
      pBox.innerHTML = runObj.periods.map(p => 
        `<button class="btn ${p.id === currentPeriod ? 'active' : ''}" data-period="${p.id}" onclick="setPeriod('${p.id}')">${p.name}</button>`
      ).join('');
    }

    function initSplitScreen(activeRun) {
      document.getElementById('pane1Run').value = currentRunId;
      populatePeriodDropdown('pane1', activeRun);
      document.getElementById('pane1Period').value = currentPeriod;
      document.getElementById('pane1Prod').value = currentProd;

      const secondaryRun = manifest.runs.length > 1 ? manifest.runs[1] : activeRun;
      document.getElementById('pane2Run').value = secondaryRun.id;
      populatePeriodDropdown('pane2', secondaryRun);

      const matchP = secondaryRun.periods.find(p => p.valid_end === currentValidEnd);
      document.getElementById('pane2Period').value = matchP ? matchP.id : secondaryRun.periods[0].id;
      document.getElementById('pane2Prod').value = manifest.runs.length > 1 ? currentProd : 'prob_2p0x';
    }

    function populatePeriodDropdown(panePrefix, runObj) {
      document.getElementById(`${panePrefix}Period`).innerHTML = runObj.periods.map(pr => 
        `<option value="${pr.id}">${pr.name}</option>`
      ).join('');
    }

    function updateNavButtons() {
      const btnNewer = document.getElementById('btnNewer');
      const btnOlder = document.getElementById('btnOlder');
      const curIdx = manifest.runs.findIndex(r => r.id === currentRunId);
      if (curIdx === -1) return;

      const isFfg = currentProd.startsWith('ffg_');

      if (isFfg) {
        btnNewer.disabled = (curIdx <= 0);
        btnOlder.disabled = (curIdx >= manifest.runs.length - 1);
        btnNewer.title = btnNewer.disabled ? "No newer run available" : "Step to newer run (Arrow Up)";
        btnOlder.title = btnOlder.disabled ? "No older run available" : "Step to older run (Arrow Down)";
      } else {
        const hasNewer = (curIdx > 0) && manifest.runs[curIdx - 1].periods.some(p => p.valid_end === currentValidEnd);
        const hasOlder = (curIdx < manifest.runs.length - 1) && manifest.runs[curIdx + 1].periods.some(p => p.valid_end === currentValidEnd);

        btnNewer.disabled = !hasNewer;
        btnOlder.disabled = !hasOlder;
        btnNewer.title = !hasNewer ? "No newer run covers this ending valid time" : "Step to newer run (Arrow Up)";
        btnOlder.title = !hasOlder ? "Reached earliest run covering this ending valid time" : "Step to older run (Arrow Down)";
      }
    }

    function changeRun(newRunId, targetPeriodId) {
      const newRunObj = manifest.runs.find(r => r.id === newRunId);
      if (!newRunObj) return;

      currentRunId = newRunId;
      document.getElementById('runSelect').value = newRunId;

      if (targetPeriodId) {
        currentPeriod = targetPeriodId;
        const pObj = newRunObj.periods.find(p => p.id === targetPeriodId);
        if (pObj) currentValidEnd = pObj.valid_end;
      } else {
        if (!currentProd.startsWith('ffg_')) {
          const match = newRunObj.periods.find(p => p.valid_end === currentValidEnd);
          if (match) {
            currentPeriod = match.id;
          } else {
            const fallback = newRunObj.periods[newRunObj.periods.length - 1];
            currentPeriod = fallback.id;
            currentValidEnd = fallback.valid_end;
          }
        }
      }

      renderPeriodButtons(newRunObj);
      preloadRunImages(newRunId);
      updateSingleViewer();
      updateNavButtons();
    }

    function getImageUrl(runId, periodId, prod) {
      if (prod.startsWith('ffg_')) {
        return `runs/${runId}/${prod}.png`;
      }
      return `runs/${runId}/${periodId}_${prod}.png`;
    }

    function preloadRunImages(runId) {
      const runObj = manifest.runs.find(r => r.id === runId);
      if (!runObj) return;

      PRODUCTS.forEach(p => {
        if (p.id.startsWith('ffg_')) {
          const u = getImageUrl(runId, 'p1', p.id);
          if (!preloadedImages.has(u)) {
            const im = new Image(); im.src = u; preloadedImages.set(u, im);
          }
        } else {
          runObj.periods.forEach(pr => {
            const u = getImageUrl(runId, pr.id, p.id);
            if (!preloadedImages.has(u)) {
              const im = new Image(); im.src = u; preloadedImages.set(u, im);
            }
          });
        }
      });
    }

    function setProduct(btn) {
      document.querySelectorAll('[data-prod]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentProd = btn.getAttribute('data-prod');
      
      const pBox = document.getElementById('periodBox');
      if (currentProd.startsWith('ffg_')) {
        pBox.style.opacity = '0.35';
        pBox.style.pointerEvents = 'none';
      } else {
        pBox.style.opacity = '1.0';
        pBox.style.pointerEvents = 'auto';
      }
      updateSingleViewer();
      updateNavButtons();
    }

    function setPeriod(pId) {
      currentPeriod = pId;
      const activeRun = manifest.runs.find(r => r.id === currentRunId);
      if (activeRun) {
        const pObj = activeRun.periods.find(p => p.id === pId);
        if (pObj) currentValidEnd = pObj.valid_end;
        renderPeriodButtons(activeRun);
      }
      updateSingleViewer();
      updateNavButtons();
    }

    function onMainRunChange() {
      changeRun(document.getElementById('runSelect').value, null);
    }

    function stepRun(delta) {
      const curIdx = manifest.runs.findIndex(r => r.id === currentRunId);
      if (curIdx === -1) return;

      const nextIdx = curIdx + delta;
      if (nextIdx < 0 || nextIdx >= manifest.runs.length) return;

      const isFfg = currentProd.startsWith('ffg_');
      if (isFfg) {
        changeRun(manifest.runs[nextIdx].id, null);
        return;
      }

      const candidateRun = manifest.runs[nextIdx];
      const matchP = candidateRun.periods.find(p => p.valid_end === currentValidEnd);
      if (!matchP) {
        return;
      }

      changeRun(candidateRun.id, matchP.id);
    }

    function stepProduct(delta) {
      const prodIds = PRODUCTS.map(p => p.id);
      let idx = prodIds.indexOf(currentProd) + delta;
      if (idx < 0) idx = prodIds.length - 1;
      if (idx >= prodIds.length) idx = 0;
      const nextBtn = document.querySelector(`[data-prod="${prodIds[idx]}"]`);
      if (nextBtn) setProduct(nextBtn);
    }

    function setLayout(mode) {
      layoutMode = mode;
      document.getElementById('btnSingle').classList.toggle('active', mode === 'single');
      document.getElementById('btnSplit').classList.toggle('active', mode === 'split');
      
      const isSplit = mode === 'split';
      document.getElementById('cardSecondary').style.display = isSplit ? 'flex' : 'none';
      document.getElementById('pane1Bar').style.display = isSplit ? 'flex' : 'none';
      document.getElementById('pane2Bar').style.display = isSplit ? 'flex' : 'none';

      if (isSplit) {
        document.getElementById('pane1Run').value = currentRunId;
        const activeRun = manifest.runs.find(r => r.id === currentRunId);
        populatePeriodDropdown('pane1', activeRun);
        document.getElementById('pane1Period').value = currentPeriod;
        document.getElementById('pane1Prod').value = currentProd;
        updatePane1();
        updatePane2();
      } else {
        updateSingleViewer();
      }
    }

    function updateSingleViewer() {
      if (layoutMode === 'split') return;
      document.getElementById('imgPrimary').src = getImageUrl(currentRunId, currentPeriod, currentProd);
    }

    function updatePane1() {
      const r = document.getElementById('pane1Run').value;
      const p = document.getElementById('pane1Period').value;
      const prod = document.getElementById('pane1Prod').value;
      document.getElementById('imgPrimary').src = getImageUrl(r, p, prod);
    }

    function updatePane2() {
      const r = document.getElementById('pane2Run').value;
      const p = document.getElementById('pane2Period').value;
      const prod = document.getElementById('pane2Prod').value;
      preloadRunImages(r);
      document.getElementById('imgSecondary').src = getImageUrl(r, p, prod);
    }

    function onPane1RunChange() {
      const rId = document.getElementById('pane1Run').value;
      const runObj = manifest.runs.find(r => r.id === rId);
      populatePeriodDropdown('pane1', runObj);
      updatePane1();
    }

    function onPane2RunChange() {
      const rId = document.getElementById('pane2Run').value;
      const runObj = manifest.runs.find(r => r.id === rId);
      populatePeriodDropdown('pane2', runObj);
      updatePane2();
    }

    function toggleHelpModal(show) {
      document.getElementById('helpModal').classList.toggle('open', show);
    }

    function onModalOverlayClick(e) {
      if (e.target.classList.contains('modal-overlay')) {
        toggleHelpModal(false);
      }
    }

    document.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowUp') { e.preventDefault(); stepRun(-1); }
      if (e.key === 'ArrowDown') { e.preventDefault(); stepRun(1); }
      if (e.key === 'ArrowLeft') { e.preventDefault(); stepProduct(-1); }
      if (e.key === 'ArrowRight') { e.preventDefault(); stepProduct(1); }
      if (e.key === 's' || e.key === 'S') { setLayout(layoutMode === 'single' ? 'split' : 'single'); }
      if (e.key === '?' || e.key === 'h' || e.key === 'H') {
        const modal = document.getElementById('helpModal');
        toggleHelpModal(!modal.classList.contains('open'));
      }
      if (e.key === 'Escape') { toggleHelpModal(false); }
      if (e.key >= '1' && e.key <= '9') {
        const target = PRODUCTS[parseInt(e.key) - 1].id;
        const btn = document.querySelector(`[data-prod="${target}"]`);
        if (btn) setProduct(btn);
      }
      if (e.key === '0') {
        const btn = document.querySelector(`[data-prod="${PRODUCTS[9].id}"]`);
        if (btn) setProduct(btn);
      }
    });

    init();
  </script>
</body>
</html>
"""
    with open(dest_path, "w") as f:
        f.write(html)

# ==========================================
# 8. MAIN PRODUCTION ENGINE & BACKFILLER
# ==========================================
def process_single_cycle(init_dt):
    cycle_id = init_dt.strftime("%Y%m%d_%H")
    current_run_dir = RUNS_DIR / cycle_id
    current_run_dir.mkdir(parents=True, exist_ok=True)
    flag_file = current_run_dir / "complete.json"

    print(f"\n==========================================")
    print(f"PROCESSING REFS RUN: {cycle_id}")
    print(f"==========================================")

    periods = build_synoptic_periods(init_dt)
    full_catalog = get_member_catalog(init_dt)

    test_task = get_member_file_info(full_catalog[0], 1)
    download_apcp_slice(test_task[0], test_task[1], test_task[2], test_task[3])
    ds_init = xr.open_dataset(test_task[2], engine="cfgrib")
    ref_lats = ds_init.latitude.values
    ref_lons = ds_init.longitude.values
    ds_init.close()
    if test_task[2].exists():
        test_task[2].unlink()

    print("\n--- Rendering RFC 5km Flash Flood Guidance (FFG) Baselines ---")
    ffg = get_aligned_ffg(ref_lats, ref_lons, init_dt)

    for dur_key, dur_label in [("1h", "1 HR"), ("3h", "3 HR"), ("6h", "6 HR")]:
        render_plot(
            ffg[dur_key], ref_lats, ref_lons,
            f"{init_dt.strftime('%H')}Z 5km RFC {dur_label} FFG (inches)",
            f"Antecedent Flash Flood Guidance valid at Model Initialization ({init_dt.strftime('%Y-%m-%d %H')}Z)",
            "Inches",
            f"ffg_{dur_key}.png", current_run_dir, plot_type="ffg"
        )
        print(f" -> Generated ffg_{dur_key}.png")

    for p in periods:
        p_id = p["id"]
        req_lead_max = p["f_end"]
        valid_mems = [m for m in full_catalog if m["max_lead"] >= req_lead_max]
        k_members = len(valid_mems)

        print(f"\n>>> Running Period {p['name']} (F{p['f_start']:03d}–F{p['f_end']:03d})")
        print(f"    Active Eligible Members: {k_members}/14")

        if k_members == 0:
            print("    No members available for this period. Skipping.")
            continue

        envelope_sum = {t: np.zeros((1059, 1799), dtype=np.float32) for t in THRESHOLDS}
        member_coverages = []

        for m_idx, mem in enumerate(valid_mems):
            print(f"    [{m_idx + 1}/{k_members}] Streaming member: {mem['id']}")
            fxx_list = list(range(p["f_start"], p["f_end"] + 1))
            tasks = [get_member_file_info(mem, lead) for lead in fxx_list]

            with ThreadPoolExecutor(max_workers=16) as pool:
                list(pool.map(lambda t: download_apcp_slice(*t), tasks))

            qpf_list = []
            for _, _, fpath, _ in tasks:
                if not fpath.exists():
                    qpf_list.append(np.zeros((1059, 1799), dtype=np.float32))
                    continue
                try:
                    ds = xr.open_dataset(fpath, engine="cfgrib")
                    v = list(ds.data_vars)[0]
                    qpf_list.append((ds[v] / 25.4).values.astype(np.float32))
                    ds.close()
                except Exception:
                    qpf_list.append(np.zeros((1059, 1799), dtype=np.float32))

            qpf_stack = np.stack(qpf_list, axis=0)
            nt = qpf_stack.shape[0]
            max_ratio = np.zeros((1059, 1799), dtype=np.float32)

            for t in range(nt):
                with np.errstate(divide="ignore", invalid="ignore"):
                    max_ratio = np.fmax(max_ratio, qpf_stack[t] / ffg["1h"])
                if t >= 2:
                    q3 = np.sum(qpf_stack[t-2:t+1], axis=0)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        max_ratio = np.fmax(max_ratio, q3 / ffg["3h"])
                if t >= 5:
                    q6 = np.sum(qpf_stack[t-5:t+1], axis=0)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        max_ratio = np.fmax(max_ratio, q6 / ffg["6h"])

            for t_val in THRESHOLDS:
                bin_exceed = (max_ratio >= t_val).astype(np.float32)
                dilated = maximum_filter(bin_exceed, footprint=CIRCULAR_FOOTPRINT)
                envelope_sum[t_val] += dilated

            bin_1x = (max_ratio >= 1.0).astype(np.float32)
            cov_grid = convolve(bin_1x, CIRCULAR_FOOTPRINT, mode="constant", cval=0.0)
            cov_pct = (cov_grid / FOOTPRINT_SUM) * 100.0
            member_coverages.append(cov_pct.astype(np.float16))

            for _, _, fpath, _ in tasks:
                if fpath.exists():
                    fpath.unlink()
            del qpf_stack, max_ratio
            gc.collect()

        valid_range_str = f"Valid: {p['valid_start'].strftime('%Y-%m-%d %H')}Z – {p['valid_end'].strftime('%Y-%m-%d %H')}Z"
        sub_title = f"{p['name']} | {valid_range_str} ({k_members}-Member Ensemble Blend)"

        for t_val, fname in [(1.0, f"{p_id}_prob_1p0x.png"), (1.5, f"{p_id}_prob_1p5x.png"), (2.0, f"{p_id}_prob_2p0x.png")]:
            raw_prob = (envelope_sum[t_val] / float(k_members)) * 100.0
            smoothed = ndimage.gaussian_filter(raw_prob, sigma=4.0)
            data_disp = np.where(smoothed >= 5.0, smoothed, np.nan)
            render_plot(data_disp, ref_lats, ref_lons,
                        f"REFS {init_dt.strftime('%Y%m%d %H')}z: Probability of Exceeding {t_val}x FFG",
                        sub_title, "40-km Neighborhood Probability of Exceedance (%) [Smoothed]",
                        fname, current_run_dir, plot_type="prob")

        refs_ppffg = extract_precomputed_refs(init_dt, p)
        render_plot(refs_ppffg, ref_lats, ref_lons,
                    f"REFS {init_dt.strftime('%Y%m%d %H')}z: Official Pre-Computed Max FFG Exceedance",
                    f"{p['name']} | {valid_range_str} (NCEP Operational ffri)",
                    "Official REFS Exceedance Probability (%) [Smoothed]",
                    f"{p_id}_refs_precomputed.png", current_run_dir, plot_type="prob")

        cov_stack = np.stack(member_coverages, axis=0)
        c_min = np.min(cov_stack, axis=0).astype(np.float32)
        c_avg = np.mean(cov_stack, axis=0).astype(np.float32)
        c_max = np.max(cov_stack, axis=0).astype(np.float32)

        for c_data, c_label, fname in [
            (c_min, "Minimum Ensemble Areal Coverage", f"{p_id}_coverage_min.png"),
            (c_avg, "Ensemble Mean Areal Coverage", f"{p_id}_coverage_avg.png"),
            (c_max, "Maximum Ensemble Areal Coverage", f"{p_id}_coverage_max.png")
        ]:
            c_disp = np.where(c_data >= 1.0, c_data, np.nan)
            render_plot(c_disp, ref_lats, ref_lons,
                        f"REFS {init_dt.strftime('%Y%m%d %H')}z: 1.0x FFG Exceedance Coverage ({c_label})",
                        sub_title, "40-km Neighborhood Fraction of Exceedance (%)",
                        fname, current_run_dir, plot_type="cov")

        del member_coverages, cov_stack
        gc.collect()

    with open(flag_file, "w") as f:
        json.dump({"cycle": cycle_id, "completed_at": datetime.now(timezone.utc).isoformat()}, f)
    print(f"Cycle {cycle_id} marked complete.")

def run_production():
    print("==========================================")
    print("CHECKING FOR REFS CYCLES TO PROCESS...")
    print("==========================================")

    pending_cycles = get_pending_and_available_cycles(max_lookback_hours=36)

    if not pending_cycles:
        print("All recent available cycles already processed and cached.")
        update_manifest_and_html()
        print("Manifest and viewer verified. Exiting cleanly.")
        return

    print(f"Found {len(pending_cycles)} pending cycle(s): {[c.strftime('%Y%m%d_%H') for c in pending_cycles]}")

    for cand_dt in pending_cycles[:3]:
        process_single_cycle(cand_dt)

    update_manifest_and_html()
    print("\nBatch production and manifest update complete!")

if __name__ == "__main__":
    run_production()
