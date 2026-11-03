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
def get_latest_cycle_dt():
    now = datetime.now(timezone.utc)
    anchor = now.replace(minute=0, second=0, microsecond=0)
    for h_back in range(0, 36, 6):
        cand = anchor - timedelta(hours=h_back)
        cycle_hour = (cand.hour // 6) * 6
        cand = cand.replace(hour=cycle_hour)
        date_str = cand.strftime("%Y%m%d")
        
        test_url = f"https://noaa-rrfs-ops-pds.s3.amazonaws.com/rrfs.{date_str}/{cycle_hour:02d}/rrfs.t{cycle_hour:02d}z.2dfld.3km.f001.conus.grib2.idx"
        try:
            r = requests.head(test_url, timeout=4)
            if r.status_code == 200:
                print(f"Locked in latest available cycle: {date_str} {cycle_hour:02d}z")
                return cand
        except Exception:
            continue
    raise RuntimeError("No recent REFS cycle found on AWS Open Data.")

def build_synoptic_periods(init_dt):
    cycle = init_dt.hour
    periods = []

    if cycle == 12:
        periods.append({"id": "p1", "name": "Day 1 ERO", "f_start": 1, "f_end": 24, "hours": 24,
                        "valid_start": init_dt, "valid_end": init_dt + timedelta(hours=24)})
        periods.append({"id": "p2", "name": "Day 2 ERO", "f_start": 25, "f_end": 48, "hours": 24,
                        "valid_start": init_dt + timedelta(hours=24), "valid_end": init_dt + timedelta(hours=48)})
        periods.append({"id": "p3", "name": "Day 3 ERO", "f_start": 49, "f_end": 72, "hours": 24,
                        "valid_start": init_dt + timedelta(hours=48), "valid_end": init_dt + timedelta(hours=72)})
    elif cycle == 18:
        periods.append({"id": "p1", "name": "Day 1 (Partial 18h)", "f_start": 1, "f_end": 18, "hours": 18,
                        "valid_start": init_dt, "valid_end": init_dt + timedelta(hours=18)})
        periods.append({"id": "p2", "name": "Day 2 ERO", "f_start": 19, "f_end": 42, "hours": 24,
                        "valid_start": init_dt + timedelta(hours=18), "valid_end": init_dt + timedelta(hours=42)})
        periods.append({"id": "p3", "name": "Day 3 ERO", "f_start": 43, "f_end": 66, "hours": 24,
                        "valid_start": init_dt + timedelta(hours=42), "valid_end": init_dt + timedelta(hours=66)})
    elif cycle == 0:
        periods.append({"id": "p1", "name": "Day 1 (Remainder 12h)", "f_start": 1, "f_end": 12, "hours": 12,
                        "valid_start": init_dt, "valid_end": init_dt + timedelta(hours=12)})
        periods.append({"id": "p2", "name": "Next Day ERO", "f_start": 13, "f_end": 36, "hours": 24,
                        "valid_start": init_dt + timedelta(hours=12), "valid_end": init_dt + timedelta(hours=36)})
        periods.append({"id": "p3", "name": "Day 2 ERO", "f_start": 37, "f_end": 60, "hours": 24,
                        "valid_start": init_dt + timedelta(hours=36), "valid_end": init_dt + timedelta(hours=60)})
    elif cycle == 6:
        periods.append({"id": "p1", "name": "Day 1 (Remainder 6h)", "f_start": 1, "f_end": 6, "hours": 6,
                        "valid_start": init_dt, "valid_end": init_dt + timedelta(hours=6)})
        periods.append({"id": "p2", "name": "Next Day ERO", "f_start": 7, "f_end": 30, "hours": 24,
                        "valid_start": init_dt + timedelta(hours=6), "valid_end": init_dt + timedelta(hours=30)})
        periods.append({"id": "p3", "name": "Day 2 ERO", "f_start": 31, "f_end": 54, "hours": 24,
                        "valid_start": init_dt + timedelta(hours=30), "valid_end": init_dt + timedelta(hours=54)})

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
def update_manifest_and_html(periods):
    all_runs = sorted([d for d in os.listdir(RUNS_DIR) if (RUNS_DIR / d).is_dir()], reverse=True)
    if len(all_runs) > 10:
        for old in all_runs[10:]:
            shutil.rmtree(RUNS_DIR / old, ignore_errors=True)
    all_runs = all_runs[:10]

    manifest = {
        "latest": all_runs[0] if all_runs else "",
        "runs": [{"id": r, "label": datetime.strptime(r, "%Y%m%d_%H").strftime("%Y-%m-%d %Hz")} for r in all_runs],
        "periods": [{"id": p["id"], "name": p["name"]} for p in periods]
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
    .header { width: 100%; max-width: 1320px; padding: 12px 20px 4px 20px; text-align: center; }
    .header h2 { margin: 0 0 4px 0; font-size: 1.35rem; }
    .header p { margin: 0; color: var(--muted); font-size: 0.85rem; }

    .toolbar {
      width: 100%; max-width: 1320px; background: var(--panel); border: 1px solid var(--border);
      border-radius: 8px; padding: 10px 14px; margin: 8px 0; box-sizing: border-box;
      display: flex; flex-direction: column; gap: 8px;
    }
    .row { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; }
    .btn-group { display: flex; gap: 4px; align-items: center; flex-wrap: wrap; }
    .btn {
      background: #1e293b; color: #cbd5e1; border: 1px solid var(--border);
      padding: 5px 11px; border-radius: 5px; font-size: 12px; font-weight: 600; cursor: pointer; transition: 0.15s;
    }
    .btn:hover { background: #334155; color: #fff; }
    .btn.active { background: var(--accent); color: #fff; border-color: var(--accent-hover); }
    .btn-ffg { background: #1e3a29; border-color: #2e593e; color: #a7f3d0; }
    .btn-ffg:hover { background: #264e37; color: #fff; }
    .btn-ffg.active { background: #059669; border-color: #10b981; color: #fff; }
    
    select {
      background: #0f172a; color: #fff; border: 1px solid var(--border);
      padding: 4px 8px; border-radius: 5px; font-size: 12px; cursor: pointer;
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

    .footer { margin: 12px 0 20px 0; color: var(--muted); font-size: 11.5px; }
    kbd { background: #1e293b; border: 1px solid #334155; padding: 2px 5px; border-radius: 3px; }
  </style>
</head>
<body>
  <div class="header">
    <h2>REFS Flash Flood Guidance (FFG) & Excessive Rainfall Viewer</h2>
    <p>Operational 14-Member REFS / HRRR Ensemble Guidance Suite</p>
  </div>

  <div class="toolbar" id="mainToolbar">
    <div class="row">
      <div class="btn-group">
        <b>Model Run:</b>
        <button class="btn" onclick="stepRun(-1)">▲ Newer</button>
        <select id="runSelect" onchange="onMainRunChange()"></select>
        <button class="btn" onclick="stepRun(1)">▼ Older</button>
      </div>

      <div class="btn-group" id="periodBox">
        <b>Synoptic Period:</b>
        <div id="periodButtons" class="btn-group"></div>
      </div>

      <div class="btn-group">
        <b>Layout:</b>
        <button class="btn active" id="btnSingle" onclick="setLayout('single')">Single</button>
        <button class="btn" id="btnSplit" onclick="setLayout('split')">Split Screen</button>
      </div>
    </div>

    <div class="row" style="border-top: 1px solid var(--border); padding-top: 8px;">
      <div class="btn-group">
        <b>Product:</b>
        <button class="btn active" data-prod="prob_1p0x" onclick="setProduct(this)">1.0x FFG</button>
        <button class="btn" data-prod="prob_1p5x" onclick="setProduct(this)">1.5x FFG</button>
        <button class="btn" data-prod="prob_2p0x" onclick="setProduct(this)">2.0x FFG</button>
        <button class="btn" data-prod="refs_precomputed" onclick="setProduct(this)">Official REFS</button>
        <span style="border-left: 1px solid var(--border); margin: 0 4px; height: 18px;"></span>
        <button class="btn" data-prod="coverage_min" onclick="setProduct(this)">Cov (Min)</button>
        <button class="btn" data-prod="coverage_avg" onclick="setProduct(this)">Cov (Mean)</button>
        <button class="btn" data-prod="coverage_max" onclick="setProduct(this)">Cov (Max)</button>
        <span style="border-left: 1px solid var(--border); margin: 0 4px; height: 18px;"></span>
        <button class="btn btn-ffg" data-prod="ffg_1h" onclick="setProduct(this)">1-hr FFG</button>
        <button class="btn btn-ffg" data-prod="ffg_3h" onclick="setProduct(this)">3-hr FFG</button>
        <button class="btn btn-ffg" data-prod="ffg_6h" onclick="setProduct(this)">6-hr FFG</button>
      </div>
    </div>
  </div>

  <div class="viewer-container" id="viewerContainer">
    <!-- PANE 1 -->
    <div class="viewer-card" id="cardPrimary">
      <div class="pane-toolbar" id="pane1Bar">
        <span style="font-weight:bold; color:#60a5fa;">Pane 1 (Left):</span>
        <div style="display:flex; gap:6px; align-items:center;">
          <select id="pane1Run" onchange="updatePane1()"></select>
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
          <select id="pane2Run" onchange="updatePane2()"></select>
          <select id="pane2Period" onchange="updatePane2()"></select>
          <select id="pane2Prod" onchange="updatePane2()"></select>
        </div>
      </div>
      <img id="imgSecondary" src="" alt="Secondary Graphic">
    </div>
  </div>

  <div class="footer">
    Shortcuts: <kbd>←</kbd> / <kbd>→</kbd> Change Product | <kbd>↑</kbd> / <kbd>↓</kbd> Trend Cycles (dprog/dt) | <kbd>1</kbd>–<kbd>0</kbd> Direct Product Jump | <kbd>S</kbd> Toggle Split.
  </div>

  <script>
    let manifest = { runs: [], periods: [] };
    let currentProd = 'prob_1p0x';
    let currentPeriod = 'p1';
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
        
        const runSel = document.getElementById('runSelect');
        const optionsHtml = manifest.runs.map((r, i) => 
          `<option value="${r.id}">${r.label}${i === 0 ? ' (Latest)' : ''}</option>`
        ).join('');
        runSel.innerHTML = optionsHtml;
        runSel.value = manifest.latest;

        const pBox = document.getElementById('periodButtons');
        pBox.innerHTML = manifest.periods.map((p, i) => 
          `<button class="btn ${i === 0 ? 'active' : ''}" data-period="${p.id}" onclick="setPeriod(this)">${p.name}</button>`
        ).join('');
        
        if (manifest.periods.length > 0) currentPeriod = manifest.periods[0].id;
        
        // Populate Split Screen Dropdowns
        ['pane1', 'pane2'].forEach(p => {
          document.getElementById(`${p}Run`).innerHTML = optionsHtml;
          document.getElementById(`${p}Period`).innerHTML = manifest.periods.map(pr => 
            `<option value="${pr.id}">${pr.name}</option>`
          ).join('');
          document.getElementById(`${p}Prod`).innerHTML = PRODUCTS.map(pr => 
            `<option value="${pr.id}">${pr.name}</option>`
          ).join('');
        });

        // Smart defaults for Split Screen
        document.getElementById('pane1Run').value = manifest.latest;
        document.getElementById('pane1Period').value = currentPeriod;
        document.getElementById('pane1Prod').value = currentProd;

        if (manifest.runs.length > 1) {
          document.getElementById('pane2Run').value = manifest.runs[1].id; // previous run
          document.getElementById('pane2Prod').value = currentProd;
        } else {
          document.getElementById('pane2Run').value = manifest.latest;
          document.getElementById('pane2Prod').value = 'prob_2p0x'; // severity contrast
        }
        document.getElementById('pane2Period').value = currentPeriod;

        preloadRunImages(manifest.latest);
        updateSingleViewer();
      } catch (e) {
        console.error("Manifest load failed.", e);
      }
    }

    function getImageUrl(runId, periodId, prod) {
      if (prod.startsWith('ffg_')) {
        return `runs/${runId}/${prod}.png`;
      }
      return `runs/${runId}/${periodId}_${prod}.png`;
    }

    function preloadRunImages(runId) {
      PRODUCTS.forEach(p => {
        if (p.id.startsWith('ffg_')) {
          const u = getImageUrl(runId, 'p1', p.id);
          if (!preloadedImages.has(u)) {
            const im = new Image(); im.src = u; preloadedImages.set(u, im);
          }
        } else {
          manifest.periods.forEach(pr => {
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
        pBox.style.opacity = '0.4';
        pBox.style.pointerEvents = 'none';
      } else {
        pBox.style.opacity = '1.0';
        pBox.style.pointerEvents = 'auto';
      }
      updateSingleViewer();
    }

    function setPeriod(btn) {
      document.querySelectorAll('[data-period]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentPeriod = btn.getAttribute('data-period');
      updateSingleViewer();
    }

    function onMainRunChange() {
      updateSingleViewer();
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
        document.getElementById('pane1Run').value = document.getElementById('runSelect').value;
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
      const curRun = document.getElementById('runSelect').value;
      document.getElementById('imgPrimary').src = getImageUrl(curRun, currentPeriod, currentProd);
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

    function stepRun(delta) {
      const sel = document.getElementById('runSelect');
      let idx = sel.selectedIndex + delta;
      if (idx >= 0 && idx < sel.options.length) {
        sel.selectedIndex = idx;
        preloadRunImages(sel.value);
        updateSingleViewer();
      }
    }

    function stepProduct(delta) {
      const prodIds = PRODUCTS.map(p => p.id);
      let idx = prodIds.indexOf(currentProd) + delta;
      if (idx < 0) idx = prodIds.length - 1;
      if (idx >= prodIds.length) idx = 0;
      const nextBtn = document.querySelector(`[data-prod="${prodIds[idx]}"]`);
      if (nextBtn) setProduct(nextBtn);
    }

    document.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowUp') { e.preventDefault(); stepRun(-1); }
      if (e.key === 'ArrowDown') { e.preventDefault(); stepRun(1); }
      if (e.key === 'ArrowLeft') { e.preventDefault(); stepProduct(-1); }
      if (e.key === 'ArrowRight') { e.preventDefault(); stepProduct(1); }
      if (e.key === 's' || e.key === 'S') { setLayout(layoutMode === 'single' ? 'split' : 'single'); }
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
# 8. MAIN PRODUCTION ENGINE
# ==========================================
def run_production():
    init_dt = get_latest_cycle_dt()
    cycle_id = init_dt.strftime("%Y%m%d_%H")
    current_run_dir = RUNS_DIR / cycle_id
    periods = build_synoptic_periods(init_dt)

    # CHECK IF ALREADY COMPLETED (FAST SKIP LOGIC)
    flag_file = current_run_dir / "complete.json"
    if current_run_dir.exists() and flag_file.exists():
        print(f"\n==========================================")
        print(f"Cycle {cycle_id} is already downloaded and fully processed.")
        print(f"Skipping heavy downloads. Verifying manifest and HTML...")
        print(f"==========================================")
        update_manifest_and_html(periods)
        print("Done! Exiting cleanly in < 3 seconds.")
        return

    current_run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n==========================================")
    print(f"PROCESSING REFS ERO VIEWER: {cycle_id}")
    print(f"==========================================")

    full_catalog = get_member_catalog(init_dt)

    # 1. Fetch grid geometry and render FFG baseline maps
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

    # 2. Process synoptic periods
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

    # 3. Mark run complete
    with open(flag_file, "w") as f:
        json.dump({"cycle": cycle_id, "completed_at": datetime.now(timezone.utc).isoformat()}, f)

    # 4. Update manifest and HTML frontend
    update_manifest_and_html(periods)
    print(f"\nExecution Complete! Output archived in {current_run_dir}")

if __name__ == "__main__":
    run_production()
