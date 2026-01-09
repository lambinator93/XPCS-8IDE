# ============================================================
# Imports
# ============================================================

# from source_functions import *

import time
import random
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np

import matplotlib as mpl
mpl.use("macosx")  # must be set before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import gspread

import httplib2
from google_auth_httplib2 import AuthorizedHttp

from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from googleapiclient.errors import HttpError


# ============================================================
# Sheets scanning helpers
# ============================================================

def get_ids_for_position(ws, position_name, id_col=1, position_col=3, header_rows=1):
    """
    Return a list of (row_number, sample_id) for rows where column C == position_name.

    Parameters
    ----------
    ws : gspread.Worksheet
    position_name : str
        e.g. "A5"
    id_col : int
        Column index for sample ID (A = 1)
    position_col : int
        Column index for position name (C = 3)
    header_rows : int
        Number of header rows to skip (usually 1)

    Returns
    -------
    list of (int, str)
        [(row_number, sample_id), ...]
    """
    col_ids = ws.col_values(id_col)
    col_pos = ws.col_values(position_col)

    n = min(len(col_ids), len(col_pos))
    results = []

    for i in range(header_rows, n):
        if col_pos[i].strip() == position_name:
            results.append((i + 1, col_ids[i].strip()))

    return results


def find_results_hdf(base_dir: Path, sample_id: str) -> Path | None:
    pattern = f"{sample_id}_*_results.hdf"
    matches = sorted(base_dir.glob(pattern))
    return matches[0] if matches else None


# ============================================================
# Plot selection / filtering helpers
# ============================================================

def normalize_keys(keys):
    return ALL_PLOT_KEYS if keys is None else set(keys)


def should_generate(key: str, generate_keys) -> bool:
    return key in normalize_keys(generate_keys)


def should_upload(key: str, upload_enabled: bool, upload_keys, generate_keys) -> bool:
    if not upload_enabled:
        return False
    if not should_generate(key, generate_keys):
        return False
    if upload_keys is None:
        return True
    return key in set(upload_keys)


# ============================================================
# Google auth + API clients
# ============================================================

def get_creds(token_path, creds_path):
    creds = None
    if Path(token_path).exists():
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)
            Path(token_path).write_text(creds.to_json())

    return creds


def execute_with_backoff(request, tries=6, base_delay=1.0):
    for attempt in range(tries):
        try:
            return request.execute()
        except (HttpError, ConnectionResetError, TimeoutError) as e:
            if attempt == tries - 1:
                raise
            sleep_s = base_delay * (2 ** attempt) + random.random()
            print(f"Upload failed ({type(e).__name__}), retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)


def get_ws_and_drive(creds, spreadsheet_id, tab_name):
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(spreadsheet_id)
    ws = sh.worksheet(tab_name)

    authed_http = AuthorizedHttp(creds, http=httplib2.Http(timeout=300))
    drive = build("drive", "v3", http=authed_http, cache_discovery=False)

    print("Opened spreadsheet:", sh.title, "| worksheet:", ws.title)
    return ws, drive


# ============================================================
# Upload + local save helpers
# ============================================================

def upload_fig_to_cell(ws, drive, fig, cell, upload_name, *, dpi=300, height=180, width=180):
    buf = BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)

        media = MediaIoBaseUpload(buf, mimetype="image/png", resumable=True)
        req = drive.files().create(
            body={"name": upload_name, "parents": [UPLOAD_FOLDER_ID]},
            media_body=media,
            fields="id",
        )
        created = execute_with_backoff(req)

        file_id = created["id"]

        drive.permissions().create(
            fileId=file_id,
            body={"type": "anyone", "role": "reader"},
        ).execute()

        image_url = f"https://drive.google.com/uc?export=view&id={file_id}"
        formula = f'=IMAGE("{image_url}")'

        ws.update(
            range_name=cell,
            values=[[formula]],
            value_input_option="USER_ENTERED",
        )
        return file_id

    finally:
        buf.close()


def save_fig_local(fig, out_dir: Path, position_name: str, fig_key: str, sample_id: str, *, dpi: int):
    subdir = out_dir / position_name / FIGTYPE_DIR[fig_key]
    subdir.mkdir(parents=True, exist_ok=True)
    out_path = subdir / f"{sample_id}.png"
    fig.savefig(out_path, format="png", dpi=dpi, bbox_inches="tight")
    return out_path


# ============================================================
# HDF loading + mask selection
# ============================================================

def load_common_arrays(hdf_path: Path):
    """Load arrays used by multiple plots once."""
    with h5py.File(hdf_path, "r") as f:
        dynamic_roi_map = f["xpcs/qmap/dynamic_roi_map"][...]
        scattering_2d = f["xpcs/temporal_mean/scattering_2d"][...]
    scattering_2d_reshape = scattering_2d[0, :, :]
    return dynamic_roi_map, scattering_2d_reshape


def neighborhood_offsets(n: int, stride: int = 30) -> list[int]:
    """
    Offsets for an n×n neighborhood around a center index in a flattened (q,phi) grid.

    This ordering matches your original 3×3 layout:
        [-29,   1,  31,
         -30,   0,  30,
         -31,  -1,  29]
    i.e. rows step in phi (dx), columns step in q (dy*stride).
    """
    if n % 2 == 0:
        raise ValueError("n must be odd (e.g., 3, 5).")

    r = n // 2
    offsets = []

    # rows: dx = +r ... -r (top to bottom)
    # cols: dy = -r ... +r (left to right)
    for dx in range(r, -r - 1, -1):
        for dy in range(-r, r + 1):
            offsets.append(dy * stride + dx)

    return offsets


def compute_idxs(dynamic_roi_map, scattering_2d_reshape, n_masks=300, grid_n=3, stride=30):
    intens = np.zeros(n_masks, dtype=np.float64)
    for mi in range(n_masks):
        m = (dynamic_roi_map == mi).astype(np.int8)
        intens[mi] = np.sum(scattering_2d_reshape * m)

    peak = int(np.argmax(intens))
    offsets = neighborhood_offsets(grid_n, stride=stride)
    idxs = [peak + off for off in offsets]
    return peak, idxs


def load_g2_q_phi(hdf_path: Path):
    with h5py.File(hdf_path, "r") as f:
        g2 = f["xpcs/twotime/normalized_g2"][...]
        q = f["xpcs/qmap/dynamic_v_list_dim0"][...]
        phi = f["xpcs/qmap/dynamic_v_list_dim1"][...]
    return g2, q, phi


def load_c2_map(hdf_path: Path, mask_idx: int):
    ttc_tree = f"xpcs/twotime/correlation_map/c2_00{mask_idx:03d}"
    with h5py.File(hdf_path, "r") as f:
        C = f[ttc_tree][...]
    return C


# ============================================================
# Plot builders
# ============================================================

def make_overview_fig(sample_id, dynamic_roi_map, scattering_2d_reshape, idxs, title_suffix="overview"):
    combined_mask = np.isin(dynamic_roi_map, idxs).astype(np.int8)

    fig, ax = plt.subplots()
    I = scattering_2d_reshape.astype(float, copy=False).copy()
    I[combined_mask == 1] *= 10

    cmap = plt.cm.plasma.copy()
    cmap.set_under("black")
    cmap.set_bad("black")

    ys, xs = np.where(combined_mask == 1)
    cy = int(np.round(ys.mean()))
    cx = int(np.round(xs.mean()))

    half = 200
    ymin = max(cy - half, 0)
    ymax = min(cy + half, I.shape[0])
    xmin = max(cx - half, 0)
    xmax = min(cx + half, I.shape[1])

    img_crop = I[ymin:ymax, xmin:xmax]

    im = ax.imshow(
        img_crop,
        origin="lower",
        cmap=cmap,
        norm=LogNorm(vmin=0.1, vmax=float(np.nanmax(I))),
    )
    fig.colorbar(im, ax=ax)
    ax.set_title(f"{sample_id} {title_suffix}")
    fig.tight_layout()
    return fig


def make_g2s_fig(sample_id, g2, q, phi, idxs, title_suffix="g2"):
    fig, ax = plt.subplots(figsize=(7, 7))
    x = np.arange(g2.shape[0])

    for mi in idxs:
        j = mi - 1
        if j < 0 or j >= g2.shape[1]:
            continue
        ax.semilogx(x, g2[:, j], label=f"M{mi}")

    ax.set_title(f"{sample_id} {title_suffix}")
    ax.set_ylabel("g2(q,tau)")
    ax.set_xlabel("Delay Time, tau")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    return fig


def make_twotime_fig(sample_id, hdf_path, idxs, grid_n=3, figsize=(7, 7), title_suffix="two-time"):
    fig, axes = plt.subplots(grid_n, grid_n, figsize=figsize)
    axes = np.array(axes).reshape(grid_n, grid_n)

    for k, ax in enumerate(axes.flat):
        mi = idxs[k]
        ttc_tree = f"xpcs/twotime/correlation_map/c2_00{mi:03d}"
        with h5py.File(hdf_path, "r") as f:
            C = f[ttc_tree][...]

        C = C + C.T - np.diag(np.diag(C))
        lo, hi = np.percentile(C, [0, 99.9])
        C = np.clip(C, lo, hi)

        ax.axis("off")
        ax.imshow(C, origin="lower", cmap="plasma")

        label = (
            f"M{mi}\n"
            f"min {np.min(C):.2f}\n"
            f"max {np.max(C):.2f}"
        )

        ax.text(
            0.05, 0.95,
            label,
            transform=ax.transAxes,  # axes-relative coordinates
            ha="left",
            va="top",
            fontsize=12,
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="black",
                alpha=0.6,
                edgecolor="none",
            ),
        )

    fig.suptitle(f"{sample_id} {title_suffix}", fontsize=24)
    fig.tight_layout()
    return fig


# ============================================================
# Main processing pipeline
# ============================================================

def process_one_scan(
    sample_id,
    row,
    hdf_path,
    ws,
    drive,
    *,
    position_name: str,
    out_dir: Path,
    generate_keys=None,
    upload_enabled=True,
    upload_keys=None,
):
    gen = normalize_keys(generate_keys)

    dynamic_roi_map, scattering_2d_reshape = load_common_arrays(hdf_path)
    g2, q, phi = load_g2_q_phi(hdf_path)

    need_9 = any(k in gen for k in ("overview_9", "g2s_9", "twotime_9"))
    need_25 = any(k in gen for k in ("overview_25", "g2s_25", "twotime_25"))

    idxs_9 = idxs_25 = None
    if need_9:
        _, idxs_9 = compute_idxs(dynamic_roi_map, scattering_2d_reshape, grid_n=3, stride=30)
    if need_25:
        _, idxs_25 = compute_idxs(dynamic_roi_map, scattering_2d_reshape, grid_n=5, stride=30)

    figs = {}

    # 9-mask set
    if "overview_9" in gen:
        figs["overview_9"] = make_overview_fig(sample_id, dynamic_roi_map, scattering_2d_reshape, idxs_9, "9-mask overview")
    if "g2s_9" in gen:
        figs["g2s_9"] = make_g2s_fig(sample_id, g2, q, phi, idxs_9, "9-mask g2")
    if "twotime_9" in gen:
        figs["twotime_9"] = make_twotime_fig(sample_id, hdf_path, idxs_9, grid_n=3, figsize=(7, 7), title_suffix="9-mask two-time")

    # 25-mask set
    if "overview_25" in gen:
        figs["overview_25"] = make_overview_fig(sample_id, dynamic_roi_map, scattering_2d_reshape, idxs_25, "25-mask overview")
    if "g2s_25" in gen:
        figs["g2s_25"] = make_g2s_fig(sample_id, g2, q, phi, idxs_25, "25-mask g2")
    if "twotime_25" in gen:
        figs["twotime_25"] = make_twotime_fig(sample_id, hdf_path, idxs_25, grid_n=5, figsize=(12, 12), title_suffix="25-mask two-time")

    try:
        for key, fig in figs.items():
            dpi = DPI_BY_PLOT[key]

            # local save (always for generated figures)
            local_path = save_fig_local(fig, out_dir, position_name, key, sample_id, dpi=dpi)

            # upload (only if enabled + selected)
            if should_upload(key, upload_enabled, upload_keys, generate_keys):
                print(f"Saved local + uploaded: {local_path}")
                cell = f"{PLOT_COLS[key]}{row}"
                upload_name = f"{sample_id}_{key}.png"
                upload_fig_to_cell(ws, drive, fig, cell, upload_name, dpi=dpi)
                print(f"  → Sheets: {ws.title}!{cell}")
            else:
                if not upload_enabled:
                    print(f"Saved local (Sheets disabled): {local_path}")
                elif upload_keys is not None and key not in set(upload_keys):
                    print(f"Saved local (Sheets skipped – key '{key}' not selected): {local_path}")
                else:
                    print(f"Saved local (Sheets skipped): {local_path}")

    finally:
        for fig in figs.values():
            plt.close(fig)


def process_position(
    position_name,
    base_dir: Path,
    ws,
    drive,
    *,
    start_sample_id: str | None = None,
    start_row: int | None = None,
    start_index: int = 0,
):
    rows_and_ids = get_ids_for_position(ws, position_name)
    print(f"Found {len(rows_and_ids)} scans at position {position_name}")

    # Decide where to start
    if start_row is not None:
        rows_and_ids = [(r, sid) for (r, sid) in rows_and_ids if r >= start_row]

    elif start_sample_id is not None:
        start_pos = None
        for i, (r, sid) in enumerate(rows_and_ids):
            if sid == start_sample_id:
                start_pos = i
                break
        if start_pos is None:
            raise ValueError(f"start_sample_id={start_sample_id} not found in position {position_name}")
        rows_and_ids = rows_and_ids[start_pos:]

    else:
        rows_and_ids = rows_and_ids[start_index:]

    print(f"Starting from: {rows_and_ids[0] if rows_and_ids else 'nothing to do'}")

    for row, sample_id in rows_and_ids:
        hdf_path = find_results_hdf(base_dir, sample_id)
        if hdf_path is None:
            print(f"SKIP: no HDF file found for {sample_id}")
            continue

        print(f"Processing {sample_id} (row {row})")
        process_one_scan(
            sample_id, row, hdf_path, ws, drive,
            position_name=POSITION_NAME,
            out_dir=OUT_DIR,
            generate_keys=GENERATE_KEYS,
            upload_enabled=UPLOAD_TO_SHEETS,
            upload_keys=UPLOAD_KEYS,
        )


def plot_single_mask_scan(
    sample_id: str,
    mask_n: int,
    base_dir: Path,
    out_dir: Path,
    *,
    dpi: int = 250,
    figsize=(14, 5),
):
    """
    Plot overview + g2 + TTC for ONE mask index from ONE scan,
    and save locally as a single combined PNG.

    Output:
      out_dir/individual_scan_plots/<sample_id>/<sample_id>_mask_<mask_n>.png
    """

    # --------------------------------------------------
    # Locate HDF
    # --------------------------------------------------
    hdf_path = find_results_hdf(base_dir, sample_id)
    if hdf_path is None:
        raise FileNotFoundError(f"No results HDF found for {sample_id}")

    # --------------------------------------------------
    # Load required data
    # --------------------------------------------------
    dynamic_roi_map, scattering_2d_reshape = load_common_arrays(hdf_path)
    g2, q, phi = load_g2_q_phi(hdf_path)
    C = load_c2_map(hdf_path, mask_n)

    # --------------------------------------------------
    # Build combined figure
    # --------------------------------------------------
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.2, 1.1])

    # ---------- (1) Overview ----------
    ax0 = fig.add_subplot(gs[0])
    mask = (dynamic_roi_map == mask_n).astype(np.int8)

    I = scattering_2d_reshape.astype(float, copy=False).copy()
    I[mask == 1] *= 10

    ys, xs = np.where(mask == 1)
    cy, cx = int(ys.mean()), int(xs.mean())
    half = 200
    ymin, ymax = max(cy-half, 0), min(cy+half, I.shape[0])
    xmin, xmax = max(cx-half, 0), min(cx+half, I.shape[1])

    im0 = ax0.imshow(
        I[ymin:ymax, xmin:xmax],
        origin="lower",
        cmap="plasma",
        norm=LogNorm(vmin=0.1, vmax=np.nanmax(I)),
    )
    ax0.set_title(f"{sample_id} mask {mask_n}\noverview")
    ax0.axis("off")
    fig.colorbar(im0, ax=ax0, fraction=0.046)

    # ---------- (2) g2 ----------
    ax1 = fig.add_subplot(gs[1])
    tau = np.arange(g2.shape[0])
    j = mask_n - 1
    if 0 <= j < g2.shape[1]:
        ax1.semilogx(tau, g2[:, j], lw=2)
        ax1.set_title(
            f"g2\nq={q[int(mask_n//30)]:.3f}, φ={phi[int(mask_n%30)]:.3f}"
        )
    ax1.set_xlabel("Delay time τ")
    ax1.set_ylabel("g2(τ)")
    ax1.grid(True, alpha=0.3)

    # ---------- (3) Two-time ----------
    ax2 = fig.add_subplot(gs[2])
    C = C + C.T - np.diag(np.diag(C))
    lo, hi = np.percentile(C, [0, 99.9])
    C = np.clip(C, lo, hi)

    im2 = ax2.imshow(C, origin="lower", cmap="plasma")
    ax2.set_title("Two-time correlation")
    ax2.set_xlabel("t₁")
    ax2.set_ylabel("t₂")
    fig.colorbar(im2, ax=ax2, fraction=0.046)

    fig.suptitle(f"{sample_id} – mask {mask_n}", fontsize=16)
    fig.tight_layout()

    # --------------------------------------------------
    # Save locally
    # --------------------------------------------------
    out_path = (
        out_dir
        / "individual_scan_plots"
        / sample_id
    )
    out_path.mkdir(parents=True, exist_ok=True)

    fname = out_path / f"{sample_id}_mask_{mask_n:03d}.png"
    fig.savefig(fname, dpi=dpi, bbox_inches="tight")

    plt.show()

    plt.close(fig)

    return fname

# ============================================================
# CONFIG
# ============================================================

SPREADSHEET_ID = "1OAA7H4I3cgas32aSZkrLB8TOKHymMAv2uk_0eTywWcQ"
TAB_NAME = "IPA NBH"
TOKEN_PATH = "token.json"
CREDS_PATH = "client_secret_180145739842-0ug37lsh4qltki62e8te8bqkde9u25jb.apps.googleusercontent.com.json"

UPLOAD_FOLDER_ID = "18IccfznbNEgAewkGqAmXwhaLA9yTrYa-"
OUT_DIR = Path("//Users/emilioescauriza/Documents/repos/006_APS_8IDE/emilio_scripts/figures_export")

FIGTYPE_DIR = {
    "overview_9": "9_mask_overview",
    "g2s_9": "9_mask_g2",
    "twotime_9": "9_mask_twotime",
    "overview_25": "25_mask_overview",
    "g2s_25": "25_mask_g2",
    "twotime_25": "25_mask_twotime",
}

PLOT_COLS = {
    "overview_9": "AJ",
    "g2s_9": "AK",
    "twotime_9": "AL",
    "overview_25": "AM",
    "g2s_25": "AN",
    "twotime_25": "AO",
}

ALL_PLOT_KEYS = {
    "overview_9", "g2s_9", "twotime_9",
    "overview_25", "g2s_25", "twotime_25",
}

DPI_BY_PLOT = {
    "overview_9": 300,
    "g2s_9": 300,
    "twotime_9": 150,
    "overview_25": 250,
    "g2s_25": 250,
    "twotime_25": 80,
}

OFFSETS_3 = neighborhood_offsets(3, stride=30)
OFFSETS_5 = neighborhood_offsets(5, stride=30)

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# BASE_DIR = Path("/Users/emilioescauriza/Desktop")
BASE_DIR = Path("/Volumes/EmilioSD4TB/APS_08-IDEI-2025-1006/Twotime_PostExpt_01")
# BASE_DIR = Path("/Volumes/Eriks_4TB/shpyrko202510/analysis/Twotime_PostExpt_01")
POSITION_NAME = "A5"
# Which plots to generate (None = generate all 6)
GENERATE_KEYS = {"twotime_9", "twotime_25"}  # None or for example: {"overview_25", "twotime_25"} to only generate these two
UPLOAD_TO_SHEETS = True  # True of False
UPLOAD_KEYS = None  # or example: {"overview_25"} to upload only overview_25 only but still generate others

if __name__ == "__main__":

    creds = get_creds(TOKEN_PATH, CREDS_PATH)
    ws, drive = get_ws_and_drive(creds, SPREADSHEET_ID, TAB_NAME)

    # process_position(POSITION_NAME, BASE_DIR, ws, drive)  # run all the files for a position AXXXX
    process_position(POSITION_NAME, BASE_DIR, ws, drive, start_sample_id="A084")  # start from a position AXXXX

    # plot_single_mask_scan(
    #     sample_id="A073",
    #     mask_n=116,
    #     base_dir=BASE_DIR,
    #     out_dir=OUT_DIR,
    # )

    pass