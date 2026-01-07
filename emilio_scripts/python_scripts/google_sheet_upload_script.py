# from source_functions import *
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from googleapiclient.errors import HttpError
from matplotlib.colors import LogNorm
mpl.use('macosx')
import gspread
from io import BytesIO
import time
import random
import httplib2
from google_auth_httplib2 import AuthorizedHttp
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload
from pathlib import Path

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


# ---------------- Google auth + clients ----------------
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
            # exponential backoff + jitter
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


def upload_fig_to_cell(ws, drive, fig, cell, upload_name, *, dpi=300, height=180, width=180):
    buf = BytesIO()
    try:

        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)

        # buf.seek(0, 2)
        # print("PNG bytes:", buf.tell())
        # buf.seek(0)

        media = MediaIoBaseUpload(buf, mimetype="image/png", resumable=True)
        req = drive.files().create(
            body={"name": upload_name},
            media_body=media,
            fields="id"
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


# ---------------- HDF helpers ----------------
def load_common_arrays(hdf_path: Path):
    """Load arrays used by multiple plots once."""
    with h5py.File(hdf_path, "r") as f:
        dynamic_roi_map = f["xpcs/qmap/dynamic_roi_map"][...]
        scattering_2d = f["xpcs/temporal_mean/scattering_2d"][...]
    scattering_2d_reshape = scattering_2d[0, :, :]
    return dynamic_roi_map, scattering_2d_reshape


def compute_idxs(dynamic_roi_map: np.ndarray, scattering_2d_reshape: np.ndarray, n_masks=300):
    """Find the brightest mask index and return the 3x3 neighborhood idx list you use."""
    intens = np.zeros(n_masks, dtype=np.float64)

    for mi in range(n_masks):
        m = (dynamic_roi_map == mi).astype(np.int8)  # small dtype
        intens[mi] = np.sum(scattering_2d_reshape * m)

    peak = int(np.argmax(intens))
    idxs = (np.array([-29, 1, 31, -30, 0, 30, -31, -1, 29]) + peak).tolist()
    return idxs


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


# ---------------- Plot functions ----------------
def make_overview_fig(sample_id: str, dynamic_roi_map, scattering_2d_reshape, idxs):
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
    ax.set_title(f"{sample_id} overview")
    fig.tight_layout()
    return fig


def make_g2s_fig(sample_id: str, g2, q, phi, idxs):
    fig, ax = plt.subplots(figsize=(7, 7))

    x = np.arange(g2.shape[0])
    for mi in idxs:
        # keeping your indexing convention, but guarded
        j = mi - 1
        if j < 0 or j >= g2.shape[1]:
            continue

        ax.semilogx(
            x,
            g2[:, j],
            label=(f"M{mi}, q={q[int(mi // 30)]:.3f}, phi={phi[int(mi % 30)]:.3f}")
        )

    ax.set_title(f"g2 autocorrelation for experiment {sample_id}")
    ax.set_ylabel("g2(q,tau)")
    ax.set_xlabel("Delay Time, tau")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def make_twotime_fig(sample_id: str, hdf_path: Path, idxs):
    fig, axes = plt.subplots(3, 3, figsize=(7, 7))

    for k, ax in enumerate(axes.flat):
        mi = idxs[k]
        C = load_c2_map(hdf_path, mi)

        C = C + C.T - np.diag(np.diag(C))
        lo, hi = np.percentile(C, [0, 99.9])
        C = np.clip(C, lo, hi)

        ax.axis("off")
        ax.imshow(C, origin="lower", cmap="plasma")

        label = f"M{mi}\nmin {np.min(C):.2f}\nmax {np.max(C):.2f}"
        ax.text(
            0.05, 0.95, label,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=12,
            color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.6, edgecolor="none")
        )

    fig.suptitle(f"{sample_id} two-time correlation plots")
    fig.tight_layout()
    return fig


def process_one_scan(sample_id, row, hdf_path, ws, drive,*, position_name: str, out_dir: Path):
    dynamic_roi_map, scattering_2d_reshape = load_common_arrays(hdf_path)
    idxs = compute_idxs(dynamic_roi_map, scattering_2d_reshape)
    g2, q, phi = load_g2_q_phi(hdf_path)

    figs = {
        "overview": make_overview_fig(sample_id, dynamic_roi_map, scattering_2d_reshape, idxs),
        "g2s": make_g2s_fig(sample_id, g2, q, phi, idxs),
        "twotime": make_twotime_fig(sample_id, hdf_path, idxs),
    }

    try:
        for key, fig in figs.items():
            dpi = DPI_BY_PLOT[key]

            # 1) save locally
            local_path = save_fig_local(fig, out_dir, position_name, key, sample_id, dpi=dpi)
            print(f"Saved local: {local_path}")

            # 2) upload to sheet (your existing method)
            cell = f"{PLOT_COLS[key]}{row}"
            upload_name = f"{sample_id}_{key}.png"
            upload_fig_to_cell(ws, drive, fig, cell, upload_name, dpi=dpi)
            print(f"Wrote {sample_id} {key} -> {ws.title}!{cell}")
    finally:
        for fig in figs.values():
            plt.close(fig)


def save_fig_local(fig, out_dir: Path, position_name: str, fig_key: str, sample_id: str, *, dpi: int):
    subdir = out_dir / position_name / FIGTYPE_DIR.get(fig_key, fig_key)
    subdir.mkdir(parents=True, exist_ok=True)

    out_path = subdir / f"{sample_id}.png"
    fig.savefig(out_path, format="png", dpi=dpi, bbox_inches="tight")
    return out_path


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
        # find the first occurrence of that ID in the filtered list
        start_pos = None
        for i, (r, sid) in enumerate(rows_and_ids):
            if sid == start_sample_id:
                start_pos = i
                break
        if start_pos is None:
            raise ValueError(f"start_sample_id={start_sample_id} not found in position {position_name}")
        rows_and_ids = rows_and_ids[start_pos:]

    else:
        # start by index (default 0 = from beginning)
        rows_and_ids = rows_and_ids[start_index:]

    print(f"Starting from: {rows_and_ids[0] if rows_and_ids else 'nothing to do'}")

    for row, sample_id in rows_and_ids:
        hdf_path = find_results_hdf(base_dir, sample_id)
        if hdf_path is None:
            print(f"SKIP: no HDF file found for {sample_id}")
            continue

        print(f"Processing {sample_id} (row {row})")
        process_one_scan(sample_id, row, hdf_path, ws, drive, position_name=position_name, out_dir=OUT_DIR)


# ---------------- CONFIG ----------------
SPREADSHEET_ID = "1OAA7H4I3cgas32aSZkrLB8TOKHymMAv2uk_0eTywWcQ"
TAB_NAME = "IPA NBH"
TOKEN_PATH = "token.json"
CREDS_PATH = "client_secret_180145739842-0ug37lsh4qltki62e8te8bqkde9u25jb.apps.googleusercontent.com.json"

# BASE_DIR = Path("/Users/emilioescauriza/Desktop")
BASE_DIR = Path("/Volumes/EmilioSD4TB/APS_08-IDEI-2025-1006/Twotime_PostExpt_01")
# BASE_DIR = Path("/Volumes/Eriks_4TB/shpyrko202510/analysis/Twotime_PostExpt_01")
POSITION_NAME = "A6"

OUT_DIR = Path("//Users/emilioescauriza/Documents/repos/006_APS_8IDE/emilio_scripts/figures_export")  # choose where you want
FIGTYPE_DIR = {
    "overview": "9_mask_overview",
    "g2s": "9_mask_g2",
    "twotime": "9_mask_ttc",
}

PLOT_COLS = {
    "overview": "AJ",
    "g2s": "AK",
    "twotime": "AL",
}

DPI_BY_PLOT = {
    "overview": 300,
    "g2s": 300,
    "twotime": 100,
}

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


if __name__ == "__main__":

    creds = get_creds(TOKEN_PATH, CREDS_PATH)
    ws, drive = get_ws_and_drive(creds, SPREADSHEET_ID, TAB_NAME)

    process_position(POSITION_NAME, BASE_DIR, ws, drive)  # run all the files for a position AXXXX
    # process_position(POSITION_NAME, BASE_DIR, ws, drive, start_sample_id="A017")  # start from a position AXXXX

    pass