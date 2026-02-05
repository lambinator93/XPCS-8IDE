import sys
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms.operators.binary import difference

from emilio_scripts.python_scripts.raw_data_inspection import integrated_intensities_inspector
# from erik_file_transfer.notebooks.xpcs_1 import intensity
from source_functions import *
from xpcs import *
from sims import *
# from autocorrelations import *
import cv2
from scipy.special import erfinv
import pyopencl as cl
from pathlib import Path
import json
from scipy.fft import fft, ifft, fftfreq


def print_h5_structure(name, obj):
    print(name)

def explore(name, obj):
    indent = "  " * (name.count("/") - 1)
    if isinstance(obj, h5py.Group):
        print(f"{indent}{name}/  (Group)")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}{name}  (Dataset)  shape={obj.shape}, dtype={obj.dtype}")

def h5_file_inspector(filename):
    with h5py.File(filename, "r") as f:
        f.visititems(explore)

def g2_plotter(filename):
    with h5py.File(filename, "r") as f:
        g2 = f["xpcs/twotime/normalized_g2"][...]

    x = np.arange(len(g2[:, 100]))

    plt.figure()
    # plt.errorbar(delay[0, :], G2_result, yerr=G2_error, fmt='none', ecolor='b', capsize=2)
    plt.semilogx(x, g2[:, 194], 'b.')
    plt.title('g2 autocorrelation with pixelwise normalisation')
    plt.ylabel('g2(q,tau)')
    plt.xlabel('Delay Time, tau (s)')
    # plt.ylim([0, 1.5])

    plt.show()

def ttc_plotter(filename):
    with h5py.File(filename, "r") as f:
        C = f["xpcs/twotime/correlation_map/c2_00194"][...]
    C = C + C.T - np.diag(np.diag(C))
    # renomalize C
    C = C - np.min(C)
    C = C / np.max(C)
    # C_extent = [0, frameSpacing * det.shape[0], 0, frameSpacing * det.shape[0]]
    # C_extent = [0, det.shape[0], 0, det.shape[0]]

    plt.rcParams.update({'font.size': 24})

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_ylabel('t2 (s)')
    ax.set_xlabel('t1 (s)')
    ax.set_ylabel('Frame number')
    ax.set_xlabel('Frame number')
    # ax.set_title(f'Ring {0}')
    # ax.set_title('Whole Mask')
    im = ax.imshow(C, origin="lower", cmap='plasma')
    cbar = fig.colorbar(im, ax=ax)
    custom_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    cbar.set_ticks(custom_ticks)

    # diag_vals = np.diag(C)
    # x_min, x_max, y_min, y_max = C_extent
    # n = C.shape[0]  # assuming square matrix
    # t_axis = np.linspace(x_min, x_max, n)
    #
    # plt.figure(figsize=(8, 4))
    # plt.plot(np.arange(0, len(t_axis), 1), 1 - diag_vals, marker='.', linestyle='-', alpha=0.8, color='C1')
    # plt.xlabel('t (s)')
    # plt.ylabel('C(t,t)')
    # plt.title('Diagonal of C (t1 = t2)')
    # plt.grid(True)
    # plt.tight_layout()

    plt.show()

def intensity_vs_time(filename):
    with h5py.File(filename, "r") as f:
        data = f["xpcs/spatial_mean/intensity_vs_time"][...]

    print(data.shape)

    plt.figure()
    plt.plot(data[0, :], data[1, :])
    plt.show()

def static_vs_dynamic_bins(filename):
    with h5py.File(filename, "r") as f:
        dynamic_roi_map = f["xpcs/qmap/dynamic_roi_map"][...]
        scattering_2d = f["xpcs/temporal_mean/scattering_2d"][...]

    scattering_2d_reshape = scattering_2d[0, :, :]

    individual_mask_intensity = []


    print(dynamic_roi_map.shape)
    print(scattering_2d.shape)
    print(scattering_2d_reshape.shape)

    for i in range(300):
        individual_mask = dynamic_roi_map.copy()
        individual_mask[individual_mask != i+1] = 0
        individual_mask[individual_mask != 0] = 1
        scattering_2d_masked = scattering_2d_reshape * individual_mask
        individual_mask_intensity.append(np.sum(scattering_2d_masked))

    # individual_mask = dynamic_roi_map.copy()
    # individual_mask[individual_mask != 100] = 0
    # individual_mask[individual_mask != 0] = 1
    # scattering_2d_masked = scattering_2d_reshape * individual_mask
    # individual_mask_intensity.append(np.sum(scattering_2d_masked))
    #
    # individual_mask_intensity = np.array(individual_mask_intensity)

    plt.figure()
    # plt.plot(individual_mask_intensity)
    plt.semilogy(np.arange(1, 301, 1), individual_mask_intensity)
    plt.xlabel('mask number')
    plt.ylabel('integrated intensity')


    # plt.figure()
    # plt.imshow(individual_mask)

    plt.figure()
    plt.imshow(dynamic_roi_map)

    plt.figure()
    plt.imshow(individual_mask)

    cmap = plt.cm.plasma.copy()
    cmap.set_under("black")
    cmap.set_bad("black")

    I = scattering_2d_reshape.astype(float)

    plt.imshow(I,
        origin="upper",
        cmap=cmap,
        norm=LogNorm(vmin=0.1, vmax=np.max(I)),
        interpolation="nearest",
    )
    # ax0.set_facecolor("black")

    # np.savetxt('dynamic_roi_map.txt', dynamic_roi_map)
    # print(np.shape(dynamic_roi_map))


    plt.show()

def combined_plot(filename):
    with h5py.File(filename, "r") as f:
        dynamic_roi_map = f["xpcs/qmap/dynamic_roi_map"][...]
        scattering_2d = f["xpcs/temporal_mean/scattering_2d"][...]
        ttc = f["xpcs/twotime/correlation_map/c2_00194"][...]
        g2 = f["xpcs/twotime/normalized_g2"][...]
        q = f["xpcs/qmap/dynamic_v_list_dim0"][...]
        phi = f["xpcs/qmap/dynamic_v_list_dim1"][...]

    run_name = os.path.basename(h5_file).split("_")[0]


    scattering_2d_reshape = scattering_2d[0, :, :]
    individual_mask_intensity = []

    print('q:', q)

    print('phi:', phi)

    for index in np.arange(0, 300, 1):
        # print('index:', index, ', q:', q[10-int(np.floor(index/10))], ', phi:', phi[int(np.floor(index/30))])
        print('index:', index, ', x:', int((index // 30)), ', q:', q[int((index // 30))],
              ', y:', int(index % 30), ', phi:', phi[int(index % 30)])

    individual_mask = dynamic_roi_map.copy()
    # individual_mask[(individual_mask != 165) & (individual_mask != 225)] = 0
    individual_mask[individual_mask != 135] = 0
    individual_mask[individual_mask != 0] = 1

    for i in range(300):
        individual_mask = dynamic_roi_map.copy()
        individual_mask[individual_mask != i] = 0
        individual_mask[individual_mask != 0] = 1
        scattering_2d_masked = scattering_2d_reshape * individual_mask
        individual_mask_intensity.append(np.sum(scattering_2d_masked))

    i = np.argmax(individual_mask_intensity)
    # idxs = [0, 1, -1, -31, -30, -29, 29, 30, 31] + i
    idxs = [-29, 1, 31, -30, 0, 30, -31, -1, 29] + i

    masks = dynamic_roi_map.copy()
    combined_mask = np.isin(masks, idxs).astype(int)

    im = dynamic_roi_map.copy()
    im[~np.isin(im, idxs)] = 0

    plt.figure()
    plt.imshow(im)

    plt.figure()
    x = np.arange(len(g2[:, 100]))
    # plt.errorbar(delay[0, :], G2_result, yerr=G2_error, fmt='none', ecolor='b', capsize=2)
    for i in idxs:
        plt.semilogx(x, g2[:, i-1], label='M' + str(i) + ', q='+f"{q[int((i // 30))]:.3f}"
                                        + ',  phi='+f"{phi[int(i % 30)]:.3f}")
    plt.title('g2 autocorrelation for experiment ' + run_name)
    plt.ylabel('g2(q,tau)')
    plt.xlabel('Delay Time, tau')
    plt.legend()

    fig, axes = plt.subplots(3, 3, figsize=(7, 7))

    for i, ax in enumerate(axes.flat):
        path = f"xpcs/twotime/correlation_map/c2_00{idxs[i]:03d}"
        with h5py.File(filename, "r") as f:
            C = f[path][...]
        C = C + C.T - np.diag(np.diag(C))
        # renomalize C
        # C = C - np.min(C)
        # C = C / np.max(C)
        lo, hi = np.percentile(C, [0, 99.9])
        C = np.clip(C, lo, hi)
        # C = (C_clip - lo) / (hi - lo)
        # ax.set_title(f"M{idxs[i]}")
        ax.axis("off")
        # ax.set_ylabel('Frame number')
        # ax.set_xlabel('Frame number')
        im = ax.imshow(C, origin="lower", cmap='plasma')
        label = (
            f"M{idxs[i]}\n"
            f"min {np.min(C):.2f}\n"
            f"max {np.max(C):.2f}"
        )
        ax.text(
            0.05, 0.95, label,
            transform=ax.transAxes,  # axes-relative coordinates
            ha="left", va="top",
            fontsize=12,
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="black",
                alpha=0.6,
                edgecolor="none"
            )
        )
        print(path)

    plt.tight_layout()

    plt.figure()

    I = scattering_2d_reshape.astype(float)
    I[combined_mask == 1] *= 10

    cmap = plt.cm.plasma.copy()
    cmap.set_under("black")  # or "navy", etc.
    cmap.set_bad("black")  # for NaN/inf

    ys, xs = np.where(combined_mask == 1)
    cy = int(np.round(ys.mean()))
    cx = int(np.round(xs.mean()))
    half = 200
    ymin = max(cy - half, 0)
    ymax = min(cy + half, I.shape[0])
    xmin = max(cx - half, 0)
    xmax = min(cx + half, I.shape[1])
    img_crop = I[ymin:ymax, xmin:xmax]
    mask_crop = combined_mask[ymin:ymax, xmin:xmax]

    plt.imshow(img_crop, origin="lower", cmap=cmap, norm=LogNorm(vmin=0.1, vmax=I.max()))
    plt.colorbar()

    # plt.figure()
    # plt.imshow(combined_mask)

    print(run_name)
    print(filename)

    plt.show()


def oauth_test():
    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = ('/Users/emilioescauriza/Documents/repos/006_APS_8IDE/emilio_scripts/python_scripts/client_secret_'
                  '180145739842-0ug37lsh4qltki62e8te8bqkde9u25jb.apps.googleusercontent.com.json')

    def get_creds():
        creds = None
        if Path("token.json").exists():
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials, SCOPES
                )
                creds = flow.run_local_server(port=0)
            Path("token.json").write_text(creds.to_json())

        return creds

    creds = get_creds()
    gc = gspread.authorize(creds)

    # Paste your spreadsheet ID here
    sh = gc.open_by_key("1OAA7H4I3cgas32aSZkrLB8TOKHymMAv2uk_0eTywWcQ")
    print("Opened spreadsheet:", sh.title)


def image_upload(fig, target_cell="AF142", upload_name="matplotlib_output.png",
                 tab_name="IPA NBH",
                 spreadsheet_id="1OAA7H4I3cgas32aSZkrLB8TOKHymMAv2uk_0eTywWcQ",
                 token_path="token.json",
                 creds_path="client_secret_180145739842-0ug37lsh4qltki62e8te8bqkde9u25jb.apps.googleusercontent.com.json"):

    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    def get_creds():
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

    creds = get_creds()

    gc = gspread.authorize(creds)
    sh = gc.open_by_key(spreadsheet_id)
    ws = sh.worksheet(tab_name)

    cols, rows = find_rows_with_position(ws, "A5")

    print(cols)

    for cell in rows_to_cells(rows, "AF"):
        print(cell)
        # image_upload(fig, target_cell=cell, upload_name=f"A5_{cell}.png")


    # drive = build("drive", "v3", credentials=creds)
    #
    # buf = BytesIO()
    # try:
    #     fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    #     buf.seek(0)
    #
    #     media = MediaIoBaseUpload(buf, mimetype="image/png", resumable=False)
    #     created = drive.files().create(
    #         body={"name": upload_name},
    #         media_body=media,
    #         fields="id"
    #     ).execute()
    #
    #     file_id = created["id"]
    #
    #     drive.permissions().create(
    #         fileId=file_id,
    #         body={"type": "anyone", "role": "reader"},
    #     ).execute()
    #
    #     image_url = f"https://drive.google.com/uc?export=view&id={file_id}"
    #     formula = f'=IMAGE("{image_url}", 4, 180, 320)'
    #
    #     ws.update(target_cell, [[formula]], value_input_option="USER_ENTERED")
    #     print(f"Inserted image into {ws.title} cell {target_cell}")
    #
    # finally:
    #     buf.close()

def figure_upload():

    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [1, 4, 2])
    image_upload(fig, target_cell="AF142", upload_name="run123_overview.png")
    plt.close(fig)

def q_spacing_inspector(filename):
    with h5py.File(filename, "r") as f:
        dynamic_roi_map = f["xpcs/qmap/dynamic_roi_map"][...]
        scattering_2d = f["xpcs/temporal_mean/scattering_2d"][...]
        ttc = f["xpcs/twotime/correlation_map/c2_00194"][...]
        g2 = f["xpcs/twotime/normalized_g2"][...]
        q = f["xpcs/qmap/dynamic_v_list_dim0"][...]
        phi = f["xpcs/qmap/dynamic_v_list_dim1"][...]

    run_name = os.path.basename(h5_file).split("_")[0]


    scattering_2d_reshape = scattering_2d[0, :, :]
    individual_mask_intensity = []

    print('q:', q)

    print('phi:', phi)

    for i in range(9):
        print(q[i + 1] - q[i])

    for j in range(29):
        print(phi[j + 1] - phi[j])

def integrated_intensities_inspector(filename):

    with h5py.File(filename, "r") as f:
        # dynamic_roi_map = f["xpcs/qmap/dynamic_roi_map"][...]
        # scattering_2d = f["xpcs/temporal_mean/scattering_2d"][...]
        # ttc = f["xpcs/twotime/correlation_map/c2_00194"][...]
        # g2 = f["xpcs/twotime/normalized_g2"][...]
        # q = f["xpcs/qmap/dynamic_v_list_dim0"][...]
        # phi = f["xpcs/qmap/dynamic_v_list_dim1"][...]
        integrated_intensities = f["xpcs/temporal_mean/scattering_1d"][...]
        integrated_intensities_segments = f["xpcs/temporal_mean/scattering_1d_segments"][...]
        q = f["xpcs/qmap/static_v_list_dim0"][...]
        phi = f["xpcs/qmap/static_v_list_dim1"][...]


        print(np.shape(integrated_intensities))
        print(np.shape(integrated_intensities_segments))
        print(integrated_intensities)

        np.savetxt("integrated_intensities.txt", integrated_intensities)
        np.savetxt("integrated_intensities_segments.txt", integrated_intensities_segments)

        print("q:", np.shape(q), q)
        print("phi:", np.shape(phi), phi)

        plt.figure()
        plt.plot(integrated_intensities[0])

        plt.figure()
        plt.plot(integrated_intensities_segments[0])
        plt.plot(integrated_intensities_segments[1])
        plt.plot(integrated_intensities_segments[2])


        # plt.ylim([0, 1])
        plt.show()

def integrated_intensities_plot(h5_file: str | Path):
    """
    Plots scattering_1d (mean) and scattering_1d_segments (10 time segments).

    Assumes flattening order where phi is the fast axis:
        flat index = iq * nphi + iphi
    so that reshape -> (nq, nphi).
    """
    h5_file = str(h5_file)

    import numpy as np
    import h5py
    import matplotlib.pyplot as plt

    with h5py.File(h5_file, "r") as f:
        I1d = f["xpcs/temporal_mean/scattering_1d"][...]
        Iseg = f["xpcs/temporal_mean/scattering_1d_segments"][...]
        q = f["xpcs/qmap/static_v_list_dim0"][...]
        phi = f["xpcs/qmap/static_v_list_dim1"][...]

    # ---- basic sanity ----
    I1d = np.asarray(I1d)
    Iseg = np.asarray(Iseg)
    q = np.asarray(q)
    phi = np.asarray(phi)

    if I1d.ndim != 2 or I1d.shape[0] != 1:
        raise ValueError(f"Expected scattering_1d shape (1, 3600), got {I1d.shape}")
    if Iseg.ndim != 2 or Iseg.shape[1] != I1d.shape[1]:
        raise ValueError(f"Expected scattering_1d_segments shape (10, 3600), got {Iseg.shape}")

    nq = int(q.size)
    nphi = int(phi.size)
    if nq * nphi != int(I1d.shape[1]):
        raise ValueError(
            f"q.size * phi.size = {nq}*{nphi}={nq*nphi} does not match scattering_1d length {I1d.shape[1]}"
        )

    # ---- reshape to (q, phi) ----
    I_mean_qphi = I1d[0].reshape(nq, nphi)
    I_seg_qphi = Iseg.reshape(Iseg.shape[0], nq, nphi)  # (nseg, nq, nphi)

    # Choose a representative phi index: closest to 0 degrees
    iphi0 = int(np.argmin(np.abs(phi - 0.0)))

    # ---- derived summaries ----
    # phi-averaged intensity vs q for each segment
    Iseg_q = I_seg_qphi.mean(axis=2)          # (nseg, nq)
    Imean_q = I_mean_qphi.mean(axis=1)        # (nq,)

    # variability over time segments at each (q,phi)
    I_std_qphi = I_seg_qphi.std(axis=0)       # (nq, nphi)
    I_relstd_qphi = I_std_qphi / np.maximum(I_mean_qphi, 1e-12)

    # ---- plotting ----
    fig = plt.figure(figsize=(14.5, 9.5))
    gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.28)

    # (A) Mean map in (q, phi)
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(
        I_mean_qphi,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[phi.min(), phi.max(), q.min(), q.max()],
    )
    ax0.set_title("Mean scattering_1d reshaped to (q, φ)")
    ax0.set_xlabel("φ (deg)")
    ax0.set_ylabel("q (Å$^{-1}$)")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.03, label="Intensity (a.u.)")

    # (B) Segment evolution as (segment index, q) for φ≈0 slice
    ax1 = fig.add_subplot(gs[0, 1])
    seg_vs_q_phi0 = I_seg_qphi[:, :, iphi0]  # (nseg, nq)
    im1 = ax1.imshow(
        seg_vs_q_phi0,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[q.min(), q.max(), 0, seg_vs_q_phi0.shape[0] - 1],
    )
    ax1.set_title(f"Segments vs q at φ≈{phi[iphi0]:.3f}° (closest to 0°)")
    ax1.set_xlabel("q (Å$^{-1}$)")
    ax1.set_ylabel("segment index")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03, label="Intensity (a.u.)")

    # (C) Line plot: φ-averaged intensity vs q for each segment + mean
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(q, Imean_q, lw=2.5, label="mean (φ-avg)")
    for s in range(Iseg_q.shape[0]):
        ax2.plot(q, Iseg_q[s], lw=1.2, alpha=0.8, label=f"seg {s}" if s < 4 else None)
    ax2.set_title("φ-averaged intensity vs q (all segments)")
    ax2.set_xlabel("q (Å$^{-1}$)")
    ax2.set_ylabel("Intensity (a.u.)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best", fontsize=9)

    # (D) Relative variability map (std/mean) in (q, phi)
    ax3 = fig.add_subplot(gs[1, 1])
    im3 = ax3.imshow(
        I_relstd_qphi,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[phi.min(), phi.max(), q.min(), q.max()],
    )
    ax3.set_title("Temporal variability: std(seg)/mean  (q, φ)")
    ax3.set_xlabel("φ (deg)")
    ax3.set_ylabel("q (Å$^{-1}$)")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.03, label="Relative std")

    fig.suptitle(f"Integrated intensity diagnostics\n{h5_file}", y=0.98, fontsize=12)
    plt.show()


def find_bragg_peak_center_from_scattering_2d_with_overlay_function(
    h5_file: str | Path,
    *,
    dataset_key: str = "xpcs/temporal_mean/scattering_2d",
    use_first_frame_if_3d: bool = True,
    smooth_sigma_px: float | None = 1.0,
    bright_percentile: float = 99.7,
    weight_mode: str = "log",  # "linear" | "sqrt" | "log"
    figsize: tuple[float, float] = (7.2, 6.4),
) -> tuple[tuple[float, float], dict]:
    """
    Option 1 (robust bright-region centroid) + ALWAYS makes an overlay plot.

    Steps
    -----
    1) Load scattering_2d (use first frame if it's (1,H,W))
    2) Light smoothing (optional)
    3) Threshold at bright_percentile to define a "bright region"
    4) Compute weighted centroid of that region
    5) Plot: scattering_2d with mask outline + centroid marker

    Returns
    -------
    (cy, cx) : (float, float)
        Estimated centre (row, col) in pixel coordinates.
    info : dict
        Debug info: threshold, n_pixels_used, etc.
    """
    h5_file = Path(h5_file)

    with h5py.File(h5_file, "r") as f:
        scat = f[dataset_key][...]

    scat = np.asarray(scat)
    if scat.ndim == 3 and use_first_frame_if_3d:
        scat = scat[0, :, :]
    if scat.ndim != 2:
        raise ValueError(f"{dataset_key} must be 2D (or 3D with first-frame), got {scat.shape}")

    img = scat.astype(np.float64, copy=False)

    # Optional smoothing (very light) to stabilize centroid under hot pixels
    if smooth_sigma_px is not None and smooth_sigma_px > 0:
        try:
            from scipy.ndimage import gaussian_filter
            img_s = gaussian_filter(img, sigma=float(smooth_sigma_px))
        except Exception:
            img_s = img
    else:
        img_s = img

    # Bright-region mask
    p = float(np.clip(bright_percentile, 0.0, 100.0))
    thr = float(np.nanpercentile(img_s, p))
    mask = np.isfinite(img_s) & (img_s >= thr)
    n = int(mask.sum())

    # If too few pixels, relax threshold slightly
    if n < 10:
        p2 = max(90.0, p - 5.0)
        thr = float(np.nanpercentile(img_s, p2))
        mask = np.isfinite(img_s) & (img_s >= thr)
        n = int(mask.sum())

    if n < 10:
        raise ValueError(
            f"Bright-region mask too small (n={n}). "
            f"Lower bright_percentile (currently {bright_percentile})."
        )

    yy, xx = np.nonzero(mask)
    vals = img_s[yy, xx]

    # Weights (to reduce dominance of extreme skew / hot pixels)
    if weight_mode == "linear":
        w = np.clip(vals, 0.0, np.inf)
    elif weight_mode == "sqrt":
        w = np.sqrt(np.clip(vals, 0.0, np.inf))
    elif weight_mode == "log":
        w = np.log1p(np.clip(vals, 0.0, np.inf))
    else:
        raise ValueError("weight_mode must be one of: 'linear', 'sqrt', 'log'")

    wsum = float(np.sum(w))
    if not np.isfinite(wsum) or wsum <= 0:
        raise ValueError("Non-positive or non-finite weight sum, cannot compute centroid")

    cy = float(np.sum(yy * w) / wsum)
    cx = float(np.sum(xx * w) / wsum)

    info = {
        "h5_file": str(h5_file),
        "dataset_key": dataset_key,
        "img_shape": tuple(img.shape),
        "smooth_sigma_px": smooth_sigma_px,
        "bright_percentile": float(bright_percentile),
        "threshold_value": float(thr),
        "n_pixels_used": n,
        "weight_mode": weight_mode,
        "centroid_cy_cx": (cy, cx),
    }


    # Overlay plot (always)

    # Use log1p for display to handle heavy skew safely (works with zeros)
    disp = np.log1p(np.clip(img, 0.0, np.inf))

    # Robust display limits
    vmin = float(np.nanpercentile(disp, 1.0))
    vmax = float(np.nanpercentile(disp, 99.8))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = None, None

    img = np.asarray(disp, dtype=np.float64)
    img = np.where(img > 0, img, np.nan)

    vmin = np.nanpercentile(img, 1.0)
    vmax = np.nanpercentile(img, 99.999)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(
        img,
        origin="upper",
        cmap="magma",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )

    # Mask outline
    ax.contour(mask.astype(np.float32), levels=[0.5], linewidths=1.2, colors="cyan")

    # Centroid marker
    ax.plot(cx, cy, marker="x", markersize=10, mew=2.2)

    ax.set_title(
        f"Bragg peak centre (bright-region centroid)\n"
        f"p={bright_percentile:.2f}, n={n}, weight={weight_mode}, σ={smooth_sigma_px}"
    )
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("log1p(scattering_2d)")

    plt.tight_layout()
    plt.show()

    return (cy, cx), info

def execute_find_bragg_peak_center_from_scattering_2d_with_overlay():

    (cy, cx), info = find_bragg_peak_center_from_scattering_2d_with_overlay_function(
        filename,
        bright_percentile=99.7,
        smooth_sigma_px=1.0,
        weight_mode="log",
    )

    print(f"Bragg peak centre: cy={cy:.2f}, cx={cx:.2f}")
    print(info)

def _ensure_2d(img):
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    if img.ndim != 2:
        raise ValueError(f"Expected 2D (or (1,H,W)) scattering_2d, got shape {img.shape}")
    return img


def _match_axes(img2d, q_vals, phi_vals):
    """
    Returns (img, q, phi) such that:
      img.shape == (len(phi), len(q))  i.e. axis0=phi, axis1=q
    If user provided transposed vectors, we transpose img.
    """
    q_vals = np.asarray(q_vals).ravel()
    phi_vals = np.asarray(phi_vals).ravel()

    H, W = img2d.shape
    if (H, W) == (phi_vals.size, q_vals.size):
        return img2d, q_vals, phi_vals
    if (H, W) == (q_vals.size, phi_vals.size):
        return img2d.T, q_vals, phi_vals

    raise ValueError(
        f"Shape mismatch: img={img2d.shape}, len(phi)={phi_vals.size}, len(q)={q_vals.size}. "
        "Expected img=(len(phi),len(q)) or transposed."
    )


def _winsorize(x, p_hi=99.9):
    x = np.asarray(x, dtype=np.float64)
    hi = np.nanpercentile(x, float(p_hi))
    lo = np.nanpercentile(x, 0.0)
    return np.clip(x, lo, hi), float(lo), float(hi)


def _local_median_3x3(img):
    """
    Fast 3x3 local median using shifted stacks (pure numpy, no scipy).
    Edge-handled by padding with edge values.
    """
    a = np.asarray(img, dtype=np.float64)
    p = np.pad(a, ((1, 1), (1, 1)), mode="edge")
    shifts = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifts.append(p[1 + dy : 1 + dy + a.shape[0], 1 + dx : 1 + dx + a.shape[1]])
    stack = np.stack(shifts, axis=0)  # (9,H,W)
    return np.median(stack, axis=0)


def _despike_hot_pixels(img, *, z_thresh=12.0, use_log=True):
    """
    Replace extreme outliers using a robust z-score on (optionally) log1p(img).
    Replacement value is local 3x3 median.
    """
    x = np.asarray(img, dtype=np.float64)
    x0 = np.clip(x, 0.0, np.inf)

    y = np.log1p(x0) if use_log else x0
    med = np.nanmedian(y)
    mad = np.nanmedian(np.abs(y - med))
    if not np.isfinite(mad) or mad == 0:
        return x0, np.zeros_like(x0, dtype=bool)

    # 1.4826 * MAD ~ sigma for normal, good robust scale
    z = (y - med) / (1.4826 * mad)
    hot = z > float(z_thresh)

    if np.any(hot):
        local_med = _local_median_3x3(x0)
        x0 = x0.copy()
        x0[hot] = local_med[hot]

    return x0, hot


def _weighted_quantile(x, w, qs):
    """
    Weighted quantile(s) of x with weights w.
    qs in [0,1] list/array.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    qs = np.asarray(qs, dtype=np.float64).ravel()

    m = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    x = x[m]
    w = w[m]
    if x.size == 0 or np.sum(w) <= 0:
        return np.full(qs.shape, np.nan, dtype=np.float64)

    idx = np.argsort(x)
    x = x[idx]
    w = w[idx]
    cdf = np.cumsum(w)
    cdf /= cdf[-1]

    return np.interp(qs, cdf, x)


def _weighted_moments_1d(x, w):
    """
    Returns (mu, sigma, skew) for weighted distribution.
    skew = E[(x-mu)^3] / sigma^3
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()

    m = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    x = x[m]
    w = w[m]
    W = np.sum(w)
    if x.size == 0 or W <= 0:
        return np.nan, np.nan, np.nan

    mu = np.sum(w * x) / W
    m2 = np.sum(w * (x - mu) ** 2) / W
    sigma = np.sqrt(max(m2, 0.0))
    if not np.isfinite(sigma) or sigma == 0:
        return float(mu), float(sigma), np.nan

    m3 = np.sum(w * (x - mu) ** 3) / W
    skew = m3 / (sigma ** 3)
    return float(mu), float(sigma), float(skew)


def bragg_peak_shape_metrics_fixed_q_phi(
    scattering_2d,
    *,
    q_vals,
    phi_vals,
    winsor_p_hi=99.9,
    use_log_weights=True,
    despike=True,
    hot_z_thresh=12.0,
):
    """
    Compute fixed-axis (q and phi) statistics for the Bragg peak shape.

    Returns a dict containing:
      - center_q, center_phi (weighted means)
      - sigma_q, sigma_phi  (weighted std dev)
      - skew_q,  skew_phi   (weighted moment skewness)
      - q_profile, phi_profile (marginals)
      - quantile skew (Bowley) in each axis
      - optional hot-pixel mask + thresholds

    Notes
    -----
    - Tails are included (no masking), but hot pixels are optionally despiked.
    - Robustness is controlled by winsor_p_hi and/or use_log_weights.
    - Axis convention enforced: img shape = (len(phi), len(q)).
    """
    img = _ensure_2d(scattering_2d)
    img, q, phi = _match_axes(img, q_vals, phi_vals)

    x = np.asarray(img, dtype=np.float64)
    x = np.clip(x, 0.0, np.inf)

    hot_mask = None
    if despike:
        x, hot_mask = _despike_hot_pixels(x, z_thresh=float(hot_z_thresh), use_log=True)

    # Winsor cap to prevent a few extreme pixels dominating moments
    xw, lo, hi = _winsorize(x, p_hi=float(winsor_p_hi))

    # Weights (optionally log-compressed)
    w = np.log1p(xw) if bool(use_log_weights) else xw
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.clip(w, 0.0, np.inf)

    # Marginals: axis0=phi, axis1=q
    Wq = np.sum(w, axis=0)   # (nq,)
    Wphi = np.sum(w, axis=1) # (nphi,)

    # Fixed-axis moments
    mu_q, sig_q, skew_q = _weighted_moments_1d(q, Wq)
    mu_phi, sig_phi, skew_phi = _weighted_moments_1d(phi, Wphi)

    # Quantile (Bowley) skewness for stability check
    q25, q50, q75 = _weighted_quantile(q, Wq, [0.25, 0.50, 0.75])
    p25, p50, p75 = _weighted_quantile(phi, Wphi, [0.25, 0.50, 0.75])

    def _bowley(a25, a50, a75):
        den = (a75 - a25)
        if not np.isfinite(den) or den == 0:
            return np.nan
        return float((a75 + a25 - 2.0 * a50) / den)

    bowley_q = _bowley(q25, q50, q75)
    bowley_phi = _bowley(p25, p50, p75)

    # Effective number of pixels (for “how concentrated are the weights”)
    wflat = w.ravel()
    sw = np.sum(wflat)
    sw2 = np.sum(wflat * wflat)
    neff = (sw * sw / sw2) if (sw2 > 0) else np.nan

    return {
        "center_q": mu_q,
        "center_phi": mu_phi,
        "sigma_q": sig_q,
        "sigma_phi": sig_phi,
        "skew_q": skew_q,
        "skew_phi": skew_phi,
        "bowley_skew_q": bowley_q,
        "bowley_skew_phi": bowley_phi,
        "q_profile": Wq,
        "phi_profile": Wphi,
        "q_vals": q,
        "phi_vals": phi,
        "winsor_lo": lo,
        "winsor_hi": hi,
        "use_log_weights": bool(use_log_weights),
        "despike": bool(despike),
        "hot_pixel_mask": hot_mask,
        "neff": float(neff) if np.isfinite(neff) else np.nan,
    }

def plot_bragg_peak_shape_metrics_overlay_from_maps(
    scattering_2d: np.ndarray,
    *,
    q_map: np.ndarray,
    phi_map: np.ndarray,
    metrics: dict,
    valid_mask: np.ndarray | None = None,
    cmap: str = "magma",
    n_q_contours: int = 7,
    n_phi_contours: int = 7,
):
    """
    Overlay iso-q and iso-phi contours on detector-space scattering_2d,
    plus markers for (q_mean, phi_mean) projected back to pixels.

    This is the detector-space counterpart to the old binned overlay.
    """
    img = np.asarray(scattering_2d)
    if img.ndim == 3:
        img = img[0]
    img = np.asarray(img, dtype=np.float64)

    q_map = np.asarray(q_map, dtype=np.float64)
    phi_map = np.asarray(phi_map, dtype=np.float64)

    if img.shape != q_map.shape or img.shape != phi_map.shape:
        raise ValueError(f"Shape mismatch: img={img.shape}, q_map={q_map.shape}, phi_map={phi_map.shape}")

    if valid_mask is None:
        valid = np.isfinite(img) & np.isfinite(q_map) & np.isfinite(phi_map)
    else:
        valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(img) & np.isfinite(q_map) & np.isfinite(phi_map)

    if not np.any(valid):
        raise RuntimeError("No valid pixels for overlay.")

    # log-ish display without blowing out
    disp = np.log10(np.clip(img, 0.0, None) + 1e-12)
    img_pos = img[img > 0]

    vmin = np.percentile(img_pos, 1.0)
    vmax = np.percentile(img_pos, 99.9)

    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(1, 1, figsize=(7.4, 6.2))
    im = ax.imshow(
        img,
        origin="lower",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )

    ax.set_title("Detector-space scattering_2d with q/phi contour overlay")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_facecolor("black")

    # Contour levels chosen from valid pixels
    qv = q_map[valid]
    phv = phi_map[valid]

    q_lo, q_hi = np.nanpercentile(qv, [2, 98])
    ph_lo, ph_hi = np.nanpercentile(phv, [2, 98])

    q_levels = np.linspace(q_lo, q_hi, int(n_q_contours))
    ph_levels = np.linspace(ph_lo, ph_hi, int(n_phi_contours))

    # Mask invalid pixels for contouring
    q_for_contour = np.where(valid, q_map, np.nan)
    ph_for_contour = np.where(valid, phi_map, np.nan)

    ax.contour(q_for_contour, levels=q_levels, linewidths=0.7, alpha=0.8)
    ax.contour(ph_for_contour, levels=ph_levels, linewidths=0.7, alpha=0.5)

    # Mark the mean (q_mean, phi_mean) by finding nearest pixel in (q,phi) space
    q_mean = float(metrics.get("q_mean"))
    phi_mean = float(metrics.get("phi_mean_rad"))

    dphi = phv - phi_mean
    dphi = (dphi + np.pi) % (2.0 * np.pi) - np.pi
    dq = qv - q_mean
    dist2 = dq * dq + dphi * dphi

    # Find pixel index of min dist2
    flat_valid_idx = np.flatnonzero(valid)
    best_flat = flat_valid_idx[int(np.argmin(dist2))]
    cy, cx = np.unravel_index(best_flat, img.shape)

    ax.plot([cx], [cy], marker="x", markersize=10, mew=2)
    ax.text(cx + 10, cy + 10, f"(q̄, φ̄)\nq={q_mean:.4f}\nφ={phi_mean:.4f} rad", fontsize=9, color="blue")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="log10(Intensity + eps)")
    fig.tight_layout()
    plt.show()

    return {"center_px_rc": (float(cy), float(cx))}

def plot_bragg_peak_shape_metrics_overlay(
    scattering_2d,
    *,
    q_vals,
    phi_vals,
    metrics: dict,
    title: str = "Bragg peak shape metrics (fixed q/phi axes)",
    cmap: str = "magma",
):
    """
    Standard overlay plot:
      - 2D image (log display) with (center_q, center_phi) marker
      - marginal profiles Wq(q) and Wphi(phi)
    """
    img = _ensure_2d(scattering_2d)
    img, q, phi = _match_axes(img, q_vals, phi_vals)

    # Display image as log1p for visibility without thresholding
    disp = np.log1p(np.clip(img, 0.0, np.inf))

    fig = plt.figure(figsize=(12.5, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(
        disp,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        extent=[q[0], q[-1], phi[0], phi[-1]],
    )
    ax0.plot([metrics["center_q"]], [metrics["center_phi"]], marker="x", markersize=10, mew=2)
    ax0.set_xlabel("q")
    ax0.set_ylabel("phi")
    ax0.set_title("log1p(scattering_2d) + center")
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.03)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(q, metrics["q_profile"], lw=1.8)
    ax1.set_xlabel("q")
    ax1.set_title(
        f"Wq(q)\n"
        f"μ={metrics['center_q']:.6g}, σ={metrics['sigma_q']:.4g}, "
        f"skew={metrics['skew_q']:.3g}"
    )
    ax1.grid(True, alpha=0.25)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(phi, metrics["phi_profile"], lw=1.8)
    ax2.set_xlabel("phi")
    ax2.set_title(
        f"Wφ(φ)\n"
        f"μ={metrics['center_phi']:.6g}, σ={metrics['sigma_phi']:.4g}, "
        f"skew={metrics['skew_phi']:.3g}"
    )
    ax2.grid(True, alpha=0.25)

    fig.suptitle(
        f"{title}\n"
        f"Bowley skew: q={metrics['bowley_skew_q']:.3g}, phi={metrics['bowley_skew_phi']:.3g} | "
        f"Neff={metrics['neff']:.1f}",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    plt.show()
    return fig

def bragg_peak_shape_metrics_fixed_q_phi_from_maps(
    scattering_2d: np.ndarray,
    *,
    q_map: np.ndarray,
    phi_map: np.ndarray,
    valid_mask: np.ndarray | None = None,
    hot_z_thresh: float = 12.0,
    eps: float = 1e-12,
) -> dict:
    """
    Compute Bragg-peak shape metrics with q and phi defined per DETECTOR PIXEL.

    Inputs
    ------
    scattering_2d : (H,W) or (1,H,W)
        Detector-space average image (NOT 120x30 binned).
    q_map, phi_map : (H,W)
        Per-pixel q and phi maps (e.g. from inferred_qphi_maps.npz).
    valid_mask : (H,W) bool or None
        Optional mask of valid pixels. If None, inferred from finiteness.
    hot_z_thresh : float
        Suppress extreme hot pixels using robust z-score on intensity.
        This only rejects extreme outliers, it does NOT mask long physical tails.

    Returns
    -------
    dict of scalar metrics (means, sigmas, skewness in q and phi).
    """
    img = np.asarray(scattering_2d)
    if img.ndim == 3:
        if img.shape[0] != 1:
            raise ValueError(f"Expected scattering_2d with leading dim 1, got {img.shape}")
        img = img[0]
    img = np.asarray(img, dtype=np.float64)

    q_map = np.asarray(q_map, dtype=np.float64)
    phi_map = np.asarray(phi_map, dtype=np.float64)

    if img.shape != q_map.shape or img.shape != phi_map.shape:
        raise ValueError(
            f"Shape mismatch: img={img.shape}, q_map={q_map.shape}, phi_map={phi_map.shape}"
        )

    if valid_mask is None:
        valid = np.isfinite(img) & np.isfinite(q_map) & np.isfinite(phi_map)
    else:
        valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(img) & np.isfinite(q_map) & np.isfinite(phi_map)

    # ---- hot pixel suppression (robust z-score on intensity)
    # This is a minimal, conservative rejection of extreme spikes.
    vv = valid
    if np.any(vv):
        med = np.nanmedian(img[vv])
        mad = np.nanmedian(np.abs(img[vv] - med))
        sigma_rob = 1.4826 * mad + eps
        z = (img - med) / sigma_rob
        vv = vv & (z < float(hot_z_thresh))

    if not np.any(vv):
        raise RuntimeError("No valid pixels after masking / hot-pixel suppression.")

    # freeze the mask used for pixel-space moments
    vv_used = vv.copy()

    w = img[vv_used].copy()

    # Weights must be non-negative for moment interpretation
    # (if you have negative values from processing, clip them softly)
    w = np.clip(w, 0.0, None)

    iy, ix = np.nonzero(vv_used)

    sw = float(np.sum(w))
    if not np.isfinite(sw) or sw <= eps:
        raise RuntimeError("Sum of weights is zero or non-finite, cannot compute moments.")

    qv = q_map[vv]
    ph = phi_map[vv]

    # ---- weighted mean in q
    q_mean = float(np.sum(w * qv) / sw)

    # ---- weighted circular mean in phi
    c = float(np.sum(w * np.cos(ph)) / sw)
    s = float(np.sum(w * np.sin(ph)) / sw)
    phi_mean = float(np.arctan2(s, c))

    # unwrap phi about mean to compute *directional* moments
    dphi = ph - phi_mean
    dphi = (dphi + np.pi) % (2.0 * np.pi) - np.pi  # wrap to [-pi, pi]

    dq = qv - q_mean

    # ---- weighted central moments and skewness
    mu2_q = float(np.sum(w * dq * dq) / sw)
    mu3_q = float(np.sum(w * dq * dq * dq) / sw)

    mu2_phi = float(np.sum(w * dphi * dphi) / sw)
    mu3_phi = float(np.sum(w * dphi * dphi * dphi) / sw)

    sigma_q = float(np.sqrt(max(mu2_q, 0.0)))
    sigma_phi = float(np.sqrt(max(mu2_phi, 0.0)))

    skew_q = float(mu3_q / (mu2_q ** 1.5 + eps))
    skew_phi = float(mu3_phi / (mu2_phi ** 1.5 + eps))

    # Optional extra diagnostics that are often useful
    peak_adu = float(np.nanmax(img[vv]))
    n_pix = int(np.sum(vv))
    frac_kept = float(n_pix / np.sum(valid)) if np.sum(valid) > 0 else float("nan")

    # ----------------------------
    # Pixel-space (x, y) metrics
    # ----------------------------
    # iy, ix = np.nonzero(valid_mask)

    w_sum = np.sum(w)

    # centroids in pixel coordinates
    x_mean = np.sum(w * ix) / w_sum
    y_mean = np.sum(w * iy) / w_sum

    # second moments
    sigma_x = np.sqrt(np.sum(w * (ix - x_mean) ** 2) / w_sum)
    sigma_y = np.sqrt(np.sum(w * (iy - y_mean) ** 2) / w_sum)

    # skewness
    skew_x = np.sum(w * (ix - x_mean) ** 3) / (w_sum * sigma_x ** 3)
    skew_y = np.sum(w * (iy - y_mean) ** 3) / (w_sum * sigma_y ** 3)

    return {
        "x_mean_px": float(x_mean),
        "y_mean_px": float(y_mean),
        "sigma_x_px": float(sigma_x),
        "sigma_y_px": float(sigma_y),
        "skew_x": float(skew_x),
        "skew_y": float(skew_y),
        "q_mean": q_mean,
        "phi_mean_rad": phi_mean,
        "sigma_q": sigma_q,
        "sigma_phi_rad": sigma_phi,
        "skew_q": skew_q,
        "skew_phi": skew_phi,
        "n_pix_used": n_pix,
        "frac_valid_used": frac_kept,
        "peak_intensity": peak_adu,
        "hot_z_thresh": float(hot_z_thresh),
    }

def exec_bragg_peak_shape_metrics_fixed_q_phi():

    with h5py.File(filename, "r") as f:
        scattering_2d = f["xpcs/temporal_mean/scattering_2d"][...]
        q = f["xpcs/qmap/static_v_list_dim0"][...]
        phi = f["xpcs/qmap/static_v_list_dim1"][...]

    # Example: point this to the file you just saved
    npz_path = Path("/Users/emilioescauriza/Desktop/Twotime_PostExpt_01/A073_inferred_qphi_maps.npz")
    d = np.load(npz_path, allow_pickle=False)

    metrics = bragg_peak_shape_metrics_fixed_q_phi_from_maps(
        scattering_2d,
        q_map=d["q_map"],
        phi_map=d["phi_map"],
        valid_mask=d.get("valid_mask", None),
        hot_z_thresh=12.0,
    )

    print("Bragg peak shape metrics (fixed q/phi via per-pixel maps):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    plot_bragg_peak_shape_metrics_overlay_from_maps(
        scattering_2d,
        q_map=d["q_map"],
        phi_map=d["phi_map"],
        valid_mask=d.get("valid_mask", None),
        metrics=metrics,
        cmap="magma",
    )

    # Print the key numbers (optional)
    print(
        "Center (q, phi) = "
        f"({metrics['center_q']:.6g}, {metrics['center_phi']:.6g})\n"
        "Sigma  (q, phi) = "
        f"({metrics['sigma_q']:.6g}, {metrics['sigma_phi']:.6g})\n"
        "Skew   (q, phi) = "
        f"({metrics['skew_q']:.6g}, {metrics['skew_phi']:.6g})\n"
        "Bowley (q, phi) = "
        f"({metrics['bowley_skew_q']:.6g}, {metrics['bowley_skew_phi']:.6g})\n"
        f"Neff = {metrics['neff']:.2f}"
    )

    return metrics


def infer_q_phi_maps_from_static_qmap(
    f,
    *,
    q_key: str = "xpcs/qmap/static_v_list_dim0",
    phi_key: str = "xpcs/qmap/static_v_list_dim1",
    roi_map_key: str = "xpcs/qmap/static_roi_map",
    index_map_key: str = "xpcs/qmap/static_index_mapping",
    num_pts_key: str = "xpcs/qmap/static_num_pts",
    invalid_roi_value: int = 0,
):
    """
    Build per-pixel q_map / phi_map (and pseudo-Qx/Qy) from the static qmap products.

    Requires:
      - static_roi_map: per-pixel ROI id
      - static_index_mapping: length Nq*Nphi, maps (iq,iphi) bin -> ROI id
      - static_v_list_dim0: q bin centers (Nq)
      - static_v_list_dim1: phi bin centers (Nphi)

    Returns
    -------
    out : dict with keys
      q_map, phi_map, Qx_map, Qy_map, valid_mask, roi_map, q_vals, phi_vals

    Notes
    -----
    - phi may be in degrees or radians; we infer by range and convert to radians for cos/sin.
    - invalid_roi_value (often 0) is treated as background/invalid.
    """
    q_vals = np.asarray(f[q_key][...], dtype=np.float64)          # (Nq,)
    phi_vals = np.asarray(f[phi_key][...], dtype=np.float64)      # (Nphi,)
    roi_map = np.asarray(f[roi_map_key][...], dtype=np.int64)     # (ny,nx)

    # index_mapping: length Nq*Nphi, values are ROI ids (uint16)
    idx_to_roi = np.asarray(f[index_map_key][...], dtype=np.int64)

    # sanity check Nq,Nphi from file (optional but helpful)
    if num_pts_key in f:
        npts = np.asarray(f[num_pts_key][...], dtype=np.int64).ravel()
        if npts.size == 2:
            Nq_file, Nphi_file = int(npts[0]), int(npts[1])
            if Nq_file != q_vals.size or Nphi_file != phi_vals.size:
                raise ValueError(
                    f"static_num_pts says (Nq,Nphi)=({Nq_file},{Nphi_file}) "
                    f"but v_list sizes are (Nq,Nphi)=({q_vals.size},{phi_vals.size})"
                )

    Nq = int(q_vals.size)
    Nphi = int(phi_vals.size)
    if idx_to_roi.size != Nq * Nphi:
        raise ValueError(
            f"static_index_mapping has length {idx_to_roi.size}, expected Nq*Nphi={Nq*Nphi}"
        )

    # infer phi units -> radians
    # (if values look like degrees, convert)
    phi_max = float(np.nanmax(phi_vals))
    phi_min = float(np.nanmin(phi_vals))
    # crude but reliable for your typical bin-centers
    if (phi_max - phi_min) > (2.0 * np.pi + 0.5):
        phi_rad = np.deg2rad(phi_vals)
    else:
        phi_rad = phi_vals.copy()

    # Build ROI->(q,phi) lookup.
    # idx = iq*Nphi + iphi  (iphi changes fastest, matches your "repeats every 30")
    # idx_to_roi[idx] gives ROI id
    roi_max = int(np.max(roi_map))
    roi_to_q = np.full((roi_max + 1,), np.nan, dtype=np.float64)
    roi_to_phi = np.full((roi_max + 1,), np.nan, dtype=np.float64)

    for iq in range(Nq):
        for iphi in range(Nphi):
            lin = iq * Nphi + iphi
            roi_id = int(idx_to_roi[lin])
            if roi_id <= 0 or roi_id > roi_max:
                continue
            # assign (q,phi) for that ROI id
            # if duplicates exist, they should match; if not, last wins
            roi_to_q[roi_id] = float(q_vals[iq])
            roi_to_phi[roi_id] = float(phi_rad[iphi])

    # Now map per-pixel
    valid_mask = (roi_map != int(invalid_roi_value)) & (roi_map >= 0) & (roi_map <= roi_max)

    q_map = np.full_like(roi_map, np.nan, dtype=np.float64)
    phi_map = np.full_like(roi_map, np.nan, dtype=np.float64)

    q_map[valid_mask] = roi_to_q[roi_map[valid_mask]]
    phi_map[valid_mask] = roi_to_phi[roi_map[valid_mask]]

    # pseudo-components
    Qx_map = np.full_like(q_map, np.nan, dtype=np.float64)
    Qy_map = np.full_like(q_map, np.nan, dtype=np.float64)

    vv = valid_mask & np.isfinite(q_map) & np.isfinite(phi_map)
    Qx_map[vv] = q_map[vv] * np.cos(phi_map[vv])
    Qy_map[vv] = q_map[vv] * np.sin(phi_map[vv])

    return {
        "q_map": q_map,
        "phi_map": phi_map,
        "Qx_map": Qx_map,
        "Qy_map": Qy_map,
        "valid_mask": vv,
        "roi_map": roi_map,
        "q_vals": q_vals,
        "phi_vals": phi_vals,
    }


def save_inferred_qphi_maps_npz(
    results_hdf_path: str | Path,
    out_npz_path: str | Path,
):
    """
    Convenience wrapper: open results .hdf, infer maps, save to .npz.
    """
    results_hdf_path = Path(results_hdf_path)
    out_npz_path = Path(out_npz_path)

    with h5py.File(results_hdf_path, "r") as f:
        # your results file stores under /xpcs/...
        maps = infer_q_phi_maps_from_static_qmap(f, q_key="xpcs/qmap/static_v_list_dim0",
                                                 phi_key="xpcs/qmap/static_v_list_dim1",
                                                 roi_map_key="xpcs/qmap/static_roi_map",
                                                 index_map_key="xpcs/qmap/static_index_mapping",
                                                 num_pts_key="xpcs/qmap/static_num_pts")

    np.savez_compressed(
        out_npz_path,
        q_map=maps["q_map"],
        phi_map=maps["phi_map"],
        Qx_map=maps["Qx_map"],
        Qy_map=maps["Qy_map"],
        valid_mask=maps["valid_mask"],
        roi_map=maps["roi_map"],
        q_vals=maps["q_vals"],
        phi_vals=maps["phi_vals"],
    )
    print(f"Saved inferred maps: {out_npz_path}")
    return out_npz_path

def exec_make_and_save_inferred_qphi_maps():
    results = Path("/Users/emilioescauriza/Desktop/Twotime_PostExpt_01/A073_IPA_NBH_1_att0100_260K_001_results.hdf")
    out = Path("/Users/emilioescauriza/Desktop/Twotime_PostExpt_01/A073_inferred_qphi_maps.npz")
    save_inferred_qphi_maps_npz(results, out)

def exec_quick_check_inferred_qphi_npz():

    npz_path = Path("/Users/emilioescauriza/Desktop/Twotime_PostExpt_01/A073_inferred_qphi_maps.npz")

    d = np.load(Path(npz_path), allow_pickle=False)

    q_map = d["q_map"]
    phi_map = d["phi_map"]
    Qx_map = d["Qx_map"]
    Qy_map = d["Qy_map"]
    valid = d["valid_mask"]

    print("Shapes:")
    print("  q_map:", q_map.shape, "phi_map:", phi_map.shape, "valid:", valid.shape)
    print("Valid fraction:", float(np.mean(valid)))

    print("Ranges on valid pixels:")
    vv = valid & np.isfinite(q_map) & np.isfinite(phi_map)
    print("  q:", float(np.nanmin(q_map[vv])), "to", float(np.nanmax(q_map[vv])))
    print("  phi(rad):", float(np.nanmin(phi_map[vv])), "to", float(np.nanmax(phi_map[vv])))
    print("  Qx:", float(np.nanmin(Qx_map[vv])), "to", float(np.nanmax(Qx_map[vv])))
    print("  Qy:", float(np.nanmin(Qy_map[vv])), "to", float(np.nanmax(Qy_map[vv])))

    # quick visual
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ax0, ax1, ax2 = axes
    im0 = ax0.imshow(q_map, origin="lower", interpolation="nearest")
    ax0.set_title("q_map")
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.03)

    im1 = ax1.imshow(phi_map, origin="lower", interpolation="nearest")
    ax1.set_title("phi_map (rad)")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)

    im2 = ax2.imshow(valid, origin="lower", interpolation="nearest")
    ax2.set_title("valid_mask")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)

    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    plt.show()



file_ID = 'A073'

if file_ID == 'A013':
    filename = (r'/Users/emilioescauriza/Desktop/A013_IPA_NBH_1_att0100_079K_001_results.hdf')
elif file_ID == 'A073':
    filename = (r'/Users/emilioescauriza/Desktop/Twotime_PostExpt_01/A073_IPA_NBH_1_att0100_260K_001_results.hdf')
else:
    base = Path("/Volumes/EmilioSD4TB/APS_08-IDEI-2025-1006/Twotime_PostExpt_01")
    filename = next(base.glob(f"{file_ID}_*_results.hdf"))
h5_file = filename


if __name__ == "__main__":

    # h5_file_inspector(h5_file)
    # g2_plotter(h5_file)
    # ttc_plotter(h5_file)
    # intensity_vs_time(h5_file)
    # static_vs_dynamic_bins(h5_file)
    # combined_plot(h5_file)
    # oauth_test()
    # image_upload()
    # figure_upload()
    # q_spacing_inspector(h5_file)
    # integrated_intensities_inspector(h5_file)
    # integrated_intensities_plot(h5_file)
    # execute_find_bragg_peak_center_from_scattering_2d_with_overlay()
    # exec_make_and_save_inferred_qphi_maps()
    # exec_quick_check_inferred_qphi_npz()
    exec_bragg_peak_shape_metrics_fixed_q_phi()

    pass