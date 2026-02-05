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

    # -------------------------
    # Overlay plot (always)
    # -------------------------
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
    execute_find_bragg_peak_center_from_scattering_2d_with_overlay()

    pass