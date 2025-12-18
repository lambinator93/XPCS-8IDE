import sys
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms.operators.binary import difference

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

    plt.figure()
    plt.imshow(np.log(scattering_2d_reshape)+1e-5)

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


file_ID = 'A089'

if file_ID == 'A013':
    filename = (r'/Users/emilioescauriza/Desktop/A013_IPA_NBH_1_att0100_079K_001_results.hdf')
else:
    base = Path("/Volumes/EmilioSD4TB/APS_08-IDEI-2025-1006")
    filename = next(base.glob(f"{file_ID}_*_results.hdf"))
h5_file = filename


if __name__ == "__main__":

    # h5_file_inspector(h5_file)
    # g2_plotter(h5_file)
    # ttc_plotter(h5_file)
    # intensity_vs_time(h5_file)
    # static_vs_dynamic_bins(h5_file)
    combined_plot(h5_file)

    pass