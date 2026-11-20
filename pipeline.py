"""Timelapse ROI pipeline (Python port of the ImageJ workflow).

Steps
-----
1. Motion correction      (ImageJ: StackReg, rigid body)      -> pystackreg
2. Background subtraction (ImageJ: Subtract Background, rolling ball)
3. Max-intensity projection
4. Draw ROIs around cell bodies in napari
5. Save ROIs
6. Measure every ROI in every frame of the processed timelapse
7. Save measurements

Usage
-----
    uv run python pipeline.py              # opens a file picker to choose the image
    uv run python pipeline.py path/to/image.ome.tif

Options: see `uv run python pipeline.py --help`.

Outputs go into `<image name>_analysis/`:
    processed.tif      registered + background-subtracted stack, uint16 (cached; reused on re-runs)
    max_projection.tif
    rois.csv           napari shapes file (reloaded automatically next time so you can edit)
    roi_labels.tif     label image, ROI n has pixel value n (open in ImageJ/Fiji to check)
    measurements.csv   one row per frame: Frame, Mean1, Mean2, ... (mean intensity per ROI)
    measurements_dFF.csv  Frame, dFF1, dFF2, ...: ΔF/F (%) = 100 * (F - F0) / F0, where F0
                       is each ROI's mean over the baseline frames (default 1-12)
    measurements_dFF_norm.csv  Frame, dFFnorm1, ...: ΔF/F scaled so each ROI's max = 100%
    traces.pdf         raw mean intensity, ΔF/F (%), and normalized ΔF/F (%) stacked vs
                       time (min); opened automatically
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from bioio import BioImage
from scipy.ndimage import uniform_filter
from skimage.measure import block_reduce
from skimage.restoration import rolling_ball
from skimage.transform import resize


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_stack(path: Path, channel: int = 0) -> tuple[np.ndarray, float | None, float | None]:
    """Return (T, Y, X) array, pixel size in µm, and frame interval in s (if known)."""
    img = BioImage(path)
    stack = img.get_image_data("TYX", C=channel, Z=0)
    pixel_um = img.physical_pixel_sizes.X
    frame_s = None
    try:  # OME metadata: average spacing between planes' DeltaT
        planes = img.metadata.images[0].pixels.planes
        dts = [p.delta_t for p in planes if p.delta_t is not None and p.the_c == channel]
        if len(dts) > 1:
            frame_s = float(np.median(np.diff(dts)))
    except Exception:
        pass
    return stack, pixel_um, frame_s


# --------------------------------------------------------------------------- #
# 1. Registration (StackReg)
# --------------------------------------------------------------------------- #
def compute_registration(stack: np.ndarray, mode: str = "rigid"):
    """Estimate the per-frame alignment, like ImageJ StackReg. Returns (StackReg, tmats).

    ImageJ StackReg aligns each slice to the previously aligned slice; pystackreg's
    reference="previous" does the same.
    """
    from pystackreg import StackReg

    modes = {
        "translation": StackReg.TRANSLATION,
        "rigid": StackReg.RIGID_BODY,
        "scaled": StackReg.SCALED_ROTATION,
        "affine": StackReg.AFFINE,
    }
    sr = StackReg(modes[mode])
    print(f"[1/7] Computing registration for {stack.shape[0]} frames ({mode})...")
    tmats = sr.register_stack(stack.astype(np.float32), reference="previous", verbose=True)
    return sr, tmats


def apply_registration(sr, tmats: np.ndarray, stack: np.ndarray) -> np.ndarray:
    """Apply the alignment. Pixels shifted in from outside the field of view become 0."""
    print("      Applying registration to background-subtracted frames...")
    return np.clip(sr.transform_stack(stack, tmats=tmats), 0, None).astype(np.float32)


# --------------------------------------------------------------------------- #
# 2. Rolling-ball background subtraction
# --------------------------------------------------------------------------- #
def _shrink_factor(radius: float) -> int:
    # Same choice ImageJ makes internally to speed up large radii.
    if radius <= 10:
        return 1
    if radius <= 30:
        return 2
    if radius <= 100:
        return 4
    return 8


def rolling_ball_background(frame: np.ndarray, radius: float, smooth: bool = True) -> np.ndarray:
    """Background estimate like ImageJ's 'Subtract Background' (light background off).

    Like ImageJ, the image is shrunk (block minimum) before rolling the ball, then the
    background is interpolated back to full size. This is ~50x faster than a full-res
    roll and gives practically the same result.
    """
    img = frame.astype(np.float32)
    if smooth:  # ImageJ default: 3x3 mean filter before computing the background
        img = uniform_filter(img, size=3)
    f = _shrink_factor(radius)
    if f > 1:
        small = block_reduce(img, (f, f), np.min)
        bg_small = rolling_ball(small, radius=radius / f)
        bg = resize(bg_small, frame.shape, order=1, preserve_range=True, anti_aliasing=False)
    else:
        bg = rolling_ball(img, radius=radius)
    # Background can never exceed the original image.
    return np.minimum(bg.astype(np.float32), frame.astype(np.float32))


def to_uint16(stack: np.ndarray) -> np.ndarray:
    """Round to the nearest whole count and convert to uint16."""
    return np.clip(np.rint(stack), 0, np.iinfo(np.uint16).max).astype(np.uint16)


def subtract_background(stack: np.ndarray, radius: float) -> np.ndarray:
    print(f"[2/7] Rolling-ball background subtraction (radius = {radius} px)...")
    out = np.empty(stack.shape, dtype=np.float32)
    for t, frame in enumerate(stack):
        out[t] = frame - rolling_ball_background(frame, radius)
        if (t + 1) % 20 == 0 or t + 1 == len(stack):
            print(f"      {t + 1}/{len(stack)}")
    return np.clip(out, 0, None)


# --------------------------------------------------------------------------- #
# 4-5. Drawing / saving ROIs in napari
# --------------------------------------------------------------------------- #
def draw_rois(processed: np.ndarray, max_proj: np.ndarray, roi_file: Path, pixel_um: float | None):
    """Open napari, let the user draw ROIs, return the Shapes layer data after the window closes."""
    import napari

    scale = (pixel_um, pixel_um) if pixel_um else (1, 1)
    viewer = napari.Viewer(title="Draw ROIs on cell bodies, then close this window")
    viewer.add_image(processed, name="processed timelapse", scale=scale, visible=False)
    viewer.add_image(max_proj, name="max projection", scale=scale, colormap="gray")

    if roi_file.exists():
        from napari_builtins.io import csv_to_layer_data

        print(f"      Loading existing ROIs from {roi_file.name} (edit or add more)")
        data, meta, _ = csv_to_layer_data(str(roi_file))
        rois = viewer.add_shapes(data, name="ROIs", ndim=2, scale=scale, **meta)
    else:
        rois = viewer.add_shapes(name="ROIs", ndim=2, scale=scale)
    rois.edge_color = "yellow"
    rois.face_color = "transparent"
    rois.edge_width = 1.5 * scale[0]
    rois.mode = "add_ellipse"

    print(
        "[4/7] napari is open.\n"
        "      - Draw ROIs on the 'ROIs' layer: ellipse (E), polygon (P), rectangle (R).\n"
        "        Polygon: click vertices, double-click / Esc to finish.\n"
        "      - Select (S) then Delete/Backspace to remove a ROI.\n"
        "      - Toggle 'processed timelapse' to check cells across frames.\n"
        "      - Close the napari window when finished."
    )
    napari.run()
    return rois


def save_rois(rois, roi_file: Path, labels_file: Path, image_shape: tuple[int, int]) -> np.ndarray:
    """Save ROIs as napari CSV and as a label image. Returns (n_rois, Y, X) boolean masks."""
    print(f"[5/7] Saving {len(rois.data)} ROIs -> {roi_file.name}, {labels_file.name}")
    from napari_builtins.io._write import napari_write_shapes

    data, meta, _ = rois.as_layer_data_tuple()
    napari_write_shapes(str(roi_file), data, meta)
    # Masks are built in data (pixel) coordinates, so scale doesn't matter here.
    masks = rois.to_masks(mask_shape=image_shape).astype(bool)
    labels = np.zeros(image_shape, dtype=np.uint16)
    for i, m in enumerate(masks, start=1):
        labels[m & (labels == 0)] = i  # earlier ROI wins where ROIs overlap
    tifffile.imwrite(labels_file, labels)
    return masks


# --------------------------------------------------------------------------- #
# 6-7. Measurement
# --------------------------------------------------------------------------- #
def measure(stack: np.ndarray, masks: np.ndarray) -> pd.DataFrame:
    """Mean intensity of each ROI in every frame (like ImageJ Multi Measure, Mean only)."""
    print(f"[6/7] Measuring {len(masks)} ROIs over {len(stack)} frames...")
    df = pd.DataFrame({"Frame": np.arange(1, len(stack) + 1)})
    for i, m in enumerate(masks, start=1):
        if not m.any():
            print(f"      warning: ROI {i} has no pixels inside the image; skipped")
            continue
        df[f"Mean{i}"] = stack[:, m].mean(axis=1)
    return df


def delta_f_over_f(df: pd.DataFrame, first: int, last: int) -> pd.DataFrame:
    """ΔF/F in percent: 100 * (F - F0) / F0 for each ROI, where F0 is its mean over
    frames first..last (inclusive). Columns: Frame, dFF1, dFF2, ..."""
    base_rows = df["Frame"].between(first, last)
    if not base_rows.any():
        raise ValueError(f"No frames in baseline window {first}-{last}")
    rois = [c for c in df.columns if c.startswith("Mean")]
    f0 = df.loc[base_rows, rois].mean()
    for col in f0[f0 <= 0].index:
        print(f"      warning: {col} baseline F0 is {f0[col]:.3g} (<= 0); its ΔF/F is meaningless")
    out = df[["Frame"]].copy()
    for col in rois:
        out[f"dFF{col[4:]}"] = 100 * (df[col] - f0[col]) / f0[col]
    return out


def normalize_to_max(df_dff: pd.DataFrame) -> pd.DataFrame:
    """Scale each ROI's ΔF/F so its maximum is 100% and 0 stays 0%.
    Columns: Frame, dFFnorm1, dFFnorm2, ..."""
    out = df_dff[["Frame"]].copy()
    for col in [c for c in df_dff.columns if c.startswith("dFF")]:
        peak = df_dff[col].max()
        if peak <= 0:
            print(f"      warning: {col} never rises above baseline (max {peak:.3g}%); its normalized values are meaningless")
        out[f"dFFnorm{col[3:]}"] = 100 * df_dff[col] / peak
    return out


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
# Fixed categorical order (colorblind-validated); ROI n always gets slot n.
ROI_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_MUTED, GRID = "#1a1a19", "#6b6a64", "#e4e3de"


def _style_axes(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK, labelsize=8)


def plot_traces(df: pd.DataFrame, df_dff: pd.DataFrame, df_norm: pd.DataFrame,
                baseline: tuple[int, int], frame_s: float | None, pdf_file: Path, title: str) -> None:
    """Save a PDF with three stacked plots for every ROI vs time (min):
    raw mean intensity, ΔF/F (%), and ΔF/F normalized to each ROI's max (%).

    Colors follow a fixed 8-color order; with more than 8 ROIs the colors repeat
    (ROI 9 = ROI 1's color, etc.) and the legend moves to the right of the plots.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if frame_s:
        to_x = lambda frames: (np.asarray(frames) - 1) * frame_s / 60  # noqa: E731
        xlabel = "Time (min)"
    else:
        print("      warning: frame interval unknown (use --frame-interval); plotting against frame number")
        to_x = lambda frames: np.asarray(frames)  # noqa: E731
        xlabel = "Frame"
    x = to_x(df["Frame"])
    rois = [c for c in df.columns if c.startswith("Mean")]
    many = len(rois) > len(ROI_COLORS)

    fig, (ax_raw, ax_dff, ax_norm) = plt.subplots(3, 1, figsize=(8.5 if many else 7, 10), sharex=True)
    for ax, data, prefix in ((ax_raw, df, "Mean"), (ax_dff, df_dff, "dFF"), (ax_norm, df_norm, "dFFnorm")):
        for i, col in enumerate(rois):
            n = col[4:]
            ax.plot(x, data[f"{prefix}{n}"], color=ROI_COLORS[i % len(ROI_COLORS)],
                    linewidth=1.0 if many else 1.5, label=f"ROI {n}")
        _style_axes(ax)
    ax_raw.set_xlim(x.min(), x.max())

    ax_raw.set_title("Raw", color=INK, fontsize=9, loc="left")
    ax_raw.set_ylabel("Mean intensity (a.u.)", color=INK)

    b0, b1 = to_x(baseline)
    for ax in (ax_dff, ax_norm):
        ax.axvspan(b0, b1, color=GRID, alpha=0.6, linewidth=0, zorder=0)
        ax.axhline(0.0, color=INK_MUTED, linewidth=0.8, linestyle="--", zorder=1)
    ax_dff.set_title(f"ΔF/F  (F₀ = mean of frames {baseline[0]}–{baseline[1]}, shaded)",
                     color=INK, fontsize=9, loc="left")
    ax_dff.set_ylabel("ΔF / F₀ (%)", color=INK)
    ax_norm.set_title("ΔF/F normalized to each ROI's maximum", color=INK, fontsize=9, loc="left")
    ax_norm.set_ylabel("ΔF/F (% of max)", color=INK)
    ax_norm.set_xlabel(xlabel, color=INK)

    if many:  # legend column(s) to the right of the top plot
        ax_raw.legend(frameon=False, fontsize=7, ncols=int(np.ceil(len(rois) / 20)),
                      loc="upper left", bbox_to_anchor=(1.01, 1.0), labelcolor=INK)
    elif len(rois) > 1:  # single row above the top plot
        ax_raw.legend(frameon=False, fontsize=8, ncols=min(len(rois), 4),
                      loc="lower right", bbox_to_anchor=(1.0, 1.0), labelcolor=INK)
    fig.suptitle(title, color=INK, fontsize=10, x=0.01, ha="left")

    fig.tight_layout()
    fig.savefig(pdf_file)
    plt.close(fig)
    print(f"      Saved plot -> {pdf_file}")


def open_file(path: Path) -> None:
    """Open a file with the system's default viewer."""
    import os
    import subprocess
    import sys

    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    elif sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


def choose_image() -> Path | None:
    """Open a file dialog to pick the timelapse. Returns None if cancelled."""
    from napari.qt import get_qapp
    from qtpy.QtWidgets import QFileDialog

    get_qapp()
    path, _ = QFileDialog.getOpenFileName(
        None, "Choose a timelapse image", str(Path.home()),
        "Images (*.tif *.tiff *.nd2);;All files (*)",
    )
    return Path(path) if path else None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", type=Path, nargs="?", default=None,
                   help="timelapse .tif / .ome.tif / .nd2 (omit to choose it in a file dialog)")
    p.add_argument("--radius", type=float, default=50, help="rolling ball radius in pixels (default 50)")
    p.add_argument("--reg-mode", default="rigid", choices=["translation", "rigid", "scaled", "affine"])
    p.add_argument("--no-registration", action="store_true", help="skip StackReg step")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--frame-interval", type=float, default=None,
                   help="seconds per frame (default: read from the file's metadata)")
    p.add_argument("--baseline", type=int, nargs=2, default=(1, 12), metavar=("FIRST", "LAST"),
                   help="frames (1-based, inclusive) averaged for F0 in ΔF/F (default: 1 12)")
    p.add_argument("--no-open", action="store_true", help="don't open the plot PDF when done")
    p.add_argument("--reprocess", action="store_true", help="ignore cached processed.tif and redo steps 1-2")
    args = p.parse_args()
    if args.image is None:
        args.image = choose_image()
        if args.image is None:
            print("No file chosen; exiting.")
            return

    outdir = args.outdir or args.image.with_name(f"{args.image.name.removesuffix('.ome.tif').removesuffix('.tif')}_analysis")
    outdir.mkdir(parents=True, exist_ok=True)
    processed_file = outdir / "processed.tif"
    maxproj_file = outdir / "max_projection.tif"
    roi_file = outdir / "rois.csv"
    labels_file = outdir / "roi_labels.tif"
    meas_file = outdir / "measurements.csv"
    dff_file = outdir / "measurements_dFF.csv"
    norm_file = outdir / "measurements_dFF_norm.csv"
    plot_file = outdir / "traces.pdf"

    raw, pixel_um, frame_s = load_stack(args.image, args.channel)
    print(f"Loaded {args.image.name}: {raw.shape} {raw.dtype}, pixel = {pixel_um} µm, frame interval = {frame_s} s")

    # Steps 1-2 (cached because they are the slow part)
    if processed_file.exists() and not args.reprocess:
        print(f"[1-2/7] Using cached {processed_file.name} (pass --reprocess to redo)")
        processed = tifffile.imread(processed_file)
        if processed.dtype != np.uint16:  # cache from an older float32 version
            processed = to_uint16(processed)
            tifffile.imwrite(processed_file, processed, imagej=True, metadata={"axes": "TYX"})
            print(f"      Converted cached {processed_file.name} to uint16")
    else:
        # Registration is estimated on the raw frames but applied AFTER background
        # subtraction. If the rolling ball runs on registered frames, the zero-filled
        # strips at shifted edges pull the background estimate down and the edges of
        # those frames come out artificially bright.
        if args.no_registration:
            processed = subtract_background(raw, args.radius)
        else:
            sr, tmats = compute_registration(raw, args.reg_mode)
            processed = apply_registration(sr, tmats, subtract_background(raw, args.radius))
        # Round to whole counts and store as uint16 (half the size of float32; changes
        # each pixel by <= 0.5 counts, negligible after averaging over an ROI).
        processed = to_uint16(processed)
        tifffile.imwrite(processed_file, processed, imagej=True, metadata={"axes": "TYX"})

    # Step 3
    print("[3/7] Max projection")
    max_proj = processed.max(axis=0)
    tifffile.imwrite(maxproj_file, max_proj)

    # Steps 4-5
    rois = draw_rois(processed, max_proj, roi_file, pixel_um)
    if len(rois.data) == 0:
        print("No ROIs drawn; nothing to measure.")
        return
    masks = save_rois(rois, roi_file, labels_file, max_proj.shape)

    # Steps 6-7
    df = measure(processed, masks)
    df.to_csv(meas_file, index=False)
    print(f"[7/7] Saved measurements -> {meas_file}")

    # ΔF/F, computed from the saved measurements.csv
    b0, b1 = args.baseline
    df = pd.read_csv(meas_file)
    df_dff = delta_f_over_f(df, b0, b1)
    df_dff.to_csv(dff_file, index=False)
    print(f"      Saved ΔF/F % (F0 = mean of frames {b0}-{b1}) -> {dff_file}")

    # ΔF/F normalized to each ROI's maximum (max = 100%)
    df_norm = normalize_to_max(df_dff)
    df_norm.to_csv(norm_file, index=False)
    print(f"      Saved ΔF/F normalized to max -> {norm_file}")

    # Plot
    plot_traces(df, df_dff, df_norm, (b0, b1), args.frame_interval or frame_s, plot_file,
                title=args.image.name)
    if not args.no_open:
        open_file(plot_file)


if __name__ == "__main__":
    main()
