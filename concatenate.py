"""Concatenate the ROI measurements from every timeseries analysed by pipeline.py.

Finds every `<image name>_analysis/` folder in a directory (chosen in a folder dialog
unless given on the command line),
and for each of
    measurements.csv, measurements_dFF.csv, measurements_dFF_norm.csv
joins the ROI columns side by side, renumbering them 1, 2, 3, ... across series
(series are taken in natural order: series 1, 2, ..., 10).

Usage
-----
    uv run python concatenate.py               # opens a folder picker
    uv run python concatenate.py folder [--outdir DIR]

Outputs go into `concatenated_analysis/`:
    measurements.csv           Frame, Mean1, Mean2, ...       (all ROIs, all series)
    measurements_dFF.csv       Frame, dFF1, dFF2, ...
    measurements_dFF_norm.csv  Frame, dFFnorm1, dFFnorm2, ...
    roi_key.csv                which series and original ROI each new ROI number came from
    average_dFF.csv            Frame, Time (min), mean and SEM across all ROIs of ΔF/F (%) and
                               normalized ΔF/F (%)
    responders.csv             per ROI: baseline SD of normalized ΔF/F, threshold, longest run of
                               consecutive frames above threshold in the response window, responder?
    average_traces.pdf         mean ± SEM of ΔF/F (%) and of normalized ΔF/F (%) vs time (min),
                               with the drug conditions in a box at the top and the responding /
                               total cell count at the bottom; opened automatically
    conditions.txt             the drug conditions text (offered as the default on the next run)

Drug conditions: pass --conditions "..." or type them when prompted (press Enter to keep the
previous text, or type "-" for no box). Use "\n" in the text to start a new line.

Responding cell (from measurements_dFF_norm.csv): within the response window (default frames
13-25, the first minute after drug addition at frame 12), normalized ΔF/F is >= 3 x the ROI's baseline SD
(SD over frames 1-12) AND >= 10, for at least 3 consecutive frames.
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import INK, INK_MUTED, ROI_COLORS, _style_axes, open_file

# file name -> column prefix used by pipeline.py
FILES = {
    "measurements.csv": "Mean",
    "measurements_dFF.csv": "dFF",
    "measurements_dFF_norm.csv": "dFFnorm",
}
OUT_NAME = "concatenated_analysis"


def natural_key(path: Path):
    """Sort 'series 10' after 'series 2'."""
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", path.name)]


def roi_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    """Columns like Mean1, Mean2 (exact prefix + number, so 'dFF' doesn't match 'dFFnorm3')."""
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    return sorted((c for c in df.columns if pat.match(c)), key=lambda c: int(pat.match(c).group(1)))


def concatenate(folders: list[Path], fname: str, prefix: str) -> tuple[pd.DataFrame, list[dict]]:
    """Join one CSV type across all folders. Returns (table, key rows)."""
    merged: pd.DataFrame | None = None
    key: list[dict] = []
    n = 0
    for folder in folders:
        df = pd.read_csv(folder / fname)
        cols = roi_columns(df, prefix)
        rename = {}
        for c in cols:
            n += 1
            rename[c] = f"{prefix}{n}"
            key.append({"ROI": n, "Series": folder.name.removesuffix("_analysis"),
                        "Original ROI": int(c[len(prefix):])})
        part = df[["Frame", *cols]].rename(columns=rename)
        if merged is None:
            merged = part
        else:
            if len(part) != len(merged) or not part["Frame"].equals(merged["Frame"]):
                print(f"      warning: {folder.name}/{fname} has {len(part)} frames vs "
                      f"{len(merged)} so far; missing frames are left blank")
            merged = merged.merge(part, on="Frame", how="outer")
    return merged.sort_values("Frame").reset_index(drop=True), key


def mean_sem(table: pd.DataFrame, prefix: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Per-frame mean, SEM (sample SD / sqrt(n)) and n across all ROI columns."""
    rois = table[roi_columns(table, prefix)]
    n = rois.notna().sum(axis=1)
    return rois.mean(axis=1), rois.std(axis=1, ddof=1) / np.sqrt(n), n


def find_responders(table: pd.DataFrame, prefix: str, baseline: tuple[int, int],
                    window: tuple[int, int], sd_mult: float, min_value: float,
                    min_frames: int) -> pd.DataFrame:
    """Classify each ROI as responding or not. A responder's signal is >= max(sd_mult x
    baseline SD, min_value) for at least min_frames consecutive frames within the window
    (frames inclusive, 1-based). Returns one row per ROI."""
    base = table.loc[table["Frame"].between(*baseline)]
    win = table.loc[table["Frame"].between(*window)]
    if base.empty or win.empty:
        raise SystemExit(f"Baseline {baseline} or response window {window} has no frames")
    rows = []
    for col in roi_columns(table, prefix):
        sd = base[col].std(ddof=1)
        threshold = max(sd_mult * sd, min_value)
        above = (win[col] >= threshold).to_numpy()
        longest = run = 0
        for a in above:  # longest run of consecutive frames above threshold
            run = run + 1 if a else 0
            longest = max(longest, run)
        rows.append({"ROI": int(col[len(prefix):]), "Baseline SD": sd, "Threshold": threshold,
                     "Longest run (frames)": longest, "Responder": longest >= min_frames})
    return pd.DataFrame(rows)


def plot_averages(avg: pd.DataFrame, n_rois: int, n_series: int, pdf_file: Path,
                  footer: str | None = None, drug_time: float | None = None,
                  conditions: str | None = None) -> None:
    """Two stacked plots vs time (min): mean ± SEM ΔF/F, and mean ± SEM normalized ΔF/F."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = avg["Time (min)"]
    color = ROI_COLORS[0]
    # Conditions box at the top: wrap to the page width and grow the figure to fit it
    lines = [w for line in (conditions or "").splitlines() for w in (textwrap.wrap(line, 95) or [""])]
    box_in = 0.35 + 0.17 * len(lines) if lines else 0.0
    fig_h = 7 + box_in
    fig, (ax_dff, ax_norm) = plt.subplots(2, 1, figsize=(7, fig_h), sharex=True)
    for ax, col, title, ylabel in (
        (ax_dff, "dFF", "ΔF/F", "ΔF / F₀ (%)"),
        (ax_norm, "dFFnorm", "ΔF/F normalized to each ROI's maximum", "ΔF/F (% of max)"),
    ):
        mean, sem = avg[f"{col} mean"], avg[f"{col} SEM"]
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.25, linewidth=0, label="SEM")
        ax.plot(x, mean, color=color, linewidth=1.5, label="Mean")
        ax.axhline(0.0, color=INK_MUTED, linewidth=0.8, linestyle="--", zorder=1)
        if drug_time is not None:
            ax.axvline(drug_time, color=INK, linewidth=1.0, linestyle=":", zorder=1,
                       label="Drug added")
        _style_axes(ax)
        ax.set_title(f"{title}  (mean ± SEM)", color=INK, fontsize=9, loc="left")
        ax.set_ylabel(ylabel, color=INK)
    ax_dff.set_xlim(x.min(), x.max())
    ax_dff.legend(frameon=False, fontsize=8, ncols=3, loc="lower right",
                  bbox_to_anchor=(1.0, 1.0), labelcolor=INK)
    ax_norm.set_xlabel("Time (min)", color=INK)
    top = 1 - box_in / fig_h
    if lines:
        fig.text(0.01, 1 - 0.15 / fig_h, "\n".join(lines), color=INK, fontsize=9, ha="left", va="top",
                 linespacing=1.4, bbox=dict(boxstyle="round,pad=0.5", facecolor="#f4f3ef",
                                            edgecolor=INK_MUTED, linewidth=0.6))
    fig.suptitle(f"Average of {n_rois} ROIs from {n_series} timeseries", color=INK,
                 fontsize=10, x=0.01, y=top - 0.1 / fig_h, ha="left", va="top")
    bottom = 0.06 * 7 / fig_h if footer else 0
    fig.tight_layout(rect=(0, bottom, 1, top))
    if footer:
        fig.text(0.01, 0.1 / fig_h, footer, color=INK, fontsize=9, ha="left", va="bottom")
    fig.savefig(pdf_file)
    plt.close(fig)
    print(f"Saved plot -> {pdf_file}")


def get_conditions(given: str | None, saved_file: Path) -> str:
    """Drug conditions text: from --conditions, else typed at a prompt (Enter keeps the text
    saved from the last run, "-" means none). Saved to saved_file for next time."""
    previous = saved_file.read_text().strip() if saved_file.exists() else ""
    if given is None:
        if sys.stdin.isatty():
            if previous:
                print(f"Previous drug conditions:\n    {previous}")
                prompt = 'Drug conditions (Enter = keep previous, "-" = none, "\\n" = new line): '
            else:
                prompt = 'Drug conditions (Enter = none, "\\n" = new line): '
            typed = input(prompt).strip()
            given = previous if typed == "" else typed
        else:  # not interactive (e.g. piped): reuse the previous text
            given = previous
    text = "" if given.strip() == "-" else given.replace("\\n", "\n").strip()
    saved_file.write_text(text + "\n" if text else "")
    return text


def choose_folder() -> Path | None:
    """Open a dialog to pick the folder holding the *_analysis/ folders. None if cancelled."""
    from napari.qt import get_qapp
    from qtpy.QtWidgets import QFileDialog

    get_qapp()
    path = QFileDialog.getExistingDirectory(
        None, "Choose the folder containing the *_analysis folders", str(Path.home()))
    return Path(path) if path else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", type=Path, nargs="?", default=None,
                   help="folder containing the *_analysis/ folders (omit to choose it in a dialog)")
    p.add_argument("--outdir", type=Path, default=None, help=f"output folder (default: <folder>/{OUT_NAME})")
    p.add_argument("--frame-interval", type=float, default=5,
                   help="seconds per frame, for the time axis (default 5)")
    p.add_argument("--baseline", type=int, nargs=2, default=(1, 12), metavar=("FIRST", "LAST"),
                   help="frames (inclusive) for the baseline SD in the responder test (default: 1 12)")
    p.add_argument("--response-window", type=int, nargs=2, default=(13, 25), metavar=("FIRST", "LAST"),
                   help="frames (inclusive) searched for a response (default: 13 25)")
    p.add_argument("--drug-frame", type=int, default=12,
                   help="frame of drug addition, marked with a dotted line on the plots (default 12)")
    p.add_argument("--sd-mult", type=float, default=3, help="threshold = this x baseline SD (default 3)")
    p.add_argument("--min-value", type=float, default=10,
                   help="threshold is at least this (%% of max) (default 10)")
    p.add_argument("--min-frames", type=int, default=3,
                   help="consecutive frames above threshold needed (default 3)")
    p.add_argument("--conditions", default=None,
                   help='drug conditions shown at the top of the plot ("\\n" = new line); '
                        "if omitted you are prompted for them")
    p.add_argument("--no-open", action="store_true", help="don't open the plot PDF when done")
    args = p.parse_args()
    if args.folder is None:
        args.folder = choose_folder()
        if args.folder is None:
            print("No folder chosen; exiting.")
            return

    outdir = args.outdir or args.folder / OUT_NAME
    folders = sorted(
        (d for d in args.folder.glob("*_analysis")
         if d.is_dir() and d.resolve() != outdir.resolve() and all((d / f).exists() for f in FILES)),
        key=natural_key,
    )
    if not folders:
        raise SystemExit(f"No *_analysis folders with {', '.join(FILES)} found in {args.folder.resolve()}")
    print(f"Found {len(folders)} analysed timeseries:")
    for d in folders:
        print(f"      {d.name}")

    outdir.mkdir(parents=True, exist_ok=True)
    conditions = get_conditions(args.conditions, outdir / "conditions.txt")
    key = None
    tables = {}
    for fname, prefix in FILES.items():
        table, this_key = concatenate(folders, fname, prefix)
        if key is None:
            key = this_key
        elif this_key != key:  # e.g. an ROI was skipped in one file but not another
            raise SystemExit(f"ROI columns in {fname} don't match measurements.csv; re-run pipeline.py")
        tables[prefix] = table
        table.to_csv(outdir / fname, index=False)
        print(f"Saved {len(table.columns) - 1} ROIs x {len(table)} frames -> {outdir / fname}")

    pd.DataFrame(key).to_csv(outdir / "roi_key.csv", index=False)
    print(f"Saved ROI key -> {outdir / 'roi_key.csv'}")

    # Mean ± SEM across all ROIs
    frames = tables["dFF"]["Frame"]
    avg = pd.DataFrame({"Frame": frames, "Time (min)": (frames - 1) * args.frame_interval / 60})
    for prefix in ("dFF", "dFFnorm"):
        mean, sem, n = mean_sem(tables[prefix], prefix)
        avg[f"{prefix} mean"], avg[f"{prefix} SEM"] = mean, sem
    avg["n ROIs"] = n
    avg.to_csv(outdir / "average_dFF.csv", index=False)
    print(f"Saved mean and SEM -> {outdir / 'average_dFF.csv'}")

    # Responding vs non-responding cells, from normalized ΔF/F
    resp = find_responders(tables["dFFnorm"], "dFFnorm", tuple(args.baseline), tuple(args.response_window),
                           args.sd_mult, args.min_value, args.min_frames)
    resp = pd.DataFrame(key).merge(resp, on="ROI")
    resp.to_csv(outdir / "responders.csv", index=False)
    n_resp, n_total = int(resp["Responder"].sum()), len(resp)
    w0, w1 = args.response_window
    footer = (f"Responding cells: {n_resp} / {n_total} ({100 * n_resp / n_total:.0f}%)    "
              f"Criterion: normalized ΔF/F ≥ {args.sd_mult:g}× baseline SD (frames "
              f"{args.baseline[0]}–{args.baseline[1]}) and ≥ {args.min_value:g}%\n"
              f"for ≥ {args.min_frames} consecutive frames within frames {w0}–{w1}")
    print(f"Responding cells: {n_resp} / {n_total}  (non-responding: {n_total - n_resp})")
    print(f"Saved per-ROI responder results -> {outdir / 'responders.csv'}")

    plot_file = outdir / "average_traces.pdf"
    drug_time = (args.drug_frame - 1) * args.frame_interval / 60
    plot_averages(avg, len(key), len(folders), plot_file, footer, drug_time, conditions)
    if not args.no_open:
        open_file(plot_file)


if __name__ == "__main__":
    main()
