"""Convert an MP4 video into a frequency-space spectrum MP4 and analyse it for
fingerprints commonly left by AI-generated video.

The analysis combines three complementary signals:
  * spatial RAPSD       - upsampling fingerprints in the per-frame 2D FFT
  * high-pass residual   - isolates the synthetic noise fingerprint from scene
                           content before measuring its spectrum
  * temporal FFT         - per-pixel flicker / temporal inconsistency that real
                           cameras rarely produce
These are fused into a single heuristic "synthetic likelihood" indicator.

Outputs (all written next to the input video):
    <name>_spectra.mp4   - 512x512 colour spectrum video (one frame per source frame)
    <name>_rapsd.png     - spatial RAPSD (raw + high-pass residual), averaged over frames
    <name>_rapsd.csv     - RAPSD curves + summary detection features
    <name>_temporal.png  - temporal power spectrum + per-pixel flicker map
    <name>_report.txt    - all features and the combined heuristic score

Usage:
    python video_spectra.py <path-to-video.mp4> [output_spectra.mp4]

You can also drag-and-drop an MP4 file onto this script (or onto the
accompanying ``video_spectra.bat``).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # headless backend, no GUI needed
import matplotlib.pyplot as plt
import numpy as np

# Spectrum video is rendered at a fixed square resolution (1:1 aspect ratio).
SPECTRUM_SIZE = 512

# Temporal analysis: frames are downscaled and a bounded number are collected
# to keep memory in check on long clips.
TEMPORAL_SIZE = 128
TEMPORAL_TARGET_FRAMES = 600
GAUSSIAN_SIGMA = 1.5  # high-pass cutoff for residual analysis

# Optional calibrated logistic-regression model produced by ``calibrate.py``.
# If this file exists next to the script it overrides the hand-tuned ramps.
CALIBRATION_PATH = Path(__file__).with_name("calibration.json")

# Feature order used by the calibrated model. Keep in sync with calibrate.py.
FEATURE_ORDER = [
    "rapsd_loglog_slope",
    "high_freq_energy_ratio",
    "high_freq_residual_mean",
    "residual_high_freq_energy_ratio",
    "temporal_hf_ratio",
    "flicker_strength",
]


def compute_power_spectrum(gray: np.ndarray) -> np.ndarray:
    """Return the centred 2D power spectrum (|FFT|^2) of a grayscale frame."""
    fft = np.fft.fft2(gray.astype(np.float32))
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)
    return magnitude * magnitude


def power_to_display(power: np.ndarray) -> np.ndarray:
    """Convert a power spectrum to an 8-bit log-scaled image for display."""
    log_power = np.log1p(power)
    spectrum = cv2.normalize(log_power, None, 0, 255, cv2.NORM_MINMAX)
    return spectrum.astype(np.uint8)


def radial_profile(power: np.ndarray) -> np.ndarray:
    """Radially average a centred 2D power spectrum into a 1D curve.

    Index ``i`` of the result is the mean power at radial frequency ``i``
    (in pixels from the spectrum centre).
    """
    h, w = power.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)

    # Sum power within each integer radius ring, then divide by ring counts.
    tbin = np.bincount(r.ravel(), power.ravel())
    nr = np.bincount(r.ravel())
    nr[nr == 0] = 1
    return tbin / nr


def high_pass_residual(gray: np.ndarray) -> np.ndarray:
    """Return the high-pass residual of a grayscale frame.

    Subtracting a Gaussian-blurred copy removes low-frequency scene content,
    leaving the fine-grained noise where synthetic fingerprints concentrate.
    """
    g = gray.astype(np.float32)
    blur = cv2.GaussianBlur(g, (0, 0), GAUSSIAN_SIGMA)
    return g - blur


def compute_features(radii: np.ndarray, rapsd: np.ndarray) -> dict[str, float]:
    """Derive simple, comparable detection features from the averaged RAPSD.

    All features are heuristics; compare them against a real-footage baseline
    captured with a similar camera/codec rather than reading absolute values.
    """
    # Work in log-log space, ignoring the DC term (radius 0).
    valid = radii > 0
    log_r = np.log10(radii[valid])
    log_p = np.log10(rapsd[valid] + 1e-12)

    # Power-law slope: natural images fall off smoothly (steep negative slope).
    slope = float(np.polyfit(log_r, log_p, 1)[0])

    # Fraction of total power living in the upper half of the frequency range.
    half = len(rapsd) // 2
    total = float(rapsd[1:].sum()) + 1e-12
    hf_ratio = float(rapsd[half:].sum()) / total

    # Residual energy in the high-frequency tail after subtracting the fitted
    # power law. A positive bump ("kick-up") is a classic generator artifact.
    fit = np.polyval(np.polyfit(log_r, log_p, 1), log_r)
    residual = log_p - fit
    hf_residual = float(residual[len(residual) // 2:].mean())

    return {
        "rapsd_loglog_slope": slope,
        "high_freq_energy_ratio": hf_ratio,
        "high_freq_residual_mean": hf_residual,
    }


def compute_temporal_analysis(
    frames: list[np.ndarray], fps: float
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """FFT a stack of small grayscale frames along the time axis.

    Returns ``(features, temporal_freqs, temporal_spectrum, flicker_map)`` where
    the spectrum is averaged over all pixels and the flicker map is the
    per-pixel non-DC temporal energy. Real cameras concentrate energy near DC;
    AI video often shows extra energy at higher temporal frequencies (flicker).
    """
    stack = np.stack(frames, axis=0).astype(np.float32)  # (T, H, W)
    t = stack.shape[0]

    # Remove the per-pixel temporal mean so DC does not dominate.
    stack -= stack.mean(axis=0, keepdims=True)

    fft = np.fft.rfft(stack, axis=0)
    power = (np.abs(fft) ** 2)  # (F, H, W)
    freqs = np.fft.rfftfreq(t, d=1.0 / fps)  # cycles per second (Hz)

    # Spectrum averaged across every pixel.
    spectrum = power.mean(axis=(1, 2))

    # Per-pixel flicker map = total non-DC temporal energy, log-scaled image.
    flicker = power[1:].sum(axis=0)
    flicker_img = cv2.normalize(np.log1p(flicker), None, 0, 255, cv2.NORM_MINMAX)
    flicker_img = flicker_img.astype(np.uint8)

    # Feature: fraction of temporal energy above the lowest few bins.
    total = float(spectrum.sum()) + 1e-12
    low_cut = max(1, len(spectrum) // 8)
    temporal_hf_ratio = float(spectrum[low_cut:].sum()) / total

    # Feature: spatial concentration of flicker (synthetic flicker is often
    # broad/global, real motion flicker is localised). High kurtosis = peaky.
    flat = flicker.ravel()
    mean = flat.mean() + 1e-12
    flicker_strength = float(np.log1p(mean))

    features = {
        "temporal_hf_ratio": temporal_hf_ratio,
        "flicker_strength": flicker_strength,
    }
    return features, freqs, spectrum, flicker_img


def load_calibration() -> dict | None:
    """Load the calibrated logistic-regression model if one exists."""
    if not CALIBRATION_PATH.is_file():
        return None
    try:
        return json.loads(CALIBRATION_PATH.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Warning: could not read calibration model ({exc}); using ramps.")
        return None


def score_with_model(features: dict[str, float], model: dict) -> float:
    """Apply a standardised logistic-regression model to a feature dict."""
    names = model["features"]
    x = np.array([float(features.get(n, 0.0)) for n in names], dtype=np.float64)
    mean = np.array(model["mean"], dtype=np.float64)
    std = np.array(model["std"], dtype=np.float64)
    coef = np.array(model["coef"], dtype=np.float64)
    z = (x - mean) / np.where(std == 0, 1.0, std)
    logit = float(np.dot(z, coef) + model["intercept"])
    return float(1.0 / (1.0 + np.exp(-logit)))


def compute_score_ramps(features: dict[str, float]) -> float:
    """Hand-tuned fallback score used when no calibration model is present.

    The ramp thresholds were calibrated against the bundled ``examples/`` clips
    (one real, two Grok-generated). On that data the synthetic clips are
    *smoother and less flickery* than the real footage - lower high-frequency
    spatial energy, lower temporal flicker, a steeper spectral falloff, and a
    relative high-frequency residual bump. These directions match common traits
    of current text-to-video models, but you should re-calibrate with your own
    labelled set before relying on the absolute number.
    """
    def ramp_up(value: float, low: float, high: float) -> float:
        # 0 at/below ``low``, 1 at/above ``high`` (higher value -> synthetic).
        if high == low:
            return 0.0
        return float(np.clip((value - low) / (high - low), 0.0, 1.0))

    def ramp_down(value: float, low: float, high: float) -> float:
        # 1 at/below ``low``, 0 at/above ``high`` (lower value -> synthetic).
        if high == low:
            return 0.0
        return float(np.clip((high - value) / (high - low), 0.0, 1.0))

    # Relative high-frequency residual bump above the fitted power law.
    residual_term = ramp_up(features["high_freq_residual_mean"], -0.17, -0.05)
    # Synthetic clips here are temporally smoother (less flicker energy).
    temporal_term = ramp_down(features["temporal_hf_ratio"], 0.05, 0.30)
    # Synthetic clips here have a lower overall per-pixel flicker strength.
    flicker_term = ramp_down(features["flicker_strength"], 15.0, 20.0)
    # Synthetic clips here carry less spatial high-frequency energy.
    hf_energy_term = ramp_down(
        features["high_freq_energy_ratio"], 0.000005, 0.00008
    )

    score = (
        0.25 * residual_term
        + 0.25 * temporal_term
        + 0.25 * flicker_term
        + 0.25 * hf_energy_term
    )
    return float(np.clip(score, 0.0, 1.0))


def compute_score(features: dict[str, float]) -> float:
    """Fuse the heuristics into a 0-1 "synthetic likelihood" indicator.

    If a calibrated model (``calibration.json`` from ``calibrate.py``) is present
    it is used; otherwise a transparent hand-tuned ramp fallback is applied.
    Either way this is NOT proof - compression weakens the signal and newer
    generators suppress these artifacts.
    """
    model = load_calibration()
    if model is not None:
        return score_with_model(features, model)
    return compute_score_ramps(features)


def analyse_and_convert(input_path: Path, spectra_path: Path) -> None:
    result = process_video(input_path, spectra_path=spectra_path)
    if result is None:
        return

    features = result["features"]
    write_rapsd_outputs(
        input_path,
        result["radii"],
        result["rapsd"],
        result["residual_rapsd"],
        features,
    )

    temporal = result["temporal"]
    if temporal is not None:
        write_temporal_outputs(
            input_path,
            temporal["freqs"],
            temporal["spectrum"],
            temporal["flicker_img"],
        )

    score = compute_score(features)
    write_report(input_path, features, score, result["frame_count"])


def process_video(
    input_path: Path,
    spectra_path: Path | None = None,
    progress: bool = True,
) -> dict | None:
    """Run the FFT analysis pipeline over a video.

    If ``spectra_path`` is given, the 512x512 spectrum video is written as a
    side effect. When it is ``None`` (e.g. during calibration) the expensive
    video encoding and per-frame colormapping are skipped for speed.

    Returns a dict with keys ``features``, ``radii``, ``rapsd``,
    ``residual_rapsd``, ``temporal`` and ``frame_count`` (or ``None`` if the
    video had no readable frames).
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    writer = None
    if spectra_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(spectra_path), fourcc, fps, (SPECTRUM_SIZE, SPECTRUM_SIZE), isColor=True
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open output for writing: {spectra_path}")
        print(f"Input : {input_path}")
        print(f"Output: {spectra_path}")
        print(f"Spectrum video: {SPECTRUM_SIZE}x{SPECTRUM_SIZE} @ {fps:.2f} fps")

    # Collect every Nth frame for temporal analysis so memory stays bounded.
    temporal_stride = max(1, total // TEMPORAL_TARGET_FRAMES) if total else 1

    rapsd_accum: np.ndarray | None = None
    residual_accum: np.ndarray | None = None
    temporal_frames: list[np.ndarray] = []
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            power = compute_power_spectrum(gray)

            # Accumulate the radial profile for the averaged RAPSD analysis.
            profile = radial_profile(power)
            if rapsd_accum is None:
                rapsd_accum = np.zeros_like(profile)
            n = min(len(rapsd_accum), len(profile))
            rapsd_accum[:n] += profile[:n]

            # High-pass residual spectrum (isolates the synthetic noise floor).
            residual = high_pass_residual(gray)
            res_profile = radial_profile(compute_power_spectrum(residual))
            if residual_accum is None:
                residual_accum = np.zeros_like(res_profile)
            m = min(len(residual_accum), len(res_profile))
            residual_accum[:m] += res_profile[:m]

            # Collect a downscaled frame for the temporal FFT.
            if frame_index % temporal_stride == 0:
                small = cv2.resize(
                    gray, (TEMPORAL_SIZE, TEMPORAL_SIZE), interpolation=cv2.INTER_AREA
                )
                temporal_frames.append(small)

            # Render the pretty 512x512 spectrum frame (only when writing video).
            if writer is not None:
                spectrum = power_to_display(power)
                colored = cv2.applyColorMap(spectrum, cv2.COLORMAP_INFERNO)
                colored = cv2.resize(
                    colored, (SPECTRUM_SIZE, SPECTRUM_SIZE), interpolation=cv2.INTER_AREA
                )
                writer.write(colored)

            frame_index += 1
            if progress:
                if total:
                    print(f"\rProcessing frame {frame_index}/{total}", end="", flush=True)
                else:
                    print(f"\rProcessing frame {frame_index}", end="", flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if progress:
        print()
    if writer is not None:
        print(f"Wrote {frame_index} spectrum frames to {spectra_path}")

    if rapsd_accum is None or frame_index == 0:
        print("No frames processed; skipping analysis.")
        return None

    rapsd = rapsd_accum / frame_index
    residual_rapsd = residual_accum / frame_index
    radii = np.arange(len(rapsd))

    features = compute_features(radii, rapsd)

    # Residual high-frequency energy ratio (same idea, on the residual spectrum).
    res_half = len(residual_rapsd) // 2
    res_total = float(residual_rapsd[1:].sum()) + 1e-12
    features["residual_high_freq_energy_ratio"] = float(
        residual_rapsd[res_half:].sum()
    ) / res_total

    # Temporal FFT analysis needs at least a few frames to be meaningful.
    temporal = None
    if len(temporal_frames) >= 4:
        t_features, t_freqs, t_spectrum, flicker_img = compute_temporal_analysis(
            temporal_frames, fps
        )
        features.update(t_features)
        temporal = {
            "freqs": t_freqs,
            "spectrum": t_spectrum,
            "flicker_img": flicker_img,
        }
    else:
        print("Too few frames for temporal analysis; skipping.")
        features.setdefault("temporal_hf_ratio", 0.0)
        features.setdefault("flicker_strength", 0.0)

    return {
        "features": features,
        "radii": radii,
        "rapsd": rapsd,
        "residual_rapsd": residual_rapsd,
        "temporal": temporal,
        "frame_count": frame_index,
    }



def write_rapsd_outputs(
    input_path: Path,
    radii: np.ndarray,
    rapsd: np.ndarray,
    residual_rapsd: np.ndarray,
    features: dict[str, float],
) -> None:
    plot_path = input_path.with_name(f"{input_path.stem}_rapsd.png")
    csv_path = input_path.with_name(f"{input_path.stem}_rapsd.csv")

    # --- Plot (log-log RAPSD: raw + high-pass residual) ---
    valid = radii > 0
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(radii[valid], rapsd[valid] + 1e-12, color="#d9480f", label="raw frame")
    ax.loglog(
        radii[valid],
        residual_rapsd[valid] + 1e-12,
        color="#1c7ed6",
        label="high-pass residual",
    )
    ax.set_title(f"Radially-Averaged Power Spectrum\n{input_path.name}")
    ax.set_xlabel("Radial spatial frequency (pixels from centre)")
    ax.set_ylabel("Mean power (log)")
    ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

    summary = (
        f"log-log slope: {features['rapsd_loglog_slope']:.3f}\n"
        f"HF energy ratio: {features['high_freq_energy_ratio']:.4f}\n"
        f"HF residual mean: {features['high_freq_residual_mean']:.4f}\n"
        f"residual HF ratio: {features['residual_high_freq_energy_ratio']:.4f}"
    )
    ax.text(
        0.02,
        0.02,
        summary,
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    # --- CSV (features header + full RAPSD curves) ---
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# summary_features"])
        for key, value in features.items():
            w.writerow([key, value])
        w.writerow([])
        w.writerow(["radius", "raw_mean_power", "residual_mean_power"])
        for radius, raw_p, res_p in zip(
            radii.tolist(), rapsd.tolist(), residual_rapsd.tolist()
        ):
            w.writerow([radius, raw_p, res_p])

    print(f"RAPSD plot: {plot_path}")
    print(f"RAPSD data: {csv_path}")


def write_temporal_outputs(
    input_path: Path,
    freqs: np.ndarray,
    spectrum: np.ndarray,
    flicker_img: np.ndarray,
) -> None:
    plot_path = input_path.with_name(f"{input_path.stem}_temporal.png")

    flicker_colored = cv2.applyColorMap(flicker_img, cv2.COLORMAP_MAGMA)
    flicker_rgb = cv2.cvtColor(flicker_colored, cv2.COLOR_BGR2RGB)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Skip the DC bin (index 0) which was already mean-removed.
    ax1.semilogy(freqs[1:], spectrum[1:] + 1e-12, color="#5f3dc4")
    ax1.set_title("Temporal Power Spectrum")
    ax1.set_xlabel("Temporal frequency (Hz)")
    ax1.set_ylabel("Mean power (log)")
    ax1.grid(True, which="both", linewidth=0.3, alpha=0.5)

    ax2.imshow(flicker_rgb)
    ax2.set_title("Per-pixel Flicker Map (non-DC energy)")
    ax2.axis("off")

    fig.suptitle(input_path.name)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    print(f"Temporal plot: {plot_path}")


def write_report(
    input_path: Path,
    features: dict[str, float],
    score: float,
    frame_count: int,
) -> None:
    report_path = input_path.with_name(f"{input_path.stem}_report.txt")

    if score >= 0.55:
        verdict = "LEANS SYNTHETIC"
    elif score >= 0.35:
        verdict = "UNCERTAIN"
    else:
        verdict = "LEANS REAL"

    model = load_calibration()
    if model is not None:
        method = (
            f"calibrated logistic regression "
            f"(n={model.get('n_samples', '?')}, "
            f"LOO acc={model.get('loo_accuracy', float('nan')):.2f})"
        )
    else:
        method = "hand-tuned ramps (no calibration model found)"

    lines = [
        f"Spectral AI-video analysis report",
        f"=================================",
        f"File         : {input_path.name}",
        f"Frames used  : {frame_count}",
        f"Scorer       : {method}",
        f"",
        f"Synthetic likelihood score: {score:.3f}  (0 = real-like, 1 = synthetic-like)",
        f"Verdict                   : {verdict}",
        f"",
        f"This score is a transparent heuristic, NOT proof. Compression weakens",
        f"the signal and newer generators suppress these artifacts. Always compare",
        f"against a real-footage baseline from a similar camera/codec.",
        f"",
        f"Features:",
    ]
    for key, value in features.items():
        lines.append(f"  {key:<32}: {value:.6f}")

    report_path.write_text("\n".join(lines) + "\n")

    print(f"Report: {report_path}")
    print(f"Synthetic likelihood score: {score:.3f} ({verdict})")
    print("Features (compare against a real-footage baseline):")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")


def resolve_output(input_path: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    return input_path.with_name(f"{input_path.stem}_spectra.mp4")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python video_spectra.py <input.mp4> [output_spectra.mp4]")
        print("Or drag-and-drop an MP4 file onto this script.")
        return 1

    input_path = Path(argv[1]).expanduser().resolve()
    if not input_path.is_file():
        print(f"Error: file not found: {input_path}")
        return 1

    spectra_path = resolve_output(input_path, argv[2] if len(argv) > 2 else None)

    try:
        analyse_and_convert(input_path, spectra_path)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the user
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
