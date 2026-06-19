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
    <name>_result.png    - one consolidated image: spatial RAPSD, temporal power
                           spectrum, flicker map, residual-fingerprint FFT, the
                           feature readings, and the score/verdict in bold
    <name>_rapsd.csv     - RAPSD curves + summary detection features

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

# The fingerprint FFT amplitude is analysed at a higher fixed resolution than the
# spectrum video. More pixels between lattice nodes keeps the periodic generator
# peaks compact and separable instead of being merged by the area-downsample, so
# faint lattices stand a better chance of being detected and circled. All
# fingerprint detection thresholds below are fractions of this size, so they
# scale with it automatically.
FINGERPRINT_SIZE = 1024

# Temporal analysis: frames are downscaled and a bounded number are collected
# to keep memory in check on long clips.
TEMPORAL_SIZE = 128
TEMPORAL_TARGET_FRAMES = 600
GAUSSIAN_SIGMA = 1.5  # high-pass cutoff for residual analysis

# Fingerprint peak detection (circled hot spots in the residual FFT amplitude).
# The genuine upsampling fingerprint is a *single regular 2D lattice* of compact
# blobs: every peak sits at DC + m*v1 + n*v2 for integer m, n. The detector
# band-pass filters to keep compact blobs (rejecting the broad bright lobes and
# pixel noise), gathers candidates, then fits ONE global lattice and circles only
# the candidates that fall on it and span several grid cells. Isolated blobs,
# clustered lobe texture, and curved arcs do not form a wide 2D lattice, so they
# are rejected.
FINGERPRINT_DC_RADIUS_FRAC = 0.045      # exclude only the DC term / its disk
FINGERPRINT_OUTER_RADIUS_FRAC = 0.49    # ignore the near-edge / Nyquist noise
FINGERPRINT_CAND_SIGMA = 3.0            # permissive candidate (MAD) threshold
FINGERPRINT_MAX_PEAKS = 80              # cap on the number of circled hot spots
# Band-pass (difference-of-Gaussians) scales: small = denoise, large = remove
# broad lobes/streaks. Both scale with the image size so blob/noise proportions
# stay constant regardless of the analysis resolution.
FINGERPRINT_BP_SMALL_DIV = 425.0        # small sigma = image_size / divisor
FINGERPRINT_BP_LARGE_DIV = 40.0         # large sigma = image_size / divisor
FINGERPRINT_MIN_SEPARATION_FRAC = 0.022  # min spacing so one blob -> one candidate
FINGERPRINT_CAND_CAP = 220              # cap candidates (keeps O(n^2) fit cheap)
# Global lattice fit parameters.
FINGERPRINT_GRID_TOL_FRAC = 0.018       # spacing-vote tolerance (basis histogram)
FINGERPRINT_FIT_TOL_FRAC = 0.010        # how close to a node counts as an inlier
FINGERPRINT_GRID_MIN_PERIOD_FRAC = 0.05  # smallest believable lattice spacing
FINGERPRINT_GRID_MAX_PERIOD_FRAC = 0.32  # largest believable basis spacing
FINGERPRINT_BASIS_MIN_VOTES = 6         # pairs that must share a basis spacing
FINGERPRINT_GRID_MIN_INLIERS = 8        # min on-lattice peaks before drawing
FINGERPRINT_GRID_MIN_SPAN = 2           # lattice must span >= this many cells per axis
FINGERPRINT_GRID_MIN_FRACTION = 0.45    # >= this share of candidates must be on-grid
FINGERPRINT_GRID_SIGNIF_FACTOR = 3.0    # inliers must exceed chance count by this x

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


def compute_fingerprint_spectrum(fingerprint: np.ndarray) -> dict:
    """2D FFT amplitude (log) of the averaged high-pass residual fingerprint.

    Averaging the *signed* residual over many frames cancels incoherent scene
    content while the generator's spatially-coherent periodic fingerprint
    survives. Its FFT amplitude exposes the regular peak grid left by
    transposed-convolution / upsampling layers - a classic AI-video artifact.

    Returns a dict with ``image`` (8-bit centred amplitude at the fixed square
    resolution, ready for colormapping) and ``peaks`` (a list of ``(x, y)``
    hot-spot coordinates in that image, the candidate AI fingerprint peaks).
    """
    f = fingerprint.astype(np.float32)
    f = f - f.mean()

    # Hann window suppresses the FFT edge-wrap cross artifact so genuine
    # generator peaks are not masked by spectral leakage.
    wy = np.hanning(f.shape[0])
    wx = np.hanning(f.shape[1])
    f = f * np.outer(wy, wx)

    fft = np.fft.fftshift(np.fft.fft2(f))
    amplitude = np.log1p(np.abs(fft))

    # Resample the float amplitude to the fixed square (1:1) analysis resolution
    # so the display image and detected peak coordinates share one space.
    amplitude = cv2.resize(
        amplitude, (FINGERPRINT_SIZE, FINGERPRINT_SIZE), interpolation=cv2.INTER_AREA
    )

    img = cv2.normalize(amplitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    peaks = detect_spectral_peaks(amplitude)
    return {"image": img, "peaks": peaks}


def detect_spectral_peaks(amplitude: np.ndarray) -> list[tuple[int, int]]:
    """Locate the regular *lattice* of hot spots in a centred FFT log-amplitude.

    The genuine upsampling fingerprint is a single 2D lattice of compact blobs.
    We therefore:

      1. band-pass filter (difference of Gaussians) so only compact blobs
         survive - this removes the broad bright low-frequency lobes and the
         smooth radial streaks that previously produced false candidates;
      2. gather candidate local maxima in the valid annulus;
      3. fit ONE global lattice and keep only candidates that fall on it and
         span several grid cells (see ``_select_lattice_peaks``).

    Isolated blobs, clustered lobe texture and curved arcs do not form a wide
    2D lattice, so they are rejected. An empty list means no lattice was found
    (real footage is usually featureless here).
    """
    amp = amplitude.astype(np.float32)
    h, w = amp.shape
    cy, cx = h / 2.0, w / 2.0
    size = min(h, w)

    # Band-pass: keep compact grid blobs, suppress BOTH pixel noise (small scale)
    # and the broad bright lobes / radial streaks (large scale). A plain wide
    # background left the lobes behind; the difference of Gaussians removes them.
    small = cv2.GaussianBlur(amp, (0, 0), sigmaX=max(1.0, size / FINGERPRINT_BP_SMALL_DIV))
    large = cv2.GaussianBlur(amp, (0, 0), sigmaX=max(2.0, size / FINGERPRINT_BP_LARGE_DIV))
    prominence = small - large

    # Mask the central DC disk, the thin leakage cross, and the Nyquist edge.
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    inner = FINGERPRINT_DC_RADIUS_FRAC * size
    outer = FINGERPRINT_OUTER_RADIUS_FRAC * size
    exclude = (
        (r < inner)
        | (r > outer)
        | (np.abs(x - cx) < 2)
        | (np.abs(y - cy) < 2)
    )

    valid = prominence[~exclude]
    if valid.size == 0:
        return []

    # Permissive robust threshold to gather grid candidates (even dim ones); the
    # global lattice fit below removes the false positives this lets in.
    med = float(np.median(valid))
    mad = float(np.median(np.abs(valid - med))) + 1e-6
    threshold = med + FINGERPRINT_CAND_SIGMA * 1.4826 * mad

    floor = float(prominence.min()) - 1.0
    work = prominence.copy()
    work[exclude] = floor
    # Local-max window scales with resolution so one blob yields one maximum
    # regardless of the analysis size.
    win = max(3, int(round(size / 100.0)) | 1)
    dilated = cv2.dilate(work, np.ones((win, win), np.uint8))
    is_peak = (work == dilated) & (work > threshold)

    ys, xs = np.nonzero(is_peak)
    if ys.size == 0:
        return []

    # Non-max suppression so a single textured blob yields one candidate.
    order = np.argsort(work[ys, xs])[::-1]
    min_sep = FINGERPRINT_MIN_SEPARATION_FRAC * size
    min_sep_sq = min_sep * min_sep
    candidates: list[tuple[int, int]] = []
    for i in order:
        px, py = int(xs[i]), int(ys[i])
        if all((px - ax) ** 2 + (py - ay) ** 2 >= min_sep_sq for ax, ay in candidates):
            candidates.append((px, py))
        if len(candidates) >= FINGERPRINT_CAND_CAP:
            break

    return _select_lattice_peaks(candidates, (cx, cy), size)


def _candidate_basis_vectors(
    pts: np.ndarray, tol: float, min_period: float, max_period: float, top_k: int = 16
) -> list[np.ndarray]:
    """Return the most frequently-occurring pairwise spacing vectors.

    The basis vectors of a lattice (and their low multiples) recur far more
    often than random spacings. We histogram canonical pairwise differences,
    merge neighbouring bins, drop near-duplicates, and return the top vectors by
    vote so the caller can search for the basis that best explains the peaks.
    """
    n = len(pts)
    buckets: dict[tuple[int, int], int] = {}
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[j, 0] - pts[i, 0]
            dy = pts[j, 1] - pts[i, 1]
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < min_period or dist > max_period:
                continue
            key = (int(round(dx / tol)), int(round(dy / tol)))
            buckets[key] = buckets.get(key, 0) + 1
    if not buckets:
        return []

    def merged_votes(key: tuple[int, int]) -> int:
        kx, ky = key
        total = 0
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                total += buckets.get((kx + ox, ky + oy), 0)
        return total

    scored = sorted(
        ((merged_votes(k), k) for k in buckets), key=lambda t: t[0], reverse=True
    )

    vectors: list[np.ndarray] = []
    for votes, key in scored:
        if votes < FINGERPRINT_BASIS_MIN_VOTES:
            break
        v = np.array([key[0] * tol, key[1] * tol], dtype=np.float64)
        # Drop near-duplicates already collected (within tolerance).
        if any(float(np.hypot(*(v - u))) <= tol for u in vectors):
            continue
        vectors.append(v)
        if len(vectors) >= top_k:
            break
    return vectors


def _select_lattice_peaks(
    candidates: list[tuple[int, int]],
    center: tuple[float, float],
    size: float,
) -> list[tuple[int, int]]:
    """Keep only candidates that fall on a single global 2D lattice.

    Candidate basis vectors are taken from the dominant pairwise spacings, then
    every non-parallel pair ``(v1, v2)`` is tried as a basis; the basis that
    puts the most candidates on integer lattice nodes wins. Because a dense field
    of candidates lands near *some* node by chance, acceptance of the winning
    basis still requires the inlier count to (a) far exceed the chance count for
    that basis, (b) form a large fraction of all candidates, and (c) span several
    cells along both axes. Clustered noise, curved arcs and uniform texture all
    fail at least one test and are rejected.
    """
    n = len(candidates)
    if n < FINGERPRINT_GRID_MIN_INLIERS:
        return []

    pts = np.asarray(candidates, dtype=np.float64)
    tol = max(1.0, FINGERPRINT_GRID_TOL_FRAC * size)
    min_period = FINGERPRINT_GRID_MIN_PERIOD_FRAC * size
    max_period = FINGERPRINT_GRID_MAX_PERIOD_FRAC * size
    fit_tol = max(2.0, FINGERPRINT_FIT_TOL_FRAC * size)
    c = np.asarray(center, dtype=np.float64)
    rel_all = pts - c

    vectors = _candidate_basis_vectors(pts, tol, min_period, max_period)
    if len(vectors) < 2:
        return []

    def fit_basis(basis_matrix: np.ndarray):
        cell_area = abs(float(np.linalg.det(basis_matrix)))
        if cell_area < 1e-6:
            return None
        try:
            inv_basis = np.linalg.inv(basis_matrix)
        except np.linalg.LinAlgError:
            return None
        coeffs = rel_all @ inv_basis.T          # fractional (m, n) per candidate
        ints = np.round(coeffs)
        recon = ints @ basis_matrix.T           # nearest node offset from centre
        resid = np.hypot(recon[:, 0] - rel_all[:, 0], recon[:, 1] - rel_all[:, 1])
        on = (resid <= fit_tol) & ~((ints[:, 0] == 0) & (ints[:, 1] == 0))
        idx = np.nonzero(on)[0]
        return basis_matrix, cell_area, idx, ints[idx]

    def refine(v1: np.ndarray, v2: np.ndarray):
        # The histogram only locates the basis to bucket resolution, which is
        # too coarse to use directly (period error accumulates over cells).
        # Iteratively re-fit the basis from its own inliers by least squares.
        basis_matrix = np.column_stack([v1, v2]).astype(np.float64)
        res = fit_basis(basis_matrix)
        for _ in range(5):
            if res is None:
                return None
            _, _, idx, node_ints = res
            if len(idx) < 4 or np.linalg.matrix_rank(node_ints) < 2:
                return res
            # Solve rel_inliers = node_ints @ M.T  for M (2x2).
            sol, *_ = np.linalg.lstsq(node_ints, rel_all[idx], rcond=None)
            new_matrix = sol.T
            new_res = fit_basis(new_matrix)
            if new_res is None:
                return res
            if np.allclose(new_matrix, basis_matrix, atol=1e-2):
                return new_res
            basis_matrix = new_matrix
            res = new_res
        return res

    # Search non-parallel basis pairs; keep the one with the most inliers.
    best = None
    for a in range(len(vectors)):
        for b in range(len(vectors)):
            if a == b:
                continue
            v1, v2 = vectors[a], vectors[b]
            cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
            if cross <= 0.25 * float(np.hypot(*v1)) * float(np.hypot(*v2)):
                continue  # near-parallel -> not a 2D basis
            res = refine(v1, v2)
            if res is None:
                continue
            _, _, idx, _ = res
            if best is None or len(idx) > best[0]:
                best = (len(idx), res)

    if best is None:
        return []

    res = best[1]
    if res is None:
        return []
    basis_matrix, cell_area, idx, node_ints = res
    n_inliers = int(len(idx))
    if n_inliers < FINGERPRINT_GRID_MIN_INLIERS:
        return []

    # (a) Significance vs a random field of n points for this basis.
    expected_by_chance = n * np.pi * fit_tol * fit_tol / cell_area
    if n_inliers < FINGERPRINT_GRID_SIGNIF_FACTOR * expected_by_chance:
        return []

    # (b) A real grid is made *of* these peaks, so most candidates are on it.
    if n_inliers / n < FINGERPRINT_GRID_MIN_FRACTION:
        return []

    # (c) Require the inliers to span several cells along BOTH axes.
    ms = node_ints[:, 0]
    ks = node_ints[:, 1]
    if (ms.max() - ms.min()) < FINGERPRINT_GRID_MIN_SPAN or (
        ks.max() - ks.min()
    ) < FINGERPRINT_GRID_MIN_SPAN:
        return []
    if len(np.unique(ms)) < FINGERPRINT_GRID_MIN_SPAN or len(np.unique(ks)) < FINGERPRINT_GRID_MIN_SPAN:
        return []

    return [candidates[i] for i in idx.tolist()][:FINGERPRINT_MAX_PEAKS]



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
    score = compute_score(features)

    # One consolidated image (plots + readings + score/verdict) plus the raw
    # RAPSD curves as CSV for numeric comparison.
    write_result_image(input_path, result, features, score)
    write_rapsd_csv(
        input_path,
        result["radii"],
        result["rapsd"],
        result["residual_rapsd"],
        features,
    )


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
    fingerprint_accum: np.ndarray | None = None
    fingerprint_count = 0
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

            # Accumulate the signed residual itself to estimate the spatially
            # coherent artificial fingerprint. Normalising each frame's residual
            # by its own std equalises contributions so a few high-contrast
            # frames cannot dominate the coherent average.
            if fingerprint_accum is None:
                fingerprint_accum = np.zeros_like(residual)
            if residual.shape == fingerprint_accum.shape:
                res_std = float(residual.std())
                if res_std > 1e-6:
                    fingerprint_accum += residual / res_std
                    fingerprint_count += 1

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

        # FFT amplitude of the averaged residual fingerprint (shown next to the
        # flicker map) plus the detected hot-spot peaks. Estimated from every
        # processed frame's std-normalised residual.
        fingerprint = None
        if fingerprint_accum is not None and fingerprint_count > 0:
            mean_residual = fingerprint_accum / fingerprint_count
            fingerprint = compute_fingerprint_spectrum(mean_residual)

        temporal = {
            "freqs": t_freqs,
            "spectrum": t_spectrum,
            "flicker_img": flicker_img,
            "fingerprint": fingerprint,
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



def write_rapsd_csv(
    input_path: Path,
    radii: np.ndarray,
    rapsd: np.ndarray,
    residual_rapsd: np.ndarray,
    features: dict[str, float],
) -> None:
    """Write the summary features and full RAPSD curves to ``<name>_rapsd.csv``."""
    csv_path = input_path.with_name(f"{input_path.stem}_rapsd.csv")
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

    print(f"RAPSD data: {csv_path}")


def _verdict_for(score: float) -> tuple[str, str]:
    """Map a 0-1 score to a (verdict label, display colour)."""
    if score >= 0.55:
        return "LEANS SYNTHETIC", "#c92a2a"
    if score >= 0.35:
        return "UNCERTAIN", "#e8590c"
    return "LEANS REAL", "#2b8a3e"


def _scorer_method() -> str:
    """Describe which scorer (calibrated model vs ramps) is in use."""
    model = load_calibration()
    if model is not None:
        return (
            f"calibrated logistic regression "
            f"(n={model.get('n_samples', '?')}, "
            f"LOO acc={model.get('loo_accuracy', float('nan')):.2f})"
        )
    return "hand-tuned ramps (no calibration model found)"


def write_result_image(
    input_path: Path,
    result: dict,
    features: dict[str, float],
    score: float,
) -> None:
    """Render the single consolidated ``<name>_result.png``.

    Combines the RAPSD plot, temporal power spectrum, flicker map and
    fingerprint FFT amplitude with the numeric readings, and the synthetic
    likelihood score / verdict in bold at the bottom.
    """
    plot_path = input_path.with_name(f"{input_path.stem}_result.png")

    radii = result["radii"]
    rapsd = result["rapsd"]
    residual_rapsd = result["residual_rapsd"]
    temporal = result["temporal"]
    frame_count = result["frame_count"]

    verdict, verdict_color = _verdict_for(score)
    method = _scorer_method()

    has_temporal = temporal is not None
    has_fingerprint = has_temporal and temporal.get("fingerprint") is not None
    n_fingerprint_peaks = 0

    # --- Figure layout ---
    if has_temporal:
        fig = plt.figure(figsize=(13, 14))
        gs = fig.add_gridspec(
            4, 2, height_ratios=[1.0, 1.55, 0.65, 0.18], hspace=0.45, wspace=0.25
        )
        ax_rapsd = fig.add_subplot(gs[0, 0])
        ax_tspec = fig.add_subplot(gs[0, 1])
        # Give the two image panels their own tighter sub-grid so they fill
        # more of the row and sit closer together (without touching).
        img_gs = gs[1, :].subgridspec(1, 2, wspace=0.06)
        ax_flicker = fig.add_subplot(img_gs[0, 0])
        ax_finger = fig.add_subplot(img_gs[0, 1])
        ax_text = fig.add_subplot(gs[2, :])
    else:
        fig = plt.figure(figsize=(9, 9))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.4, 0.8, 0.18], hspace=0.5)
        ax_rapsd = fig.add_subplot(gs[0, 0])
        ax_text = fig.add_subplot(gs[1, 0])
        ax_tspec = ax_flicker = ax_finger = None

    # --- RAPSD (raw + high-pass residual) ---
    valid = radii > 0
    ax_rapsd.loglog(
        radii[valid], rapsd[valid] + 1e-12, color="#d9480f", label="raw frame"
    )
    ax_rapsd.loglog(
        radii[valid],
        residual_rapsd[valid] + 1e-12,
        color="#1c7ed6",
        label="high-pass residual",
    )
    ax_rapsd.set_title("Radially-Averaged Power Spectrum")
    ax_rapsd.set_xlabel("Radial spatial frequency (pixels from centre)")
    ax_rapsd.set_ylabel("Mean power (log)")
    ax_rapsd.grid(True, which="both", linewidth=0.3, alpha=0.5)
    ax_rapsd.legend(loc="upper right", fontsize=8)

    # --- Temporal panels ---
    if has_temporal:
        freqs = temporal["freqs"]
        spectrum = temporal["spectrum"]
        flicker_img = temporal["flicker_img"]

        # Skip the DC bin (index 0) which was already mean-removed.
        ax_tspec.semilogy(freqs[1:], spectrum[1:] + 1e-12, color="#5f3dc4")
        ax_tspec.set_title("Temporal Power Spectrum")
        ax_tspec.set_xlabel("Temporal frequency (Hz)")
        ax_tspec.set_ylabel("Mean power (log)")
        ax_tspec.grid(True, which="both", linewidth=0.3, alpha=0.5)

        flicker_rgb = cv2.cvtColor(
            cv2.applyColorMap(flicker_img, cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB
        )
        ax_flicker.imshow(flicker_rgb)
        ax_flicker.set_title("Per-pixel Flicker Map (non-DC energy)")
        ax_flicker.axis("off")

        if has_fingerprint:
            fingerprint = temporal["fingerprint"]
            fingerprint_rgb = cv2.cvtColor(
                cv2.applyColorMap(fingerprint["image"], cv2.COLORMAP_INFERNO),
                cv2.COLOR_BGR2RGB,
            )
            ax_finger.imshow(fingerprint_rgb, aspect="equal")

            # Circle the detected high-frequency hot spots (candidate AI
            # upsampling fingerprint peaks).
            peaks = fingerprint.get("peaks", [])
            n_fingerprint_peaks = len(peaks)
            circle_r = FINGERPRINT_SIZE * 0.028
            for px, py in peaks:
                ax_finger.add_patch(
                    plt.Circle(
                        (px, py),
                        radius=circle_r,
                        fill=False,
                        edgecolor="#00e5ff",
                        linewidth=1.4,
                        alpha=0.9,
                    )
                )
            if n_fingerprint_peaks:
                cue = (
                    f"{n_fingerprint_peaks} hot spot"
                    f"{'s' if n_fingerprint_peaks != 1 else ''} circled "
                    f"(possible AI fingerprint)"
                )
            else:
                cue = "no strong peaks"
            ax_finger.set_title(
                f"Fingerprint FFT Amplitude (avg residual, log)\n{cue}"
            )
        else:
            ax_finger.text(
                0.5,
                0.5,
                "fingerprint\nunavailable",
                ha="center",
                va="center",
                fontsize=10,
                color="#868e96",
            )
        ax_finger.axis("off")

    # --- Readings block (formerly the .txt report) ---
    ax_text.axis("off")
    feature_lines = "\n".join(f"{k:<32}: {v:.6f}" for k, v in features.items())
    readings = (
        f"File        : {input_path.name}\n"
        f"Frames used : {frame_count}\n"
        f"Scorer      : {method}\n"
        f"Fingerprint hot spots circled : {n_fingerprint_peaks}\n"
        f"\n"
        f"Features:\n"
        f"{feature_lines}\n"
        f"\n"
        f"Note: transparent heuristic, NOT proof. Compression weakens the signal and\n"
        f"newer generators suppress these artifacts. Compare against a real-footage\n"
        f"baseline from a similar camera/codec."
    )
    ax_text.text(
        0.0,
        1.0,
        readings,
        transform=ax_text.transAxes,
        fontsize=9,
        family="monospace",
        va="top",
        ha="left",
    )

    # --- Score / verdict in bold at the bottom ---
    fig.text(
        0.5,
        0.045,
        f"Synthetic likelihood: {score:.3f}    \u00b7    {verdict}",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=verdict_color,
    )
    fig.text(
        0.5,
        0.018,
        "(0 = real-like, 1 = synthetic-like)",
        ha="center",
        va="center",
        fontsize=9,
        color="#495057",
    )

    fig.suptitle(
        f"Spectral AI-video analysis \u2014 {input_path.name}",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    print(f"Result image: {plot_path}")
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
