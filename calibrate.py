"""Calibrate the AI-video detector from labelled clips.

Fits a standardised, L2-regularised logistic-regression model over the spectral
features extracted by ``video_spectra.py`` and writes ``calibration.json`` next
to it. Once that file exists, ``video_spectra.py`` uses the fitted model instead
of the hand-tuned ramps.

Anti-overfitting safeguards (important on small datasets):
  * Features are standardised (z-scored).
  * Logistic regression is L2-regularised; the strength can be raised on tiny
    sets via ``--l2``.
  * Honest generalisation is estimated with leave-one-out cross-validation
    (LOO-CV), with per-fold standardisation so no test sample leaks into
    training statistics.
  * ``--num-features K`` keeps only the K most-separating features, which is the
    single most effective way to avoid overfitting when you have few clips.
  * A loud warning is printed when the sample count is small relative to the
    number of features.

Usage:
    python calibrate.py --real <paths...> --synthetic <paths...>
    python calibrate.py --real real_dir --synthetic ai_dir --num-features 3 --l2 5

Each path may be a directory (all ``*.mp4`` inside are used) or a single file.
Extracted features are cached in ``.feature_cache.json`` so re-runs are fast.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from video_spectra import (
    CALIBRATION_PATH,
    FEATURE_ORDER,
    process_video,
)

CACHE_PATH = Path(__file__).with_name(".feature_cache.json")


def collect_videos(paths: list[str]) -> list[Path]:
    """Expand a list of files/directories into a sorted list of .mp4 files."""
    videos: list[Path] = []
    for raw in paths:
        p = Path(raw).expanduser().resolve()
        if p.is_dir():
            videos.extend(sorted(p.glob("*.mp4")))
        elif p.is_file() and p.suffix.lower() == ".mp4":
            videos.append(p)
        else:
            print(f"Warning: skipping '{raw}' (not an .mp4 file or directory).")
    # Exclude the generated spectrum videos so they are never used as inputs.
    return [v for v in videos if not v.stem.endswith("_spectra")]


def load_cache() -> dict:
    if CACHE_PATH.is_file():
        try:
            return json.loads(CACHE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_cache(cache: dict) -> None:
    try:
        CACHE_PATH.write_text(json.dumps(cache, indent=2))
    except OSError as exc:
        print(f"Warning: could not write feature cache ({exc}).")


def features_for(video: Path, cache: dict) -> dict[str, float] | None:
    """Return the feature dict for a video, using/refreshing the cache."""
    key = str(video)
    try:
        mtime = video.stat().st_mtime
    except OSError:
        mtime = 0.0

    cached = cache.get(key)
    if cached and cached.get("mtime") == mtime:
        return cached["features"]

    print(f"Extracting features: {video.name}")
    result = process_video(video, spectra_path=None, progress=False)
    if result is None:
        print(f"  (no frames; skipped)")
        return None
    features = {k: float(v) for k, v in result["features"].items()}
    cache[key] = {"mtime": mtime, "features": features}
    return features


def build_dataset(
    real: list[Path], synthetic: list[Path], cache: dict
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return ``(X, y, names)`` where y=1 means synthetic."""
    rows: list[list[float]] = []
    labels: list[int] = []
    names: list[str] = []

    for label, group in ((0, real), (1, synthetic)):
        for video in group:
            feats = features_for(video, cache)
            if feats is None:
                continue
            rows.append([feats.get(f, 0.0) for f in FEATURE_ORDER])
            labels.append(label)
            names.append(video.name)

    return np.array(rows, dtype=np.float64), np.array(labels, dtype=np.float64), names


def select_features(X: np.ndarray, y: np.ndarray, k: int) -> list[int]:
    """Rank features by absolute standardised mean difference (Cohen's d)."""
    real = X[y == 0]
    syn = X[y == 1]
    pooled_std = np.sqrt((real.var(axis=0) + syn.var(axis=0)) / 2.0) + 1e-12
    cohen_d = np.abs(real.mean(axis=0) - syn.mean(axis=0)) / pooled_std
    order = np.argsort(cohen_d)[::-1]
    return sorted(order[:k].tolist())


def fit_logreg(
    X: np.ndarray, y: np.ndarray, l2: float, iters: int = 5000, lr: float = 0.3
) -> tuple[np.ndarray, float]:
    """Fit standardised-input logistic regression with L2 (no penalty on bias)."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        err = p - y
        grad_w = X.T @ err / n + (l2 / n) * w
        grad_b = float(err.mean())
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def standardise(
    X: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return (X - mean) / np.where(std == 0, 1.0, std)


def leave_one_out_accuracy(X: np.ndarray, y: np.ndarray, l2: float) -> float:
    """LOO-CV accuracy with per-fold standardisation (no leakage)."""
    n = len(y)
    if n < 2:
        return float("nan")
    correct = 0
    for i in range(n):
        mask = np.arange(n) != i
        X_tr, y_tr = X[mask], y[mask]
        mean = X_tr.mean(axis=0)
        std = X_tr.std(axis=0)
        Xs = standardise(X_tr, mean, std)
        w, b = fit_logreg(Xs, y_tr, l2)
        xi = standardise(X[i : i + 1], mean, std)
        p = 1.0 / (1.0 + np.exp(-(xi @ w + b)[0]))
        if int(p >= 0.5) == int(y[i]):
            correct += 1
    return correct / n


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real", nargs="+", required=True, help="Real video files or folders."
    )
    parser.add_argument(
        "--synthetic", nargs="+", required=True, help="AI video files or folders."
    )
    parser.add_argument(
        "--l2", type=float, default=2.0, help="L2 regularisation strength."
    )
    parser.add_argument(
        "--num-features",
        type=int,
        default=0,
        help="Keep only the K most-separating features (0 = use all).",
    )
    args = parser.parse_args(argv)

    real = collect_videos(args.real)
    synthetic = collect_videos(args.synthetic)
    if not real or not synthetic:
        print("Error: need at least one real and one synthetic clip.")
        return 1

    print(f"Real clips     : {len(real)}")
    print(f"Synthetic clips: {len(synthetic)}")

    cache = load_cache()
    X, y, names = build_dataset(real, synthetic, cache)
    save_cache(cache)

    n = len(y)
    if n < 2 or len(np.unique(y)) < 2:
        print("Error: need clips from both classes with readable frames.")
        return 1

    # Feature selection (optional but recommended for small datasets).
    if args.num_features and 0 < args.num_features < len(FEATURE_ORDER):
        idx = select_features(X, y, args.num_features)
    else:
        idx = list(range(len(FEATURE_ORDER)))
    selected = [FEATURE_ORDER[i] for i in idx]
    Xsel = X[:, idx]

    if n < 3 * len(selected):
        print(
            f"\nWARNING: only {n} clips for {len(selected)} features. The model is"
            f"\nlikely overfit. Add more labelled clips, raise --l2, or lower"
            f"\n--num-features. Trust the LOO-CV accuracy below, not the fit.\n"
        )

    # Fit the final model on all data (standardised on the full set).
    mean = Xsel.mean(axis=0)
    std = Xsel.std(axis=0)
    Xs = standardise(Xsel, mean, std)
    w, b = fit_logreg(Xs, y, args.l2)

    loo = leave_one_out_accuracy(Xsel, y, args.l2)

    model = {
        "type": "logreg",
        "features": selected,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "coef": w.tolist(),
        "intercept": float(b),
        "l2": args.l2,
        "n_samples": int(n),
        "n_real": int((y == 0).sum()),
        "n_synthetic": int((y == 1).sum()),
        "loo_accuracy": float(loo),
    }
    CALIBRATION_PATH.write_text(json.dumps(model, indent=2))

    # Report.
    print("\n=== Calibration summary ===")
    print(f"Features used : {', '.join(selected)}")
    print(f"L2 strength   : {args.l2}")
    print(f"Samples       : {n} ({model['n_real']} real, {model['n_synthetic']} synthetic)")
    print(f"LOO-CV accuracy: {loo:.3f}  (honest generalisation estimate)")
    print("Coefficients (standardised; +ve pushes toward synthetic):")
    for name, coef in zip(selected, w):
        print(f"  {name:<32}: {coef:+.3f}")
    print(f"  intercept                       : {b:+.3f}")

    # Per-clip fitted scores (for sanity only - this is training data).
    p = 1.0 / (1.0 + np.exp(-(Xs @ w + b)))
    print("\nFitted scores on training clips (not a generalisation estimate):")
    for name, label, prob in zip(names, y, p):
        tag = "synthetic" if label == 1 else "real"
        print(f"  {name:<28} [{tag:>9}] -> {prob:.3f}")

    print(f"\nSaved model to {CALIBRATION_PATH}")
    print("video_spectra.py will now use this model automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
