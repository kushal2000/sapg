import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import correlate, correlation_lags, detrend
from typing import Optional


def estimate_time_shift_irregular(
    time_A: np.ndarray,
    y_A: np.ndarray,
    time_B: np.ndarray,
    y_B: np.ndarray,
    resample_dt: Optional[float] = None,
    use_derivative: bool = True,   # <-- default True for joint targets
) -> float:
    """
    Estimate time shift Δt such that:
        time_B + Δt aligns y_B with y_A
    """

    time_A = np.asarray(time_A, dtype=float)
    time_B = np.asarray(time_B, dtype=float)
    y_A = np.asarray(y_A, dtype=float)
    y_B = np.asarray(y_B, dtype=float)

    # Sort by time
    idxA = np.argsort(time_A)
    idxB = np.argsort(time_B)
    time_A, y_A = time_A[idxA], y_A[idxA]
    time_B, y_B = time_B[idxB], y_B[idxB]

    # Choose dt
    if resample_dt is None:
        dtA = np.median(np.diff(time_A))
        dtB = np.median(np.diff(time_B))
        resample_dt = min(dtA, dtB)

    # Use only overlap (avoid extrapolation artifacts)
    t_min = max(time_A.min(), time_B.min())
    t_max = min(time_A.max(), time_B.max())
    if t_max <= t_min:
        raise ValueError("Time ranges do not overlap")

    t = np.arange(t_min, t_max + 0.5 * resample_dt, resample_dt)

    fA = interp1d(time_A, y_A, kind="linear", bounds_error=False, fill_value=np.nan)
    fB = interp1d(time_B, y_B, kind="linear", bounds_error=False, fill_value=np.nan)
    yA = fA(t)
    yB = fB(t)

    m = np.isfinite(yA) & np.isfinite(yB)
    yA = yA[m]
    yB = yB[m]

    if len(yA) < 10:
        raise ValueError("Not enough overlapping samples after resampling")

    # For joint targets: use derivative so ramps/steps create strong features
    if use_derivative:
        yA = np.gradient(yA, resample_dt)
        yB = np.gradient(yB, resample_dt)

    # Remove DC + normalize (per-signal)
    yA = detrend(yA, type="constant")
    yB = detrend(yB, type="constant")
    yA /= (np.std(yA) + 1e-12)
    yB /= (np.std(yB) + 1e-12)

    corr = correlate(yA, yB, mode="full", method="fft")
    lags = correlation_lags(len(yA), len(yB), mode="full") * resample_dt

    i = int(np.argmax(corr))
    lag = float(lags[i])

    # Sub-sample quadratic refinement
    if 0 < i < len(corr) - 1:
        xs = lags[i - 1 : i + 2]
        ys = corr[i - 1 : i + 2]
        a, b, _ = np.polyfit(xs, ys, 2)
        if abs(a) > 1e-20:
            lag = float(-b / (2 * a))

    return lag



def underlying_signal(t: np.ndarray) -> np.ndarray:
    """
    A signal with *time-localized structure* + multiple frequencies:
    - two Gaussian bumps (strong “anchors” for correlation)
    - a sinusoid + a chirp-ish term for extra texture
    """
    bump1 = 1.2 * np.exp(-0.5 * ((t - 2.3) / 0.10) ** 2)
    bump2 = -0.9 * np.exp(-0.5 * ((t - 6.8) / 0.18) ** 2)
    sin1 = 0.35 * np.sin(2 * np.pi * 0.9 * t)
    chirp = 0.25 * np.sin(2 * np.pi * (0.2 * t + 0.08 * t**2))  # frequency increases with t
    return bump1 + bump2 + sin1 + chirp


def main():
    rng = np.random.default_rng(0)

    OFFSET = 1.7  # seconds: B’s timestamps are shifted later by +OFFSET

    # Irregular sampling for A (jittered around ~80 Hz)
    tA = np.cumsum(rng.uniform(0.010, 0.017, size=1400))
    tA = tA[tA <= 10.0]
    yA = underlying_signal(tA)
    yA = yA + 0.05 * rng.standard_normal(size=yA.shape)  # noise

    # Irregular sampling for B (different rate, different jitter), and shifted time axis
    tB_true = np.cumsum(rng.uniform(0.015, 0.030, size=800))
    tB_true = tB_true[tB_true <= 10.0]
    tB = tB_true + OFFSET  # this is the only “clock” shift we introduce

    # B sees the *same physical signal* at true times tB_true, but with slight gain + offset + noise
    yB = 1.15 * underlying_signal(tB_true) + 0.02  # small gain/DC mismatch
    yB = yB + 0.06 * rng.standard_normal(size=yB.shape)

    # Plot raw signals (misaligned x-axes)
    plt.figure()
    plt.plot(tA, yA, ".", markersize=2, label="A (irregular)")
    plt.plot(tB, yB, ".", markersize=2, label=f"B (irregular, time shifted by +{OFFSET}s)")
    plt.legend()
    plt.title("Raw signals (misaligned)")
    plt.show()

    # Estimate time shift
    dt_shift = estimate_time_shift_irregular(tA, yA, tB, yB)
    print(f"Estimated dt_shift = {dt_shift:.4f} s")
    print(f"Expected dt_shift  ≈ {-OFFSET:.4f} s")

    # Plot aligned
    tB_aligned = tB + dt_shift
    plt.figure()
    plt.plot(tA, yA, ".", markersize=2, label="A")
    plt.plot(tB_aligned, yB, ".", markersize=2, label="B shifted back")
    plt.legend()
    plt.title("After alignment")
    plt.show()


if __name__ == "__main__":
    main()
