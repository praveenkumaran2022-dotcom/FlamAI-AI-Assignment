import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# FlamAI AI/R&D assignment
# Model:
# x = t*cos(theta) - exp(M*|t|)*sin(0.3*t)*sin(theta) + X
# y = 42 + t*sin(theta) + exp(M*|t|)*sin(0.3*t)*cos(theta)
#
# theta is in degrees for the optimizer; trigonometric functions use radians.

THETA_BOUNDS = (0.0, 50.0)
M_BOUNDS = (-0.05, 0.05)
X_BOUNDS = (0.0, 100.0)


def load_data(path="xy_data.csv"):
    df = pd.read_csv(path)
    return df["x"].to_numpy(dtype=float), df["y"].to_numpy(dtype=float)


def residuals(params, x, y):
    theta_deg, M, X = params
    theta = np.deg2rad(theta_deg)

    # Rotate/translate the observed points into the curve's local frame.
    u = x - X
    v = y - 42.0

    # Along-curve coordinate: the perpendicular sinusoidal term cancels.
    t = u * np.cos(theta) + v * np.sin(theta)

    # Perpendicular coordinate should equal exp(M|t|) * sin(0.3t).
    q = -u * np.sin(theta) + v * np.cos(theta)

    return q - np.exp(M * np.abs(t)) * np.sin(0.3 * t)


def fit_parameters(x, y, n_starts=25, seed=42):
    rng = np.random.default_rng(seed)
    bounds = (
        [THETA_BOUNDS[0], M_BOUNDS[0], X_BOUNDS[0]],
        [THETA_BOUNDS[1], M_BOUNDS[1], X_BOUNDS[1]],
    )

    starts = [
        np.array([30.0, 0.03, 55.0]),  # useful deterministic start
    ]
    for _ in range(n_starts - 1):
        starts.append(np.array([
            rng.uniform(*THETA_BOUNDS),
            rng.uniform(*M_BOUNDS),
            rng.uniform(*X_BOUNDS),
        ]))

    best = None
    for start in starts:
        result = least_squares(
            residuals, start, args=(x, y),
            bounds=bounds,
            xtol=1e-13, ftol=1e-13, gtol=1e-13,
            max_nfev=10000,
        )
        sse = np.sum(result.fun ** 2)
        if best is None or sse < best[0]:
            best = (sse, result)

    return best[1]


def curve(t, theta_deg, M, X):
    theta = np.deg2rad(theta_deg)
    envelope = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    x = t * np.cos(theta) - envelope * np.sin(theta) + X
    y = 42.0 + t * np.sin(theta) + envelope * np.cos(theta)
    return x, y


def main():
    x, y = load_data()

    result = fit_parameters(x, y)
    theta, M, X = result.x

    print("Estimated parameters")
    print(f"theta = {theta:.10f} degrees")
    print(f"M     = {M:.10f}")
    print(f"X     = {X:.10f}")
    print(f"SSE   = {np.sum(result.fun**2):.12e}")
    print(f"MAE   = {np.mean(np.abs(result.fun)):.12e}")
    print(f"Max |residual| = {np.max(np.abs(result.fun)):.12e}")

    # Uniformly sample the recovered curve over the assignment domain.
    t_uniform = np.linspace(6.0, 60.0, 1500)
    xu, yu = curve(t_uniform, theta, M, X)

    # Plot the supplied points and recovered curve.
    plt.figure(figsize=(9, 6))
    plt.scatter(x, y, s=8, alpha=0.35, label="Given data")
    plt.plot(xu, yu, linewidth=2, label="Recovered curve")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("FlamAI AI Assignment: Parametric Curve Fit")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig("fit.png", dpi=180)
    plt.show()


if __name__ == "__main__":
    main()
