"""
simulation.py

Monte Carlo validation of concentration inequalities
for Binomial variables / G(n,p) degree distribution.

Generates:
1) figures/semilog_tail_degree.png
2) figures/max_degree_union_bound.png
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt


# ===============================
# KL Divergence (Bernoulli)
# ===============================
def kl_divergence(q: float, p: float) -> float:
    """KL(q || p) for Bernoulli distributions."""
    if q <= 0.0 or q >= 1.0:
        return float("inf")
    return q * math.log(q / p) + (1 - q) * math.log((1 - q) / (1 - p))


# ===============================
# Chebyshev Upper Tail Bound
# Event: deg >= (1+delta) * E[deg]
# ===============================
def chebyshev_bound(n: int, p: float, delta: float) -> float:
    N = n - 1
    mu = N * p
    var = N * p * (1 - p)
    t = delta * mu  # deviation from mean
    if t <= 0:
        return 1.0
    bound = var / (t ** 2)
    return min(1.0, bound)


# ===============================
# Chernoff / KL Bound
# Event: deg >= (1+delta) * E[deg]
# ===============================
def chernoff_bound(n: int, p: float, delta: float) -> float:
    N = n - 1
    q = (1 + delta) * p
    if q >= 1.0:
        return 0.0
    D = kl_divergence(q, p)
    bound = math.exp(-N * D)
    return min(1.0, bound)


# ===============================
# Monte Carlo Simulation
# ===============================
def simulate_degree_tail(n: int, p: float, delta: float, trials: int = 200000):
    """
    Estimate P(deg(1) >= (1+delta)E[deg(1)]) where deg(1) ~ Bin(n-1, p).
    Returns: (estimate, ci_low, ci_high)
    """
    N = n - 1
    threshold = math.ceil((1 + delta) * N * p)

    samples = np.random.binomial(N, p, size=trials)
    hits = float(np.mean(samples >= threshold))

    # 95% CI (normal approximation)
    se = math.sqrt(max(hits * (1 - hits), 0.0) / trials)
    ci_low = max(0.0, hits - 1.96 * se)
    ci_high = min(1.0, hits + 1.96 * se)

    return hits, ci_low, ci_high


# ===============================
# Main Execution
# ===============================
def main():
    # Reproducibility
    np.random.seed(7)

    p = 0.1
    delta = 0.3
    trials = 200000

    n_values = list(range(80, 600, 40))

    empirical, ci_low, ci_high = [], [], []
    cheb, cher = [], []
    union_bound = []

    for n in n_values:
        emp, lo, hi = simulate_degree_tail(n, p, delta, trials)

        c_cheb = chebyshev_bound(n, p, delta)
        c_cher = chernoff_bound(n, p, delta)
        ub = min(1.0, n * c_cher)  # union bound over vertices

        empirical.append(emp)
        ci_low.append(lo)
        ci_high.append(hi)
        cheb.append(c_cheb)
        cher.append(c_cher)
        union_bound.append(ub)

        print(
            f"n={n:3d} | emp={emp:.3e} | Cheb={c_cheb:.3e} | Cher={c_cher:.3e} | n*Cher={ub:.3e}"
        )

    os.makedirs("figures", exist_ok=True)

    # Small floor to avoid log(0) issues in semilogy + fill_between
    eps_floor = 1e-15
    emp_plot = np.maximum(empirical, eps_floor)
    lo_plot = np.maximum(ci_low, eps_floor)
    hi_plot = np.maximum(ci_high, eps_floor)
    cheb_plot = np.maximum(cheb, eps_floor)
    cher_plot = np.maximum(cher, eps_floor)
    ub_plot = np.maximum(union_bound, eps_floor)

    # =====================================
    # Figure 1: Single-vertex tail
    # =====================================
    plt.figure(figsize=(8, 5))
    plt.semilogy(n_values, emp_plot, "o-", label="Empirical")
    plt.fill_between(n_values, lo_plot, hi_plot, alpha=0.2)

    plt.semilogy(n_values, cheb_plot, "--", label="Chebyshev")
    plt.semilogy(n_values, cher_plot, "--", label="Chernoff/KL")

    plt.xlabel("n")
    plt.ylabel("Tail Probability (log scale)")
    plt.title(f"Degree Upper Tail in G(n,p), p={p}, delta={delta}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/semilog_tail_degree.png", dpi=300)
    plt.close()

    # =====================================
    # Figure 2: Max-degree union bound
    # =====================================
    plt.figure(figsize=(8, 5))
    plt.semilogy(n_values, ub_plot, "o-", label="Union Bound (n × Chernoff/KL)")
    plt.xlabel("n")
    plt.ylabel("Upper Bound (log scale)")
    plt.title(f"Max Degree Tail Bound in G(n,p), p={p}, delta={delta}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/max_degree_union_bound.png", dpi=300)
    plt.close()

    print("\nFigures generated successfully in ./figures/")


if __name__ == "__main__":
    main()
