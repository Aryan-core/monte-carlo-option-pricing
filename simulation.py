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
def kl_divergence(q, p):
    """KL(q || p) for Bernoulli distributions."""
    if q <= 0 or q >= 1:
        return float("inf")
    return q * math.log(q / p) + (1 - q) * math.log((1 - q) / (1 - p))


# ===============================
# Chebyshev Upper Tail Bound
# ===============================
def chebyshev_bound(n, p, delta):
    N = n - 1
    mu = N * p
    var = N * p * (1 - p)
    t = delta * mu
    return var / (t ** 2)


# ===============================
# Chernoff / KL Bound
# ===============================
def chernoff_bound(n, p, delta):
    N = n - 1
    q = (1 + delta) * p
    if q >= 1:
        return 0
    D = kl_divergence(q, p)
    return math.exp(-N * D)


# ===============================
# Monte Carlo Simulation
# ===============================
def simulate_degree_tail(n, p, delta, trials=200000):
    """
    Estimate P(deg(1) >= (1+delta)E[deg])
    """
    N = n - 1
    threshold = math.ceil((1 + delta) * N * p)

    samples = np.random.binomial(N, p, size=trials)
    hits = np.mean(samples >= threshold)

    # 95% CI (normal approximation)
    se = math.sqrt(hits * (1 - hits) / trials)
    ci_low = max(0, hits - 1.96 * se)
    ci_high = min(1, hits + 1.96 * se)

    return hits, ci_low, ci_high


# ===============================
# Main Execution
# ===============================
def main():

    p = 0.1
    delta = 0.3
    trials = 200000

    n_values = range(80, 600, 40)

    empirical = []
    ci_low = []
    ci_high = []
    cheb = []
    cher = []
    union_bound = []

    for n in n_values:
        emp, lo, hi = simulate_degree_tail(n, p, delta, trials)

        empirical.append(emp)
        ci_low.append(lo)
        ci_high.append(hi)

        cheb.append(chebyshev_bound(n, p, delta))
        cher.append(chernoff_bound(n, p, delta))

        # Union bound for max degree
        ub = n * chernoff_bound(n, p, delta)
        union_bound.append(min(1, ub))

    os.makedirs("figures", exist_ok=True)

    # =====================================
    # Figure 1: Single-vertex tail
    # =====================================
    plt.figure(figsize=(8, 5))
    plt.semilogy(n_values, empirical, 'o-', label="Empirical")
    plt.fill_between(n_values, ci_low, ci_high, alpha=0.2)

    plt.semilogy(n_values, cheb, '--', label="Chebyshev")
    plt.semilogy(n_values, cher, '--', label="Chernoff/KL")

    plt.xlabel("n")
    plt.ylabel("Tail Probability (log scale)")
    plt.title("Degree Upper Tail in G(n,p)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/semilog_tail_degree.png", dpi=300)
    plt.close()

    # =====================================
    # Figure 2: Max-degree union bound
    # =====================================
    plt.figure(figsize=(8, 5))
    plt.semilogy(n_values, union_bound, 'o-', label="Union Bound + Chernoff")
    plt.xlabel("n")
    plt.ylabel("Upper Bound (log scale)")
    plt.title("Maximum Degree Tail Bound in G(n,p)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/max_degree_union_bound.png", dpi=300)
    plt.close()

    print("Figures generated successfully.")


if __name__ == "__main__":
    main()
