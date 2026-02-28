Monte Carlo Option Pricing

Risk-Neutral Simulation of Geometric Brownian Motion

This project implements a Monte Carlo engine for pricing European call options under the Black–Scholes framework.

The objective is to combine mathematical derivation, stochastic modeling, and numerical simulation to understand:
	•	Risk-neutral pricing
	•	Geometric Brownian Motion (GBM)
	•	Monte Carlo convergence behavior
	•	Variance reduction techniques
	•	Comparison with closed-form Black–Scholes solution

Project Structure

monte-carlo-option-pricing/
│
├── report.pdf          # Full mathematical write-up (LaTeX compiled)
├── report.tex          # LaTeX source code
├── simulation.py   # Monte Carlo implementation
└── README.md

Mathematical Framework

Under the risk-neutral measure, the stock price follows the stochastic differential equation:

dSₜ = r Sₜ dt + σ Sₜ dWₜ

where:
	•	r = risk-free rate
	•	σ = volatility
	•	Wₜ = standard Brownian motion

The solution is:

S_T = S₀ exp((r − ½σ²)T + σ√T Z)

where Z ~ N(0,1).

⸻

Option Pricing Formula

A European call option price is:

C = e^(−rT) E[(S_T − K)⁺]

The Monte Carlo estimator is:

Ĉ_N = e^(−rT) (1/N) Σ (S_T^(i) − K)⁺

By the Law of Large Numbers:

Ĉ_N → C

By the Central Limit Theorem:

√N (Ĉ_N − C) → Normal(0, variance)

Convergence rate:

O(N^(-1/2))

⸻

Variance Reduction

This project implements:

1️⃣ Antithetic Variates

Instead of sampling Z alone, we also simulate −Z.

This preserves unbiasedness and reduces estimator variance.

⸻

Simulation Features

The script:
	•	Simulates terminal stock prices
	•	Computes Monte Carlo option estimate
	•	Computes Black–Scholes closed-form price
	•	Reports:
	•	Absolute pricing error
	•	Standard error
	•	Generates:
	•	Convergence vs N plot
	•	Monte Carlo vs Black–Scholes comparison

Run with: python simulation.py


References
	•	Shreve – Stochastic Calculus for Finance II
	•	Glasserman – Monte Carlo Methods in Financial Engineering
	•	Hull – Options, Futures, and Other Derivatives

⸻

Author
Aryan Khan
Drexel University – Computer Engineering
