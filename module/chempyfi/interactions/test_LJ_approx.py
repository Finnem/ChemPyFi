import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- 1. Lennard-Jones potential --------------------------------
epsilon = 1.0
sigma = 1.0

def lj(r, epsilon=1.0, sigma=1.0):
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

r = np.linspace(0.8, 3.0, 500)
V = lj(r, epsilon, sigma)
V_abs = np.abs(V)

# --- 2. Exclude region 0.9σ–1.1σ -------------------------------
mask = (r < 0.9) | (r > 1.1)
r_fit = r[mask]
V_abs_fit = V_abs[mask]

# --- 3. Two-power model ----------------------------------------
def two_power(r, A, n, B, m):
    return A / r**n + B / r**m

p0     = (1.0, 12.0, 1.0, 6.0)
bounds = ([0.0, 4.0, 0.0, 2.0],   # lower bounds
          [100.0, 20.0, 100.0, 12.0])  # upper bounds

# --- 4. Fit using masked data ----------------------------------
popt_two, _ = curve_fit(two_power, r_fit, V_abs_fit, p0=p0,
                        bounds=bounds, maxfev=10000)
A, n, B, m = popt_two
print("Two-power fit parameters (excluding 0.9–1.1 σ):")
print(f"  A = {A:.3f}, n = {n:.3f}")
print(f"  B = {B:.3f}, m = {m:.3f}")

# Evaluate fit on full range for plotting
V_two = two_power(r, *popt_two)

# --- 5. Plot ----------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(r, V_abs, label="|Lennard-Jones| (magnitude)",
         color="royalblue", linewidth=2, alpha=0.6)
plt.plot(r, V_two, "--", label=f"Two-power fit (excluding 0.9–1.1σ)\n"
                               f"A={A:.2f}, n={n:.2f}, B={B:.2f}, m={m:.2f}",
         color="darkorange", linewidth=3)
plt.axvline(2**(1/6)*sigma, color="gray", linestyle="--", linewidth=1.5,
            label=r"$r_{min}$")
plt.axvspan(0.9, 1.1, color="lightgray", alpha=0.3,
            label="excluded region")

plt.xlabel("Distance r (σ units)")
plt.ylabel("|Potential| (|V(r)|/ε)")
plt.title("Two-power fit to |Lennard-Jones| Potential (0.9–1.1σ excluded)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("test_LJ_approx.png")
