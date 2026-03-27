"""Validate numerical solvers for LWR, ARZ, and Euler equations.

Generates data with default parameters and checks for garbage:
NaN, Inf, all-zeros, extreme values, negative density/pressure.
"""

import sys
import numpy as np

sys.path.insert(0, "/Users/mchami/ETH/Thesis/saint-venant")

from numerical_solvers.src.lwr import generate_n as lwr_generate_n
from numerical_solvers.src.arz import generate_n as arz_generate_n
from numerical_solvers.src.euler import generate_n as euler_generate_n

# Default grid config from training.yaml
NX = 50
NT = 250
DX = 0.02
DT = 0.004
N_SAMPLES = 100
SEED = 42


def check_array(name, arr, max_value=None, must_be_positive=False):
    """Check an array for garbage data. Returns list of issue strings."""
    issues = []
    a = arr.numpy() if hasattr(arr, "numpy") else np.asarray(arr)

    n = len(a)
    flat = a.reshape(n, -1)

    nan_count = (~np.isfinite(flat)).any(axis=1).sum()
    zero_count = (flat == 0).all(axis=1).sum()

    if nan_count > 0:
        issues.append(f"  NaN/Inf samples: {nan_count}/{n}")
    if zero_count > 0:
        issues.append(f"  All-zero samples: {zero_count}/{n}")

    if max_value is not None:
        extreme = (np.abs(flat) > max_value).any(axis=1).sum()
        if extreme > 0:
            issues.append(f"  Extreme (|val|>{max_value}) samples: {extreme}/{n}")

    if must_be_positive:
        neg = (flat < 0).any(axis=1).sum()
        if neg > 0:
            issues.append(f"  Negative values: {neg}/{n}")

    # Stats
    print(f"  {name}: shape={a.shape}, min={a.min():.6f}, max={a.max():.6f}, "
          f"mean={a.mean():.6f}, std={a.std():.6f}")

    return issues


def validate_lwr():
    print("\n" + "=" * 60)
    print("LWR VALIDATION")
    print("=" * 60)
    issues = []

    for max_steps in [2, 3, 4]:
        print(f"\n--- max_steps={max_steps} ---")
        result = lwr_generate_n(
            n=N_SAMPLES, k=max_steps, nx=NX, nt=NT, dx=DX, dt=DT,
            only_shocks=False, show_progress=False, batch_size=4,
        )
        rho = result["rho"]
        iss = check_array("rho", rho, max_value=1.5, must_be_positive=True)
        if iss:
            issues.extend([f"steps={max_steps}:"] + iss)

    if issues:
        print("\nLWR ISSUES FOUND:")
        for i in issues:
            print(i)
    else:
        print("\nLWR: ALL CLEAN")
    return issues


def validate_arz():
    print("\n" + "=" * 60)
    print("ARZ VALIDATION")
    print("=" * 60)
    issues = []

    for max_steps in [2, 3, 4]:
        print(f"\n--- max_steps={max_steps} ---")
        result = arz_generate_n(
            n=N_SAMPLES, k=max_steps, nx=NX, nt=NT - 1, dx=DX, dt=DT,
            show_progress=False, gamma=1.0, flux_type="hll",
            reconstruction="weno5", bc_type="zero_gradient",
        )
        rho = result["rho"]
        v = result["v"]
        iss_rho = check_array("rho", rho, max_value=10.0, must_be_positive=True)
        iss_v = check_array("v", v, max_value=10.0)
        if iss_rho:
            issues.extend([f"steps={max_steps} rho:"] + iss_rho)
        if iss_v:
            issues.extend([f"steps={max_steps} v:"] + iss_v)

    if issues:
        print("\nARZ ISSUES FOUND:")
        for i in issues:
            print(i)
    else:
        print("\nARZ: ALL CLEAN")
    return issues


def validate_euler():
    print("\n" + "=" * 60)
    print("EULER VALIDATION")
    print("=" * 60)
    issues = []

    for max_steps in [2, 3, 4]:
        print(f"\n--- max_steps={max_steps} ---")
        result = euler_generate_n(
            n=N_SAMPLES, k=max_steps, nx=NX, nt=NT - 1, dx=DX, dt=DT,
            show_progress=False, gamma=1.4, flux_type="hllc",
            reconstruction="weno5", bc_type="extrap",
        )
        rho = result["rho"]
        u = result["u"]
        p = result["p"]
        iss_rho = check_array("rho", rho, max_value=1e4, must_be_positive=True)
        iss_u = check_array("u", u, max_value=1e4)
        iss_p = check_array("p", p, max_value=1e4, must_be_positive=True)
        if iss_rho:
            issues.extend([f"steps={max_steps} rho:"] + iss_rho)
        if iss_u:
            issues.extend([f"steps={max_steps} u:"] + iss_u)
        if iss_p:
            issues.extend([f"steps={max_steps} p:"] + iss_p)

    if issues:
        print("\nEULER ISSUES FOUND:")
        for i in issues:
            print(i)
    else:
        print("\nEULER: ALL CLEAN")
    return issues


if __name__ == "__main__":
    lwr_issues = validate_lwr()
    arz_issues = validate_arz()
    euler_issues = validate_euler()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"LWR:   {'ISSUES' if lwr_issues else 'CLEAN'}")
    print(f"ARZ:   {'ISSUES' if arz_issues else 'CLEAN'}")
    print(f"Euler: {'ISSUES' if euler_issues else 'CLEAN'}")
