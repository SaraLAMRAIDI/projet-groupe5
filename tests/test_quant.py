"""
test_quant.py - Sanity check and validation test suite for Member B.
Validates:
1. Function correctness on Sphere, Rastrigin, Rosenbrock.
2. Exact budget evaluation enforcement (no evaluation exceeds budget).
3. Boundary constraint adherence (np.clip checks).
4. Reproducibility across fixed seeds.
5. Benchmark convergence comparison plot generation.
"""

import numpy as np
import matplotlib.pyplot as plt
from fitness_functions import sphere, rastrigin, rosenbrock, BOUNDS, FUNCTIONS
from optimizers import pso, ga, aco, clonalg, ALGORITHMS


class CountingWrapper:
    """Wraps an objective function to strictly count calls and verify budget adherence."""
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, x):
        self.count += 1
        return self.func(x)


def test_quick_smoke_tests():
    print("=" * 60)
    print("1. RUNNING SMOKE TESTS ON ALL 4 ALGORITHMS")
    print("=" * 60)

    # Sphere (dim=2)
    for name, algo in ALGORITHMS.items():
        pos, fit, hist = algo(sphere, dim=2, bounds=BOUNDS["sphere"], budget=3000, seed=42)
        print(f"[{name:7s}] Sphere dim=2    -> Best Fitness: {fit:.6e} | Final Pos: {np.round(pos, 4)}")
        assert fit < 1.0, f"{name} failed to optimize Sphere dim=2"
        assert len(hist) > 0, f"{name} history is empty"

    # Rastrigin (dim=5)
    print("-" * 60)
    for name, algo in ALGORITHMS.items():
        pos, fit, hist = algo(rastrigin, dim=5, bounds=BOUNDS["rastrigin"], budget=5000, seed=42)
        print(f"[{name:7s}] Rastrigin dim=5 -> Best Fitness: {fit:.6e}")
        assert fit < 50.0, f"{name} failed Rastrigin dim=5"

    # Rosenbrock (dim=2)
    print("-" * 60)
    for name, algo in ALGORITHMS.items():
        pos, fit, hist = algo(rosenbrock, dim=2, bounds=BOUNDS["rosenbrock"], budget=5000, seed=42)
        print(f"[{name:7s}] Rosenbrock dim=2-> Best Fitness: {fit:.6e} | Final Pos: {np.round(pos, 4)}")

    print("\n[OK] All smoke tests passed successfully!\n")


def test_budget_strictness():
    print("=" * 60)
    print("2. TESTING BUDGET COMPLIANCE (Golden Rule: count <= budget)")
    print("=" * 60)

    budgets = [123, 500, 2500]
    for b in budgets:
        for name, algo in ALGORITHMS.items():
            counter = CountingWrapper(sphere)
            algo(counter, dim=5, bounds=BOUNDS["sphere"], budget=b, seed=42)
            assert counter.count <= b, f"FAIL: {name} exceeded budget {b}: {counter.count} calls!"
            print(f"[{name:7s}] Target budget: {b:4d} | Actual evaluations: {counter.count:4d} -> OK")

    print("\n[OK] Budget strictness confirmed for all algorithms!\n")


def test_bounds_compliance():
    print("=" * 60)
    print("3. TESTING BOUNDARY CONSTRAINTS")
    print("=" * 60)

    for name, algo in ALGORITHMS.items():
        lower, upper = BOUNDS["rastrigin"]
        pos, fit, _ = algo(rastrigin, dim=10, bounds=(lower, upper), budget=3000, seed=42)
        assert np.all(pos >= lower - 1e-9) and np.all(pos <= upper + 1e-9), \
            f"FAIL: {name} returned position outside bounds [{lower}, {upper}]"
        print(f"[{name:7s}] Bounds [{lower}, {upper}] fully respected. -> OK")

    print("\n[OK] Boundary constraints confirmed!\n")


def test_reproducibility():
    print("=" * 60)
    print("4. TESTING SEED REPRODUCIBILITY")
    print("=" * 60)

    for name, algo in ALGORITHMS.items():
        pos1, fit1, hist1 = algo(sphere, dim=5, bounds=BOUNDS["sphere"], budget=2000, seed=123)
        pos2, fit2, hist2 = algo(sphere, dim=5, bounds=BOUNDS["sphere"], budget=2000, seed=123)
        assert np.isclose(fit1, fit2), f"FAIL: {name} is not reproducible on identical seed!"
        assert np.allclose(pos1, pos2), f"FAIL: {name} positions differ on identical seed!"
        assert hist1 == hist2, f"FAIL: {name} history differs on identical seed!"
        print(f"[{name:7s}] Exact match across two identical seed=123 runs. -> OK")

    print("\n[OK] Seed reproducibility verified for all algorithms!\n")


def generate_benchmark_plot():
    print("=" * 60)
    print("5. GENERATING CONVERGENCE BENCHMARK PLOT")
    print("=" * 60)

    dim = 5
    budget = 5000
    seed = 42
    results = {}

    plt.figure(figsize=(10, 6))

    for name, algo in ALGORITHMS.items():
        _, fit, hist = algo(rastrigin, dim=dim, bounds=BOUNDS["rastrigin"], budget=budget, seed=seed)
        results[name] = fit
        plt.plot(hist, label=f"{name} (final: {fit:.3f})", linewidth=2)

    plt.title(f"Convergence Comparison on Rastrigin (dim={dim}, budget={budget}, seed={seed})", fontsize=14)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Best Fitness (Log scale)", fontsize=12)
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    output_path = "convergence_rastrigin_dim5.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Plot saved successfully to: {output_path}")


if __name__ == "__main__":
    test_quick_smoke_tests()
    test_budget_strictness()
    test_bounds_compliance()
    test_reproducibility()
    generate_benchmark_plot()
    print("=" * 60)
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 60)

