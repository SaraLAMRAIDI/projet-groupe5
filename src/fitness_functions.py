"""
fitness_functions.py - Benchmark objective functions for optimization algorithms.
Project: Multi-agent Optimization (PSO, ACO, CLONALG, GA)
Role: Member B (Quant / Python)
"""

import numpy as np


def sphere(x):
    """
    Sphere benchmark function (unimodal, convex).
    Formula: f(x) = sum(x_i^2)
    Global optimum: f(0, ..., 0) = 0.0
    Recommended bounds: [-100, 100]
    """
    x = np.asarray(x, dtype=float)
    return float(np.sum(x**2))


def rastrigin(x):
    """
    Rastrigin benchmark function (highly multimodal, challenging for local search).
    Formula: f(x) = 10 * d + sum(x_i^2 - 10 * cos(2 * pi * x_i))
    Global optimum: f(0, ..., 0) = 0.0
    Recommended bounds: [-5.12, 5.12]
    """
    x = np.asarray(x, dtype=float)
    dim = len(x)
    return float(10.0 * dim + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))


def rosenbrock(x):
    """
    Rosenbrock benchmark function (narrow parabolic valley).
    Formula: f(x) = sum(100 * (x_{i+1} - x_i^2)^2 + (x_i - 1)^2)
    Global optimum: f(1, ..., 1) = 0.0
    Recommended bounds: [-30, 30]
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        raise ValueError("Rosenbrock function requires dim >= 2")
    return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1.0)**2))


# Standard search bounds for each continuous function
BOUNDS = {
    "sphere": (-100.0, 100.0),
    "rastrigin": (-5.12, 5.12),
    "rosenbrock": (-30.0, 30.0)
}

# Known global minimum values for reference
OPTIMA = {
    "sphere": 0.0,
    "rastrigin": 0.0,
    "rosenbrock": 0.0
}

# Dispatcher for functions by string name (useful for Member C / LangChain agent)
FUNCTIONS = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock
}


# --- Optional TSP Utilities (for TSP extensions in the project sheet) ---

def generate_tsp_instance(n_cities=20, seed=42, box_size=100.0):
    """
    Generate reproducible 2D coordinates and Euclidean distance matrix for TSP.
    """
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, box_size, size=(n_cities, 2))
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    return coords, dist_matrix


def tsp_fitness(tour, dist_matrix):
    """
    Compute total round-trip distance of a TSP permutation tour.
    """
    tour = np.asarray(tour, dtype=int)
    return float(np.sum(dist_matrix[tour, np.roll(tour, -1)]))


def nearest_neighbor_tsp(dist_matrix, start_city=0):
    """
    Greedy nearest neighbor baseline for TSP (heuristic mentioned in project sheet).
    """
    n = len(dist_matrix)
    unvisited = set(range(n))
    current = start_city
    unvisited.remove(current)
    tour = [current]

    while unvisited:
        next_city = min(unvisited, key=lambda city: dist_matrix[current, city])
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    return tour, tsp_fitness(tour, dist_matrix)

