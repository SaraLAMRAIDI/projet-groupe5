"""
test_optim_core.py - Tests unitaires pour optim_core.py.
"""

import pytest
import numpy as np
from optim_core import (
    sphere,
    rastrigin,
    rosenbrock,
    DEFAULT_BOUNDS,
    OPTIMA,
    CompteurEvaluations,
    BudgetDepasseException,
    generer_instance_tsp,
    longueur_tour,
    plus_proche_voisin
)


def test_optima_fonctions_continues():
    """Vérifie que les fonctions de test valent exactement 0 à leur optimum global."""
    for dim in (2, 5, 10):
        # Sphere : min en (0, ..., 0)
        x_sphere = np.zeros(dim)
        assert np.isclose(sphere(x_sphere), OPTIMA["sphere"]), f"Échec Sphere dim={dim}"

        # Rastrigin : min en (0, ..., 0)
        x_rastrigin = np.zeros(dim)
        assert np.isclose(rastrigin(x_rastrigin), OPTIMA["rastrigin"]), f"Échec Rastrigin dim={dim}"

        # Rosenbrock : min en (1, ..., 1)
        x_rosenbrock = np.ones(dim)
        assert np.isclose(rosenbrock(x_rosenbrock), OPTIMA["rosenbrock"]), f"Échec Rosenbrock dim={dim}"


def test_compteur_evaluations_respect_budget():
    """Vérifie que le compteur compte exactement et bloque au-delà du budget."""
    budget = 25
    compteur = CompteurEvaluations(sphere, budget)

    for i in range(budget):
        val = compteur(np.array([float(i), 1.0]))
        assert isinstance(val, float)

    assert compteur.nb_evals == budget
    assert compteur.est_epuise()
    assert len(compteur.historique) == budget

    # Un appel supplémentaire doit lever BudgetDepasseException
    with pytest.raises(BudgetDepasseException):
        compteur(np.array([0.0, 0.0]))


def test_compteur_historique_monotone_decroissant():
    """L'historique du meilleur fitness doit être monotone décroissant ou constant."""
    compteur = CompteurEvaluations(sphere, budget=10)
    points = [
        np.array([5.0, 5.0]),  # fit = 50
        np.array([2.0, 2.0]),  # fit = 8
        np.array([4.0, 4.0]),  # fit = 32 (ne doit pas augmenter le meilleur)
        np.array([1.0, 0.0]),  # fit = 1
    ]
    for p in points:
        compteur(p)

    hist = compteur.historique
    for i in range(len(hist) - 1):
        assert hist[i] >= hist[i + 1], "L'historique du meilleur fitness doit être décroissant."


def test_tsp_instance_et_longueur():
    """Vérifie la génération d'instance TSP et le calcul de longueur de tour."""
    coords, D = generer_instance_tsp(n_villes=20, seed=42)
    assert coords.shape == (20, 2)
    assert D.shape == (20, 20)
    assert np.all(D >= 0.0)
    assert np.allclose(np.diag(D), 0.0)

    # Tour valide
    tour = list(range(20))
    cout = longueur_tour(tour, D)
    assert cout > 0.0

    # Tour invalide (doublon)
    tour_invalide = [0] * 20
    with pytest.raises(ValueError):
        longueur_tour(tour_invalide, D)


def test_plus_proche_voisin():
    """Vérifie que l'heuristique du plus proche voisin renvoie une permutation valide."""
    _, D = generer_instance_tsp(n_villes=20, seed=42)
    tour, cout = plus_proche_voisin(D, depart=0)
    assert len(tour) == 20
    assert len(set(tour)) == 20
    assert cout > 0.0
