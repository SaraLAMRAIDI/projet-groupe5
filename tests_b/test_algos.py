"""
test_algos.py - Tests unitaires pour les algorithmes (algos.py).
"""

import pytest
import numpy as np
from optim_core import sphere, rastrigin, DEFAULT_BOUNDS, generer_instance_tsp
from algos import (
    algo_genetique,
    algo_genetique_tsp,
    pso,
    aco_tsp,
    aco_continu,
    clonalg,
    ALGOS_CONTINUS
)


@pytest.mark.parametrize("nom_algo, algo_func", ALGOS_CONTINUS.items())
@pytest.mark.parametrize("budget", [35, 120, 300])
def test_budget_exact_algos_continus(nom_algo, algo_func, budget):
    """Vérifie que chaque algorithme s'arrête exactement au budget spécifié."""
    sol, fit, hist = algo_func(
        sphere,
        dim=3,
        bornes=DEFAULT_BOUNDS["sphere"],
        budget=budget,
        seed=42
    )
    assert len(hist) == budget, f"{nom_algo} n'a pas respecté le budget exact ({len(hist)} != {budget})"
    assert fit == hist[-1], f"{nom_algo} fitness final différent de l'historique"


@pytest.mark.parametrize("nom_algo, algo_func", ALGOS_CONTINUS.items())
def test_bornes_respectees(nom_algo, algo_func):
    """Vérifie que les solutions trouvées restent strictement dans l'intervalle des bornes."""
    lower, upper = DEFAULT_BOUNDS["rastrigin"]
    sol, fit, _ = algo_func(
        rastrigin,
        dim=5,
        bornes=(lower, upper),
        budget=500,
        seed=42
    )
    assert np.all(sol >= lower - 1e-9), f"{nom_algo} solution inférieure à lower"
    assert np.all(sol <= upper + 1e-9), f"{nom_algo} solution supérieure à upper"


@pytest.mark.parametrize("nom_algo, algo_func", ALGOS_CONTINUS.items())
def test_reproductibilite_graine(nom_algo, algo_func):
    """Vérifie qu'une même graine produit un résultat strictement identique."""
    sol1, fit1, hist1 = algo_func(sphere, dim=4, bornes=DEFAULT_BOUNDS["sphere"], budget=400, seed=99)
    sol2, fit2, hist2 = algo_func(sphere, dim=4, bornes=DEFAULT_BOUNDS["sphere"], budget=400, seed=99)

    assert np.isclose(fit1, fit2), f"{nom_algo} fitness non reproductible"
    assert np.allclose(sol1, sol2), f"{nom_algo} position non reproductible"
    assert hist1 == hist2, f"{nom_algo} historique non reproductible"


def test_algo_genetique_tsp_invariant_permutation():
    """Vérifie l'invariant de permutation et le respect du budget pour le GA TSP."""
    _, D = generer_instance_tsp(n_villes=20, seed=42)
    budget = 500
    tour, cout, hist = algo_genetique_tsp(D, budget=budget, seed=123)

    # Invariants
    assert len(tour) == 20, "Taille du tour incorrecte"
    assert len(set(tour)) == 20, "Villes dupliquées dans le tour (invariant violé)"
    assert set(tour) == set(range(20)), "Toutes les villes doivent être visitées"
    assert cout > 0.0, "Coût nul ou négatif"
    assert len(hist) == budget, f"Budget non respecté : {len(hist)} != {budget}"

    # Reproductibilité
    tour2, cout2, _ = algo_genetique_tsp(D, budget=budget, seed=123)
    assert np.array_equal(tour, tour2)
    assert np.isclose(cout, cout2)


def test_aco_tsp_invariant_permutation():
    """Vérifie l'invariant de permutation et le respect du budget pour ACO TSP."""
    _, D = generer_instance_tsp(n_villes=20, seed=42)
    budget = 400
    tour, cout, hist = aco_tsp(D, budget=budget, seed=123)

    assert len(tour) == 20
    assert len(set(tour)) == 20
    assert set(tour) == set(range(20))
    assert cout > 0.0
    assert len(hist) == budget

    tour2, cout2, _ = aco_tsp(D, budget=budget, seed=123)
    assert np.array_equal(tour, tour2)
    assert np.isclose(cout, cout2)


def test_pso_cours_w_convergence_divergence():
    """
    Vérification de l'hypothèse du cours :
    - w = 0.7 converge vers l'optimum.
    - w = 1.2 diverge ou présente un fitness nettement plus dégradé.
    """
    # TODO: valeur de référence du cours (à fournir par le Membre A)
    dim = 5
    budget = 2500
    seed = 42

    _, fit_convergent, _ = pso(sphere, dim=dim, bornes=DEFAULT_BOUNDS["sphere"], budget=budget, seed=seed, w=0.7)
    _, fit_divergent, _ = pso(sphere, dim=dim, bornes=DEFAULT_BOUNDS["sphere"], budget=budget, seed=seed, w=1.2)

    # w = 0.7 doit donner une valeur beaucoup plus proche de 0 que w = 1.2
    assert fit_convergent < fit_divergent, (
        f"L'inertie w=0.7 ({fit_convergent}) devrait mieux converger que w=1.2 ({fit_divergent})"
    )
    assert fit_convergent < 1.0, f"w=0.7 n'a pas convergé : {fit_convergent}"
