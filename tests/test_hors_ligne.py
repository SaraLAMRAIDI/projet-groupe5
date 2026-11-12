from outils_optim import comparer_optimiseurs, mann_whitney_optimiseurs

def test_comparer():
    r = comparer_optimiseurs.invoke({"fonction": "rastrigin", "dim": 5,
                                     "budget": 1000, "nb_graines": 3})
    assert set(r["resultats"]) == {"PSO", "ACO", "CLONALG", "GA"}

def test_plafond():
    r = comparer_optimiseurs.invoke({"fonction": "sphere", "dim": 2,
                                     "budget": 10**9, "nb_graines": 2})
    assert r["budget"] == 20000

def test_mann_whitney():
    r = mann_whitney_optimiseurs.invoke({"fonction": "sphere", "dim": 2, "budget": 1000,
                                         "nb_graines": 5, "algo_a": "PSO", "algo_b": "GA"})
    assert 0.0 <= r["p_value"] <= 1.0