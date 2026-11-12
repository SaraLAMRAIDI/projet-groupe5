import numpy as np
from scipy.stats import mannwhitneyu
from langchain.tools import tool   # sinon : from langchain_core.tools import tool

from optimizers import ALGORITHMS
from fitness_functions import FUNCTIONS, BOUNDS

# Plafonds de securite (regle n°5)
MAX_BUDGET = 20000
MAX_GRAINES = 30
MAX_DIM = 50


def _valider(fonction, dim, budget, nb_graines):
    fonction = fonction.lower()
    if fonction not in FUNCTIONS:
        return None, {"erreur": f"fonction inconnue, choisir parmi {list(FUNCTIONS)}"}
    return (fonction, min(dim, MAX_DIM), min(budget, MAX_BUDGET),
            min(nb_graines, MAX_GRAINES)), None


def _scores(algo, fonction, dim, budget, nb_graines):
    f, b = FUNCTIONS[fonction], BOUNDS[fonction]
    return [float(ALGORITHMS[algo](f, dim=dim, bounds=b, budget=budget, seed=s)[1])
            for s in range(nb_graines)]


@tool
def comparer_optimiseurs(fonction: str, dim: int, budget: int, nb_graines: int) -> dict:
    """Compare PSO, ACO, CLONALG et GA sur une fonction test (sphere, rastrigin
    ou rosenbrock) avec le meme budget et plusieurs graines. Retourne moyenne,
    mediane, pire cas et ecart-type de la meilleure valeur par algorithme."""
    params, erreur = _valider(fonction, dim, budget, nb_graines)
    if erreur:
        return erreur
    fonction, dim, budget, nb_graines = params
    resultats = {}
    for algo in ALGORITHMS:
        s = np.array(_scores(algo, fonction, dim, budget, nb_graines))
        resultats[algo] = {"moyenne": float(s.mean()), "mediane": float(np.median(s)),
                           "pire": float(s.max()), "ecart_type": float(s.std())}
    return {"fonction": fonction, "dim": dim, "budget": budget,
            "nb_graines": nb_graines, "resultats": resultats}


@tool
def mann_whitney_optimiseurs(fonction: str, dim: int, budget: int, nb_graines: int,
                      algo_a: str, algo_b: str) -> dict:
    """Test de Mann-Whitney entre deux algorithmes (PSO, ACO, CLONALG, GA) sur une
    fonction test. Dit si la difference de performance est significative (p < 0.05)."""
    params, erreur = _valider(fonction, dim, budget, nb_graines)
    if erreur:
        return erreur
    algo_a, algo_b = algo_a.upper(), algo_b.upper()
    if algo_a not in ALGORITHMS or algo_b not in ALGORITHMS:
        return {"erreur": f"algorithme inconnu, choisir parmi {list(ALGORITHMS)}"}
    fonction, dim, budget, nb_graines = params
    a = _scores(algo_a, fonction, dim, budget, nb_graines)
    b = _scores(algo_b, fonction, dim, budget, nb_graines)
    try:
        p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:        # valeurs toutes identiques
        p = 1.0
    return {"algo_a": algo_a, "algo_b": algo_b, "mediane_a": float(np.median(a)),
            "mediane_b": float(np.median(b)), "p_value": p, "significatif": p < 0.05}