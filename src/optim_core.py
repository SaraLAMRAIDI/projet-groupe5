"""
optim_core.py - Module central des fonctions de test et gestion du budget d'évaluation.
Projet 5 : Optimisation multi-agents (SMA)
Rôle : Membre B (Quant / Python)
Semaine 2 : Prototype Python
"""

from typing import Callable, Tuple, List, Dict, Any, Optional
import numpy as np


class BudgetDepasseException(Exception):
    """Exception levée en interne dès que le quota d'évaluations autorisées est atteint."""
    pass


class CompteurEvaluations:
    """
    Enveloppe (wrapper) autour d'une fonction objectif pour contrôler strictement
    le budget d'évaluations et tracer la dynamique de convergence.
    
    Attributs :
        fonction : Fonction objectif f(x) -> float à minimiser.
        budget : Nombre maximal d'évaluations permises (int).
        nb_evals : Compteur d'appels effectifs à la fonction (int).
        meilleur_x : Meilleur point évalué jusqu'à présent (np.ndarray ou list).
        meilleur_fitness : Meilleure valeur de fitness rencontrée (float).
        historique : Liste des meilleures valeurs de fitness enregistrées après chaque évaluation.
    """
    def __init__(self, fonction: Callable[[Any], float], budget: int):
        if budget <= 0:
            raise ValueError(f"Le budget doit être strictement positif, reçu: {budget}")
        self.fonction = fonction
        self.budget = int(budget)
        self.nb_evals = 0
        self.meilleur_x: Optional[np.ndarray] = None
        self.meilleur_fitness: float = float("inf")
        self.historique: List[float] = []

    def __call__(self, x: Any) -> float:
        """
        Évalue le candidat x, incrémente le compteur et met à jour le meilleur score.
        Lève BudgetDepasseException si le budget est déjà épuisé.
        """
        if self.nb_evals >= self.budget:
            raise BudgetDepasseException("Budget maximal d'évaluations atteint.")

        # Exécution de l'évaluation
        score = float(self.fonction(x))
        self.nb_evals += 1

        # Mise à jour du meilleur point
        if score < self.meilleur_fitness or self.meilleur_x is None:
            self.meilleur_fitness = score
            if isinstance(x, np.ndarray):
                self.meilleur_x = x.copy()
            elif isinstance(x, (list, tuple)):
                self.meilleur_x = np.array(x).copy()
            else:
                self.meilleur_x = x

        self.historique.append(self.meilleur_fitness)
        return score

    def est_epuise(self) -> bool:
        """Vérifie si le quota maximal d'évaluations a été atteint."""
        return self.nb_evals >= self.budget

    def resultats(self) -> Tuple[np.ndarray, float, List[float]]:
        """
        Retourne le triplet standard : (meilleure_solution, meilleur_fitness, historique).
        """
        return self.meilleur_x, self.meilleur_fitness, self.historique


# =============================================================================
# Fonctions de Test Continues
# =============================================================================

def sphere(x: np.ndarray) -> float:
    """
    Fonction Sphère (unimodale, convexe, séparable).
    Formule : f(x) = sum(x_i^2)
    Minimum global : f(0, ..., 0) = 0.0
    Bornes usuelles : [-100.0, 100.0]
    """
    arr = np.asarray(x, dtype=float)
    return float(np.sum(arr**2))


def rastrigin(x: np.ndarray) -> float:
    """
    Fonction de Rastrigin (hautement multimodale avec de nombreux minima locaux).
    Formule : f(x) = 10 * d + sum(x_i^2 - 10 * cos(2 * pi * x_i))
    Minimum global : f(0, ..., 0) = 0.0
    Bornes usuelles : [-5.12, 5.12]
    """
    arr = np.asarray(x, dtype=float)
    d = len(arr)
    return float(10.0 * d + np.sum(arr**2 - 10.0 * np.cos(2.0 * np.pi * arr)))


def rosenbrock(x: np.ndarray) -> float:
    """
    Fonction de Rosenbrock (« vallée de la banane », non séparable).
    Formule : f(x) = sum(100 * (x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    Minimum global : f(1, ..., 1) = 0.0
    Bornes usuelles : [-30.0, 30.0]
    """
    arr = np.asarray(x, dtype=float)
    if len(arr) < 2:
        raise ValueError("La fonction de Rosenbrock requiert une dimension >= 2.")
    return float(np.sum(100.0 * (arr[1:] - arr[:-1]**2)**2 + (1.0 - arr[:-1])**2))


# Bornes par défaut documentées
DEFAULT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "sphere": (-100.0, 100.0),
    "rastrigin": (-5.12, 5.12),
    "rosenbrock": (-30.0, 30.0)
}

# Valeurs théoriques des minima globaux
OPTIMA: Dict[str, float] = {
    "sphere": 0.0,
    "rastrigin": 0.0,
    "rosenbrock": 0.0
}

# Dictionnaire d'accès aux fonctions
FONCTIONS_CONTINUES: Dict[str, Callable[[np.ndarray], float]] = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock
}


# =============================================================================
# Utilitaires pour le Voyageur de Commerce (TSP)
# =============================================================================

def generer_instance_tsp(n_villes: int = 20, seed: int = 42, boite: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Génère les coordonnées 2D reproductibles et la matrice de distances euclidiennes pour le TSP.
    
    Paramètres :
        n_villes : Nombre de villes à visiter.
        seed : Graine du générateur pseudo-aléatoire.
        boite : Dimension de l'espace [0, boite] x [0, boite].
        
    Retourne :
        coords : Tableau numpy (n_villes, 2) des positions des villes.
        D : Matrice symétrique (n_villes, n_villes) des distances euclidiennes.
    """
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, boite, size=(n_villes, 2))
    delta = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    D = np.sqrt(np.sum(delta**2, axis=-1))
    return coords, D


def longueur_tour(tour: Any, D: np.ndarray) -> float:
    """
    Calcule la longueur totale du tour fermé d'une permutation de villes.
    
    Paramètres :
        tour : Permutation ordonnée des indices des villes (tableau ou liste).
        D : Matrice de distances.
        
    Retourne :
        Longueur totale du circuit fermé (float).
    """
    arr = np.asarray(tour, dtype=int)
    n = len(arr)
    # Vérification d'intégrité de la permutation
    if len(np.unique(arr)) != n or np.min(arr) != 0 or np.max(arr) != n - 1:
        raise ValueError("Le tour doit être une permutation valide de 0 à n-1.")
    return float(np.sum(D[arr, np.roll(arr, -1)]))


def plus_proche_voisin(D: np.ndarray, depart: int = 0) -> Tuple[List[int], float]:
    """
    Heuristique gloutonne du plus proche voisin pour le TSP (solution de référence).
    """
    n = len(D)
    non_visitees = set(range(n))
    actuel = depart
    non_visitees.remove(actuel)
    tour = [actuel]

    while non_visitees:
        suivant = min(non_visitees, key=lambda ville: D[actuel, ville])
        tour.append(suivant)
        non_visitees.remove(suivant)
        actuel = suivant

    cout = longueur_tour(tour, D)
    return tour, cout
