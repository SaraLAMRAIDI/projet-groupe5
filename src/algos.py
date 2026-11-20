"""
algos.py - Métaheuristiques d'optimisation multi-agents (PSO, ACO, CLONALG, Génétique).
Projet 5 : Optimisation multi-agents (SMA)
Rôle : Membre B (Quant / Python)
Semaine 2 : Prototype Python

Tous les algorithmes continus partagent l'interface commune :
    def algo(f, dim, bornes, budget, seed=42, **kwargs) -> (meilleure_sol, meilleur_fitness, historique)
"""

from typing import Callable, Tuple, List, Dict, Any, Optional
import numpy as np
from optim_core import (
    CompteurEvaluations,
    BudgetDepasseException,
    longueur_tour
)


# =============================================================================
# 1. Algorithme Génétique Continu
# =============================================================================

def algo_genetique(
    f: Callable[[np.ndarray], float],
    dim: int,
    bornes: Tuple[float, float],
    budget: int,
    seed: int = 42,
    pop_size: int = 40,
    pc: float = 0.8,
    pm: float = 0.1,
    n_elite: int = 1,
    tournament_size: int = 3,
    **kwargs: Any
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Algorithme Génétique pour optimisation continue dans un hypercube [lower, upper]^dim.
    
    Opérateurs :
        - Sélection par tournoi (taille k=tournament_size).
        - Croisement arithmétique / BLX-alpha (probabilité pc).
        - Mutation gaussienne proportionnelle à l'amplitude des bornes (probabilité pm par gène).
        - Élitisme strict conservant les n_elite meilleures solutions.
        - Arrêt exact au budget d'évaluations via CompteurEvaluations.
        
    Paramètres :
        f : Fonction objectif à minimiser f(x) -> float.
        dim : Dimension du problème.
        bornes : Tuple (min, max) définissant l'espace de recherche.
        budget : Nombre maximal d'évaluations de f.
        seed : Graine pseudo-aléatoire pour la reproductibilité.
        pop_size : Taille de la population (défaut 40).
        pc : Probabilité de croisement (défaut 0.8).
        pm : Probabilité de mutation par variable (défaut 0.1).
        n_elite : Nombre d'élites conservés sans modification (défaut 1).
        tournament_size : Taille du tournoi de sélection (défaut 3).
        
    Retourne :
        (meilleure_solution, meilleur_fitness, historique_par_eval)
    """
    rng = np.random.default_rng(seed)
    compteur = CompteurEvaluations(f, budget)
    lower, upper = bornes
    sigma_mut = 0.1 * (upper - lower)

    # 1. Initialisation uniforme de la population
    pop = rng.uniform(lower, upper, size=(pop_size, dim))
    fitness = np.full(pop_size, np.inf)

    try:
        for i in range(pop_size):
            fitness[i] = compteur(pop[i])
    except BudgetDepasseException:
        return compteur.resultats()

    # Fonction auxiliaire de sélection par tournoi
    def selection_tournoi() -> np.ndarray:
        candidats = rng.choice(pop_size, size=tournament_size, replace=False)
        meilleur_idx = candidats[np.argmin(fitness[candidats])]
        return pop[meilleur_idx].copy()

    # 2. Boucle générationnelle avec arrêt exact au budget
    try:
        while not compteur.est_epuise():
            # Tri des élites actuels
            indices_tries = np.argsort(fitness)
            nouvelle_pop: List[np.ndarray] = [pop[idx].copy() for idx in indices_tries[:n_elite]]
            nouveaux_fits: List[float] = [fitness[idx] for idx in indices_tries[:n_elite]]

            # Reproduction
            while len(nouvelle_pop) < pop_size:
                p1 = selection_tournoi()
                p2 = selection_tournoi()

                # Croisement
                if rng.uniform(0.0, 1.0) < pc:
                    alpha = rng.uniform(-0.1, 1.1, size=dim)
                    c1 = alpha * p1 + (1.0 - alpha) * p2
                    c2 = alpha * p2 + (1.0 - alpha) * p1
                else:
                    c1 = p1.copy()
                    c2 = p2.copy()

                for enfant in (c1, c2):
                    if len(nouvelle_pop) >= pop_size:
                        break

                    # Mutation gaussienne
                    masque = rng.uniform(0.0, 1.0, size=dim) < pm
                    if np.any(masque):
                        enfant[masque] += rng.normal(0.0, sigma_mut, size=np.sum(masque))

                    # Projection dans les bornes
                    enfant = np.clip(enfant, lower, upper)

                    # Évaluation avec contrôle strict
                    fit_enfant = compteur(enfant)
                    nouvelle_pop.append(enfant)
                    nouveaux_fits.append(fit_enfant)

            pop = np.array(nouvelle_pop)
            fitness = np.array(nouveaux_fits)

    except BudgetDepasseException:
        pass

    return compteur.resultats()


# =============================================================================
# 2. Algorithme Génétique pour le TSP (Croisement OX)
# =============================================================================

def croisement_ox(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Croisement d'ordre (Order Crossover - OX) pour permutations.
    Préserve l'ordre relatif et garantit une permutation valide.
    """
    n = len(p1)
    pts = np.sort(rng.choice(n + 1, size=2, replace=False))
    c1, c2 = pts[0], pts[1]

    def ox_one_child(parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
        child = np.full(n, -1, dtype=int)
        # Copie de la sous-séquence
        child[c1:c2] = parent_a[c1:c2]
        elements_presents = set(child[c1:c2])

        # Remplissage circulaire depuis c2 à partir de parent_b
        pos_b = [i % n for i in range(c2, c2 + n)]
        idx_fill = c2 % n

        for idx in pos_b:
            val = parent_b[idx]
            if val not in elements_presents:
                child[idx_fill] = val
                elements_presents.add(val)
                idx_fill = (idx_fill + 1) % n

        # Invariant : chaque ville présente exactement une fois
        assert len(set(child)) == n and len(child) == n, "Échec d'invariant de permutation OX"
        return child

    return ox_one_child(p1, p2), ox_one_child(p2, p1)


def algo_genetique_tsp(
    D: np.ndarray,
    budget: int,
    seed: int = 42,
    pop_size: int = 40,
    pc: float = 0.8,
    pm: float = 0.2,
    n_elite: int = 1,
    tournament_size: int = 3,
    **kwargs: Any
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Algorithme Génétique appliqué au problème du voyageur de commerce (TSP).
    
    Opérateurs :
        - Représentation : permutations de {0, ..., n_villes-1}.
        - Sélection par tournoi (k=tournament_size).
        - Croisement d'ordre (OX).
        - Mutation par échange (swap de deux villes avec probabilité pm).
        - Élitisme strict.
        - Invariant : chaque solution générée est une permutation strictement valide.
        
    Paramètres :
        D : Matrice carrée (n_villes, n_villes) des distances.
        budget : Nombre maximal d'évaluations de la longueur du tour.
        seed : Graine pseudo-aléatoire.
        pop_size : Taille de la population.
        pc : Probabilité de croisement OX.
        pm : Probabilité de mutation par permutation.
        n_elite : Nombre de meilleurs tours préservés.
        tournament_size : Taille du tournoi.
        
    Retourne :
        (meilleur_tour, meilleure_longueur, historique_par_eval)
    """
    rng = np.random.default_rng(seed)
    n_villes = len(D)

    def eval_tour(tour: np.ndarray) -> float:
        return longueur_tour(tour, D)

    compteur = CompteurEvaluations(eval_tour, budget)

    # 1. Initialisation par permutations aléatoires
    pop = [rng.permutation(n_villes) for _ in range(pop_size)]
    fitness = np.full(pop_size, np.inf)

    try:
        for i in range(pop_size):
            # Vérification invariant
            assert len(set(pop[i])) == n_villes, "Permutation initiale invalide."
            fitness[i] = compteur(pop[i])
    except BudgetDepasseException:
        return compteur.resultats()

    def selection_tournoi() -> np.ndarray:
        candidats = rng.choice(pop_size, size=tournament_size, replace=False)
        meilleur_idx = candidats[np.argmin(fitness[candidats])]
        return pop[meilleur_idx].copy()

    # 2. Reproduction
    try:
        while not compteur.est_epuise():
            indices_tries = np.argsort(fitness)
            nouvelle_pop = [pop[idx].copy() for idx in indices_tries[:n_elite]]
            nouveaux_fits = [fitness[idx] for idx in indices_tries[:n_elite]]

            while len(nouvelle_pop) < pop_size:
                p1 = selection_tournoi()
                p2 = selection_tournoi()

                if rng.uniform(0.0, 1.0) < pc:
                    c1, c2 = croisement_ox(p1, p2, rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                for enfant in (c1, c2):
                    if len(nouvelle_pop) >= pop_size:
                        break

                    # Mutation par échange (swap de 2 villes)
                    if rng.uniform(0.0, 1.0) < pm:
                        idx_a, idx_b = rng.choice(n_villes, size=2, replace=False)
                        enfant[idx_a], enfant[idx_b] = enfant[idx_b], enfant[idx_a]

                    # Vérification d'invariant
                    assert len(set(enfant)) == n_villes, "Invariant de permutation violé après mutation."

                    fit_enfant = compteur(enfant)
                    nouvelle_pop.append(enfant)
                    nouveaux_fits.append(fit_enfant)

            pop = nouvelle_pop
            fitness = np.array(nouveaux_fits)

    except BudgetDepasseException:
        pass

    return compteur.resultats()


# =============================================================================
# 3. Optimisation par Essaim Particulaire (PSO)
# =============================================================================

def pso(
    f: Callable[[np.ndarray], float],
    dim: int,
    bornes: Tuple[float, float],
    budget: int,
    seed: int = 42,
    pop_size: int = 50,
    w: float = 0.7,
    c1: float = 1.4,
    c2: float = 1.4,
    **kwargs: Any
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Optimisation par Essaim Particulaire (PSO).
    
    Formules du cours :
        v_i <- w * v_i + c1 * r1 * (pbest_i - x_i) + c2 * r2 * (gbest - x_i)
        v_i <- clip(v_i, -v_max, v_max) avec v_max = 0.2 * (upper - lower)
        x_i <- clip(x_i + v_i, lower, upper)
        
    Comportement théorique validé en cours :
        - w = 0.7 : régime convergent et stable.
        - w = 1.2 : régime divergent avec dispersion des particules.
        
    Retourne :
        (meilleure_position, meilleur_fitness, historique_par_eval)
    """
    rng = np.random.default_rng(seed)
    compteur = CompteurEvaluations(f, budget)
    lower, upper = bornes
    v_max = 0.2 * (upper - lower)

    # 1. Initialisation des particules et vitesses
    pop = rng.uniform(lower, upper, size=(pop_size, dim))
    vel = rng.uniform(-v_max, v_max, size=(pop_size, dim))
    pbest = pop.copy()
    pbest_fit = np.full(pop_size, np.inf)

    try:
        for i in range(pop_size):
            pbest_fit[i] = compteur(pop[i])
    except BudgetDepasseException:
        return compteur.resultats()

    gbest_idx = np.argmin(pbest_fit)
    gbest = pbest[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]

    # 2. Boucle de déplacement de l'essaim
    try:
        while not compteur.est_epuise():
            for i in range(pop_size):
                r1 = rng.uniform(0.0, 1.0, size=dim)
                r2 = rng.uniform(0.0, 1.0, size=dim)

                # Actualisation de la vitesse
                vel[i] = (w * vel[i] 
                          + c1 * r1 * (pbest[i] - pop[i]) 
                          + c2 * r2 * (gbest - pop[i]))

                # Bornage des vitesses
                vel[i] = np.clip(vel[i], -v_max, v_max)

                # Déplacement et bornage spatial
                pop[i] = np.clip(pop[i] + vel[i], lower, upper)

                # Évaluation
                fit = compteur(pop[i])

                # Actualisation pbest et gbest
                if fit < pbest_fit[i]:
                    pbest[i] = pop[i].copy()
                    pbest_fit[i] = fit
                    if fit < gbest_fit:
                        gbest = pop[i].copy()
                        gbest_fit = fit

    except BudgetDepasseException:
        pass

    return compteur.resultats()


# =============================================================================
# 4. Algorithme de Colonie de Fourmis (ACO)
# =============================================================================

def aco_tsp(
    D: np.ndarray,
    budget: int,
    seed: int = 42,
    n_ants: int = 20,
    alpha: float = 1.0,
    beta: float = 2.0,
    rho: float = 0.1,
    Q: float = 1.0,
    **kwargs: Any
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Système de Colonie de Fourmis (ACO - Ant System) pour le TSP.
    
    Formules du cours :
        - Probabilité de transition : p_ij = (tau_ij^alpha * eta_ij^beta) / sum(...)
        - Visibilité heuristique : eta_ij = 1.0 / D_ij
        - Mise à jour des phéromones : tau_ij <- (1 - rho) * tau_ij + sum_k (Q / L_k)
        
    Invariant : chaque fourmi produit un circuit hamiltonien (tour fermé) valide.
    """
    rng = np.random.default_rng(seed)
    n = len(D)

    def eval_tour(tour: np.ndarray) -> float:
        return longueur_tour(tour, D)

    compteur = CompteurEvaluations(eval_tour, budget)

    # Initialisation de la matrice de visibilité eta = 1 / D (avec protection diagonale)
    with np.errstate(divide='ignore'):
        eta = np.where(D > 0, 1.0 / D, 0.0)

    # Initialisation des traces de phéromones tau
    tau0 = 1.0 / (n * np.mean(D[D > 0]))
    tau = np.full((n, n), tau0)

    try:
        while not compteur.est_epuise():
            tours_generation: List[np.ndarray] = []
            longueurs_generation: List[float] = []

            for _ in range(n_ants):
                if compteur.est_epuise():
                    break

                # Construction du tour pour une fourmi
                ville_depart = rng.integers(0, n)
                visitees = [ville_depart]
                non_visitees = set(range(n)) - {ville_depart}

                courante = ville_depart
                while non_visitees:
                    candidats = list(non_visitees)
                    probas = np.array([
                        (tau[courante, j] ** alpha) * (eta[courante, j] ** beta)
                        for j in candidats
                    ], dtype=float)

                    somme = np.sum(probas)
                    if somme == 0.0 or not np.isfinite(somme):
                        p = np.full(len(candidats), 1.0 / len(candidats))
                    else:
                        p = probas / somme

                    suivante = rng.choice(candidats, p=p)
                    visitees.append(suivante)
                    non_visitees.remove(suivante)
                    courante = suivante

                tour_ant = np.array(visitees, dtype=int)
                assert len(set(tour_ant)) == n, "Tour TSP incomplet ou invalide généré par une fourmi."

                longueur = compteur(tour_ant)
                tours_generation.append(tour_ant)
                longueurs_generation.append(longueur)

            # Évaporation globale des phéromones
            tau = (1.0 - rho) * tau

            # Dépôt des phéromones par les fourmis de la génération
            for tour_ant, longueur in zip(tours_generation, longueurs_generation):
                delta_tau = Q / longueur
                for i in range(n):
                    u = tour_ant[i]
                    v = tour_ant[(i + 1) % n]
                    tau[u, v] += delta_tau
                    tau[v, u] += delta_tau

            # Borne inférieure pour éviter l'atrophie complète
            tau = np.maximum(tau, 1e-12)

    except BudgetDepasseException:
        pass

    return compteur.resultats()


def aco_continu(
    f: Callable[[np.ndarray], float],
    dim: int,
    bornes: Tuple[float, float],
    budget: int,
    seed: int = 42,
    pop_size: int = 50,
    q: float = 0.5,
    xi: float = 0.85,
    **kwargs: Any
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Colonie de Fourmis pour domaines continus (ACO_R de Socha & Dorigo).
    Archive de solutions ordonnées pondérées par un noyau gaussien.
    """
    rng = np.random.default_rng(seed)
    compteur = CompteurEvaluations(f, budget)
    lower, upper = bornes
    k_archive = pop_size

    # Initialisation de l'archive
    archive = rng.uniform(lower, upper, size=(k_archive, dim))
    fitness = np.full(k_archive, np.inf)

    try:
        for i in range(k_archive):
            fitness[i] = compteur(archive[i])
    except BudgetDepasseException:
        return compteur.resultats()

    # Tri initial
    ordre = np.argsort(fitness)
    archive = archive[ordre]
    fitness = fitness[ordre]

    # Poids gaussiens des rangs
    rangs = np.arange(k_archive)
    poids = (1.0 / (q * k_archive * np.sqrt(2.0 * np.pi))) * np.exp(- (rangs**2) / (2.0 * (q**2) * (k_archive**2)))
    probs = poids / np.sum(poids)

    try:
        while not compteur.est_epuise():
            nouveaux = []
            nouveaux_fits = []

            for _ in range(k_archive):
                if compteur.est_epuise():
                    break

                guide_idx = rng.choice(k_archive, p=probs)
                guide = archive[guide_idx]

                if k_archive > 1:
                    sigma = xi * np.sum(np.abs(archive - guide), axis=0) / (k_archive - 1)
                else:
                    sigma = np.full(dim, 0.1 * (upper - lower))
                sigma = np.maximum(sigma, 1e-12)

                ant_pos = rng.normal(guide, sigma)
                ant_pos = np.clip(ant_pos, lower, upper)

                fit = compteur(ant_pos)
                nouveaux.append(ant_pos)
                nouveaux_fits.append(fit)

            if nouveaux:
                pop_comb = np.vstack([archive, np.array(nouveaux)])
                fits_comb = np.concatenate([fitness, np.array(nouveaux_fits)])
                meilleurs_indices = np.argsort(fits_comb)[:k_archive]
                archive = pop_comb[meilleurs_indices]
                fitness = fits_comb[meilleurs_indices]

    except BudgetDepasseException:
        pass

    return compteur.resultats()


# =============================================================================
# 5. Algorithme de Sélection Clonale (CLONALG)
# =============================================================================

def clonalg(
    f: Callable[[np.ndarray], float],
    dim: int,
    bornes: Tuple[float, float],
    budget: int,
    seed: int = 42,
    pop_size: int = 50,
    clone_factor: float = 0.2,
    beta: float = 1.0,
    replace_rate: float = 0.1,
    **kwargs: Any
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Algorithme de Sélection Clonale (CLONALG - Système Immunitaire Artificiel).
    
    Mécanismes :
        1. Les anticorps représentent les points candidats dans l'espace.
        2. Tri par affinité (fitness décroissant).
        3. Nombre de clones proportionnel au rang d'affinité.
        4. Hypermutation inversement proportionnelle à l'affinité :
           - Meilleur anticorps : mutation très fine (exploitation locale).
           - Anticorps moins performants : mutation large (exploration globale).
        5. Remplacement des moins performants par réédition des récepteurs (nouvelles solutions aléatoires).
    """
    rng = np.random.default_rng(seed)
    compteur = CompteurEvaluations(f, budget)
    lower, upper = bornes
    etendue = upper - lower

    # 1. Répertoire initial d'anticorps
    pop = rng.uniform(lower, upper, size=(pop_size, dim))
    fitness = np.full(pop_size, np.inf)

    try:
        for i in range(pop_size):
            fitness[i] = compteur(pop[i])
    except BudgetDepasseException:
        return compteur.resultats()

    try:
        while not compteur.est_epuise():
            ordre = np.argsort(fitness)
            pop = pop[ordre]
            fitness = fitness[ordre]

            for i in range(pop_size):
                if compteur.est_epuise():
                    break

                rang = i + 1
                n_clones = max(1, int(round((beta * pop_size * clone_factor) / rang)))
                rang_norm = i / max(1, pop_size - 1)
                sigma = (0.01 + 0.19 * rang_norm) * etendue

                meilleur_clone = pop[i].copy()
                meilleur_clone_fit = fitness[i]

                for _ in range(n_clones):
                    if compteur.est_epuise():
                        break

                    clone = pop[i] + rng.normal(0.0, sigma, size=dim)
                    clone = np.clip(clone, lower, upper)
                    fit_clone = compteur(clone)

                    if fit_clone < meilleur_clone_fit:
                        meilleur_clone = clone.copy()
                        meilleur_clone_fit = fit_clone

                if meilleur_clone_fit < fitness[i]:
                    pop[i] = meilleur_clone
                    fitness[i] = meilleur_clone_fit

            # Édition des récepteurs : remplacement aléatoire de la fraction la moins adaptée
            n_remplace = max(1, int(round(replace_rate * pop_size)))
            for k in range(pop_size - n_remplace, pop_size):
                if compteur.est_epuise():
                    break
                nouvel_anticorps = rng.uniform(lower, upper, size=dim)
                fit_nouvel = compteur(nouvel_anticorps)
                pop[k] = nouvel_anticorps
                fitness[k] = fit_nouvel

    except BudgetDepasseException:
        pass

    return compteur.resultats()


# =============================================================================
# Dictionnaires de Routage et Adaptateurs
# =============================================================================

# Dictionnaire standard des algorithmes continus
ALGOS_CONTINUS: Dict[str, Callable] = {
    "GA": algo_genetique,
    "PSO": pso,
    "ACO": aco_continu,
    "CLONALG": clonalg
}

# Dictionnaire des algorithmes sur le TSP
ALGOS_TSP: Dict[str, Callable] = {
    "GA_TSP": algo_genetique_tsp,
    "ACO_TSP": aco_tsp
}
