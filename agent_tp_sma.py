"""
agent_tp_sma.py - Module socle SMA (Systèmes Multi-Agents) & Optimisation.
Projet 5 : Optimisation multi-agents : essaim, fourmis, immunité, génétique
Rôle : Membre B (Quant / Python)
Semaine 2 : Prototype Python

Ce module fait le pont entre le socle historique du TP SMA et la nouvelle architecture
modulaire (optim_core.py, algos.py).
"""

from typing import Callable, Tuple, List, Dict, Any
import numpy as np

# Ré-exportation des fonctions fondamentales
from optim_core import (
    sphere,
    rastrigin,
    rosenbrock,
    DEFAULT_BOUNDS,
    OPTIMA,
    FONCTIONS_CONTINUES,
    CompteurEvaluations,
    BudgetDepasseException,
    generer_instance_tsp,
    longueur_tour,
    plus_proche_voisin
)

from algos import (
    algo_genetique,
    algo_genetique_tsp,
    pso,
    aco_tsp,
    aco_continu,
    clonalg,
    ALGOS_CONTINUS,
    ALGOS_TSP
)

# Alias pour compatibilité descendante
algo_genetique_continu = algo_genetique
aco = aco_continu

__all__ = [
    "sphere",
    "rastrigin",
    "rosenbrock",
    "DEFAULT_BOUNDS",
    "OPTIMA",
    "FONCTIONS_CONTINUES",
    "CompteurEvaluations",
    "BudgetDepasseException",
    "generer_instance_tsp",
    "longueur_tour",
    "plus_proche_voisin",
    "algo_genetique",
    "algo_genetique_tsp",
    "pso",
    "aco_tsp",
    "aco_continu",
    "aco",
    "clonalg",
    "ALGOS_CONTINUS",
    "ALGOS_TSP"
]
