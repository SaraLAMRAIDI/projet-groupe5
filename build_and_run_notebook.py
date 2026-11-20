"""
build_and_run_notebook.py - Génère et exécute notebook_s2.ipynb pour valider l'exécution sans erreur.
"""

import json
import os
import sys

notebook_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Projet 5 : Optimisation Multi-Agents (Banc d'Essai)\n",
            "## Prototype Python — Semaine 2\n",
            "**Auteur :** Membre B (Quant / Python)\n",
            "\n",
            "Ce notebook valide de bout en bout l'ensemble des composants développés pour la Semaine 2 :\n",
            "1. **Fonctions de test** (Sphere, Rastrigin, Rosenbrock) et vérification de leurs optima théoriques.\n",
            "2. **Compteur d'évaluations** et gestion stricte du budget (`CompteurEvaluations`).\n",
            "3. **Algorithme Génétique continu** (sélection par tournoi, croisement arithmétique, mutation gaussienne, élitisme).\n",
            "4. **Algorithme Génétique TSP** (croisement OX, mutation échange, invariant de permutation).\n",
            "5. **Métaheuristiques du cours** (PSO, ACO, CLONALG).\n",
            "6. **Validation du comportement théorique du PSO** ($w=0.7$ vs $w=1.2$).\n",
            "7. **Graphe de convergence comparatif** (médiane et quartiles sur 10 graines, échelle logarithmique)."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 1. Initialisation et Importations"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import pandas as pd\n",
            "\n",
            "from optim_core import (\n",
            "    sphere,\n",
            "    rastrigin,\n",
            "    rosenbrock,\n",
            "    DEFAULT_BOUNDS,\n",
            "    OPTIMA,\n",
            "    CompteurEvaluations,\n",
            "    BudgetDepasseException,\n",
            "    generer_instance_tsp,\n",
            "    longueur_tour,\n",
            "    plus_proche_voisin\n",
            ")\n",
            "\n",
            "from algos import (\n",
            "    algo_genetique,\n",
            "    algo_genetique_tsp,\n",
            "    pso,\n",
            "    aco_continu,\n",
            "    aco_tsp,\n",
            "    clonalg,\n",
            "    ALGOS_CONTINUS,\n",
            "    ALGOS_TSP\n",
            ")\n",
            "\n",
            "SEED_GLOBALE = 42\n",
            "print(\"[OK] Modules importés avec succès.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2. Fonctions de Test & Vérification des Minima Globaux"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Vérification que les fonctions valent 0 à leur optimum théorique\n",
            "for dim in [2, 5, 10]:\n",
            "    v_sphere = sphere(np.zeros(dim))\n",
            "    v_rastrigin = rastrigin(np.zeros(dim))\n",
            "    v_rosenbrock = rosenbrock(np.ones(dim))\n",
            "    print(f\"Dimension {dim:2d} -> Sphere: {v_sphere:.1e} | Rastrigin: {v_rastrigin:.1e} | Rosenbrock: {v_rosenbrock:.1e}\")\n",
            "    assert np.isclose(v_sphere, 0.0)\n",
            "    assert np.isclose(v_rastrigin, 0.0)\n",
            "    assert np.isclose(v_rosenbrock, 0.0)\n",
            "\n",
            "print(\"[OK] Vérification des optima globaux réussie (tous égaux à 0.0).\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 3. Contrôle Strict du Budget d'Évaluations (`CompteurEvaluations`)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "test_budget = 50\n",
            "compteur = CompteurEvaluations(sphere, budget=test_budget)\n",
            "\n",
            "for i in range(test_budget):\n",
            "    compteur(np.array([1.0, float(i)]))\n",
            "\n",
            "assert compteur.nb_evals == test_budget\n",
            "assert compteur.est_epuise()\n",
            "\n",
            "try:\n",
            "    compteur(np.array([0.0, 0.0]))\n",
            "    raise AssertionError(\"Le compteur aurait dû lever BudgetDepasseException!\")\n",
            "except BudgetDepasseException:\n",
            "    print(f\"[OK] Le compteur a bloqué exactement après {compteur.nb_evals} évaluations.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4. Algorithme Génétique Continu"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "dim = 5\n",
            "budget = 3000\n",
            "sol_ga, fit_ga, hist_ga = algo_genetique(\n",
            "    sphere,\n",
            "    dim=dim,\n",
            "    bornes=DEFAULT_BOUNDS[\"sphere\"],\n",
            "    budget=budget,\n",
            "    seed=SEED_GLOBALE,\n",
            "    pop_size=40,\n",
            "    pc=0.8,\n",
            "    pm=0.1\n",
            ")\n",
            "\n",
            "print(f\"GA sur Sphere (dim={dim}) : Meilleur Fitness = {fit_ga:.6e}\")\n",
            "print(f\"Nombre d'évaluations effectuées = {len(hist_ga)} (Budget alloué = {budget})\")\n",
            "assert len(hist_ga) == budget"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 5. Algorithme Génétique pour le Voyageur de Commerce (TSP)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Génération d'une instance TSP à 20 villes (graine fixée)\n",
            "coords_tsp, D_tsp = generer_instance_tsp(n_villes=20, seed=SEED_GLOBALE)\n",
            "\n",
            "# Heuristique du plus proche voisin (référence de base du cours)\n",
            "tour_ppv, dist_ppv = plus_proche_voisin(D_tsp, depart=0)\n",
            "print(f\"Référence Plus Proche Voisin (20 villes) : Distance = {dist_ppv:.2f}\")\n",
            "\n",
            "# Optimisation par GA TSP (croisement OX et mutation par échange)\n",
            "budget_tsp = 4000\n",
            "tour_ga, dist_ga, hist_ga_tsp = algo_genetique_tsp(D_tsp, budget=budget_tsp, seed=SEED_GLOBALE)\n",
            "\n",
            "print(f\"GA TSP (croisement OX) : Distance = {dist_ga:.2f} (Evals: {len(hist_ga_tsp)})\")\n",
            "assert len(set(tour_ga)) == 20, \"Invariant violé : villes dupliquées ou omises !\"\n",
            "assert dist_ga < dist_ppv * 1.5, \"Le GA devrait produire un tour compétitif.\""
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 6. Métaheuristiques du Cours (PSO, ACO Continu & TSP, CLONALG)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Test comparatif rapide sur Sphere dim=5, budget=3000\n",
            "results_s2 = {}\n",
            "for nom, algo in ALGOS_CONTINUS.items():\n",
            "    sol, fit, hist = algo(\n",
            "        sphere,\n",
            "        dim=5,\n",
            "        bornes=DEFAULT_BOUNDS[\"sphere\"],\n",
            "        budget=3000,\n",
            "        seed=SEED_GLOBALE\n",
            "    )\n",
            "    results_s2[nom] = fit\n",
            "    print(f\"[{nom:7s}] Sphere dim=5 -> Fitness = {fit:.6e} | Evals = {len(hist)}\")\n",
            "\n",
            "# Test ACO TSP\n",
            "tour_aco, dist_aco, hist_aco_tsp = aco_tsp(D_tsp, budget=budget_tsp, seed=SEED_GLOBALE)\n",
            "print(f\"[ACO TSP] Distance tour = {dist_aco:.2f} (Evals = {len(hist_aco_tsp)})\")\n",
            "assert len(set(tour_aco)) == 20"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 7. Validation Théorique du Cours : Inertie du PSO ($w=0.7$ vs $w=1.2$)\n",
            "Le cours indique que $w = 0.7$ converge vers l'optimum alors que $w = 1.2$ diverge."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# TODO: valeur de référence du cours (à fournir par le Membre A)\n",
            "_, fit_w07, hist_w07 = pso(sphere, dim=5, bornes=DEFAULT_BOUNDS[\"sphere\"], budget=2500, seed=SEED_GLOBALE, w=0.7)\n",
            "_, fit_w12, hist_w12 = pso(sphere, dim=5, bornes=DEFAULT_BOUNDS[\"sphere\"], budget=2500, seed=SEED_GLOBALE, w=1.2)\n",
            "\n",
            "print(f\"PSO w=0.7 (convergent) : Fitness final = {fit_w07:.6e}\")\n",
            "print(f\"PSO w=1.2 (divergent)  : Fitness final = {fit_w12:.6e}\")\n",
            "assert fit_w07 < fit_w12\n",
            "\n",
            "plt.figure(figsize=(8, 4.5))\n",
            "plt.plot(hist_w07, label=\"PSO w=0.7 (convergent)\", color=\"green\", lw=2)\n",
            "plt.plot(hist_w12, label=\"PSO w=1.2 (divergent)\", color=\"red\", lw=2)\n",
            "plt.yscale(\"log\")\n",
            "plt.xlabel(\"Nombre d'évaluations\")\n",
            "plt.ylabel(\"Fitness (échelle log)\")\n",
            "plt.title(\"Comportement de l'inertie PSO : Convergence (w=0.7) vs Divergence (w=1.2)\")\n",
            "plt.grid(True, linestyle=\"--\", alpha=0.6)\n",
            "plt.legend()\n",
            "plt.tight_layout()\n",
            "plt.savefig(\"pso_w_convergence_divergence.png\", dpi=150)\n",
            "plt.close()\n",
            "print(\"[OK] Graphique sauvegardé sous 'pso_w_convergence_divergence.png'.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 8. Graphe de Convergence Comparatif (Médiane et Quartiles sur 10 Graines)\n",
            "Comparaison rigoureuse sur la fonction Sphère ($d=5$, budget=3000) sur 10 graines distinctes.\n",
            "Affichage des courbes médianes avec bande interquartile (25e - 75e percentile)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "graines = list(range(40, 50))  # 10 graines\n",
            "dim_bench = 5\n",
            "budget_bench = 3000\n",
            "donnees_algos = {nom: [] for nom in ALGOS_CONTINUS.keys()}\n",
            "\n",
            "print(f\"Exécution du protocole sur {len(graines)} graines...\")\n",
            "for seed in graines:\n",
            "    for nom, algo_func in ALGOS_CONTINUS.items():\n",
            "        _, _, hist = algo_func(\n",
            "            sphere,\n",
            "            dim=dim_bench,\n",
            "            bornes=DEFAULT_BOUNDS[\"sphere\"],\n",
            "            budget=budget_bench,\n",
            "            seed=seed\n",
            "        )\n",
            "        donnees_algos[nom].append(hist)\n",
            "\n",
            "# Tracé graphique\n",
            "plt.figure(figsize=(10, 6))\n",
            "couleurs = {\"GA\": \"#1f77b4\", \"PSO\": \"#2ca02c\", \"ACO\": \"#ff7f0e\", \"CLONALG\": \"#d62728\"}\n",
            "evals_x = np.arange(1, budget_bench + 1)\n",
            "\n",
            "for nom, series in donnees_algos.items():\n",
            "    mat = np.array(series)  # Forme (10, budget_bench)\n",
            "    mediane = np.median(mat, axis=0)\n",
            "    q25 = np.percentile(mat, 25, axis=0)\n",
            "    q75 = np.percentile(mat, 75, axis=0)\n",
            "    \n",
            "    # Protection échelle log pour éviter log(0)\n",
            "    mediane = np.maximum(mediane, 1e-30)\n",
            "    q25 = np.maximum(q25, 1e-30)\n",
            "    q75 = np.maximum(q75, 1e-30)\n",
            "    \n",
            "    c = couleurs.get(nom, \"gray\")\n",
            "    plt.plot(evals_x, mediane, label=f\"{nom} (médiane: {mediane[-1]:.1e})\", color=c, lw=2)\n",
            "    plt.fill_between(evals_x, q25, q75, color=c, alpha=0.15)\n",
            "\n",
            "plt.yscale(\"log\")\n",
            "plt.xlabel(\"Nombre d'évaluations de f\", fontsize=12)\n",
            "plt.ylabel(\"Meilleur Fitness (échelle log)\", fontsize=12)\n",
            "plt.title(f\"Convergence médiane avec quartiles [25%, 75%] sur Sphere (dim={dim_bench}, 10 graines)\", fontsize=13)\n",
            "plt.grid(True, which=\"both\", linestyle=\"--\", alpha=0.5)\n",
            "plt.legend(fontsize=11)\n",
            "plt.tight_layout()\n",
            "\n",
            "output_png = \"convergence_sphere_dim5_medianes.png\"\n",
            "plt.savefig(output_png, dpi=200)\n",
            "plt.close()\n",
            "print(f\"[OK] Graphe de convergence sauvegardé sous '{output_png}'.\")"
        ]
    }
]

notebook_dict = {
    "cells": notebook_cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.11"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Écriture du fichier notebook_s2.ipynb
notebook_path = "notebook_s2.ipynb"
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook_dict, f, indent=2, ensure_ascii=False)

print(f"Fichier {notebook_path} généré avec succès.")

# Test d'exécution sans erreur de tous les blocs de code du notebook
print("Test d'exécution de tous les blocs du notebook de haut en bas...")
global_env = {}
code_cells = [cell for cell in notebook_cells if cell["cell_type"] == "code"]
for idx, cell in enumerate(code_cells, start=1):
    code = "".join(cell["source"])
    print(f"Exécution cellule {idx}/{len(code_cells)}...")
    exec(code, global_env)

print("TOUTES LES CELLULES DU NOTEBOOK SE SONT EXÉCUTÉES SANS ERREUR !")
