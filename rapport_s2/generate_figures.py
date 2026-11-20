import os
import sys
import json
import numpy as np

# Ajout du dossier parent au sys.path pour importer optim_core et algos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import matplotlib as mpl

# Configuration esthétique des graphiques
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['lines.linewidth'] = 1.8

from optim_core import (
    sphere, rastrigin, rosenbrock,
    DEFAULT_BOUNDS, generer_instance_tsp,
    longueur_tour, plus_proche_voisin
)
from algos import (
    algo_genetique, algo_genetique_tsp,
    pso, aco_continu, aco_tsp, clonalg,
    ALGOS_CONTINUS
)

output_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(output_dir, exist_ok=True)

SEEDS = list(range(42, 57))  # 15 graines pour des statistiques rigoureuses
DIM = 5
BUDGET = 3000
COULEURS = {
    "GA": "#1f77b4",       # Bleu
    "PSO": "#2ca02c",      # Vert
    "ACO": "#ff7f0e",      # Orange
    "CLONALG": "#d62728"   # Rouge
}

print(f"--- 1. Génération Convergence Sphere (dim={DIM}, 15 graines) ---")
donnees_sphere = {nom: [] for nom in ALGOS_CONTINUS}
final_sphere = {nom: [] for nom in ALGOS_CONTINUS}

for s in SEEDS:
    for nom, fn in ALGOS_CONTINUS.items():
        pos, fit, hist = fn(sphere, dim=DIM, bornes=DEFAULT_BOUNDS["sphere"], budget=BUDGET, seed=s)
        donnees_sphere[nom].append(hist)
        final_sphere[nom].append(fit)

plt.figure(figsize=(7, 4.2), dpi=300)
x_evals = np.arange(1, BUDGET + 1)
for nom, series in donnees_sphere.items():
    mat = np.array(series)
    med = np.maximum(np.median(mat, axis=0), 1e-30)
    q25 = np.maximum(np.percentile(mat, 25, axis=0), 1e-30)
    q75 = np.maximum(np.percentile(mat, 75, axis=0), 1e-30)
    c = COULEURS[nom]
    plt.plot(x_evals, med, label=f"{nom} (médiane: {med[-1]:.2e})", color=c)
    plt.fill_between(x_evals, q25, q75, color=c, alpha=0.15)

plt.yscale("log")
plt.xlabel("Nombre d'évaluations de la fonction objectif")
plt.ylabel("Meilleure fitness $f(x)$ (échelle log)")
plt.title(f"Convergence comparée sur la fonction Sphère ($d={DIM}$, budget={BUDGET}, 15 graines)")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(loc="upper right")
plt.savefig(os.path.join(output_dir, "convergence_sphere_dim5_medianes.png"))
plt.close()

print(f"--- 2. Génération Convergence Rastrigin (dim={DIM}, 15 graines) ---")
donnees_rastrigin = {nom: [] for nom in ALGOS_CONTINUS}
final_rastrigin = {nom: [] for nom in ALGOS_CONTINUS}

for s in SEEDS:
    for nom, fn in ALGOS_CONTINUS.items():
        pos, fit, hist = fn(rastrigin, dim=DIM, bornes=DEFAULT_BOUNDS["rastrigin"], budget=BUDGET, seed=s)
        donnees_rastrigin[nom].append(hist)
        final_rastrigin[nom].append(fit)

plt.figure(figsize=(7, 4.2), dpi=300)
for nom, series in donnees_rastrigin.items():
    mat = np.array(series)
    med = np.maximum(np.median(mat, axis=0), 1e-30)
    q25 = np.maximum(np.percentile(mat, 25, axis=0), 1e-30)
    q75 = np.maximum(np.percentile(mat, 75, axis=0), 1e-30)
    c = COULEURS[nom]
    plt.plot(x_evals, med, label=f"{nom} (médiane: {med[-1]:.2f})", color=c)
    plt.fill_between(x_evals, q25, q75, color=c, alpha=0.15)

plt.yscale("log")
plt.xlabel("Nombre d'évaluations de la fonction objectif")
plt.ylabel("Meilleure fitness $f(x)$ (échelle log)")
plt.title(f"Convergence comparée sur la fonction de Rastrigin ($d={DIM}$, budget={BUDGET}, 15 graines)")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(loc="upper right")
plt.savefig(os.path.join(output_dir, "convergence_rastrigin_dim5.png"))
plt.close()

print(f"--- 3. Génération Convergence Rosenbrock (dim={DIM}, 15 graines) ---")
donnees_rosenbrock = {nom: [] for nom in ALGOS_CONTINUS}
final_rosenbrock = {nom: [] for nom in ALGOS_CONTINUS}

for s in SEEDS:
    for nom, fn in ALGOS_CONTINUS.items():
        pos, fit, hist = fn(rosenbrock, dim=DIM, bornes=DEFAULT_BOUNDS["rosenbrock"], budget=BUDGET, seed=s)
        donnees_rosenbrock[nom].append(hist)
        final_rosenbrock[nom].append(fit)

plt.figure(figsize=(7, 4.2), dpi=300)
for nom, series in donnees_rosenbrock.items():
    mat = np.array(series)
    med = np.maximum(np.median(mat, axis=0), 1e-30)
    q25 = np.maximum(np.percentile(mat, 25, axis=0), 1e-30)
    q75 = np.maximum(np.percentile(mat, 75, axis=0), 1e-30)
    c = COULEURS[nom]
    plt.plot(x_evals, med, label=f"{nom} (médiane: {med[-1]:.2f})", color=c)
    plt.fill_between(x_evals, q25, q75, color=c, alpha=0.15)

plt.yscale("log")
plt.xlabel("Nombre d'évaluations de la fonction objectif")
plt.ylabel("Meilleure fitness $f(x)$ (échelle log)")
plt.title(f"Convergence comparée sur la fonction de Rosenbrock ($d={DIM}$, budget={BUDGET}, 15 graines)")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(loc="upper right")
plt.savefig(os.path.join(output_dir, "convergence_rosenbrock_dim5.png"))
plt.close()

print(f"--- 4. Étude de l'inertie du PSO (w=0.7 vs w=1.2) ---")
_, fit_w07, hist_w07 = pso(sphere, dim=5, bornes=DEFAULT_BOUNDS["sphere"], budget=3000, seed=42, w=0.7)
_, fit_w09, hist_w09 = pso(sphere, dim=5, bornes=DEFAULT_BOUNDS["sphere"], budget=3000, seed=42, w=0.9)
_, fit_w12, hist_w12 = pso(sphere, dim=5, bornes=DEFAULT_BOUNDS["sphere"], budget=3000, seed=42, w=1.2)

plt.figure(figsize=(7, 4.0), dpi=300)
plt.plot(hist_w07, label=f"PSO w=0.7 (Régime convergent, final: {fit_w07:.2e})", color="#2ca02c", lw=2)
plt.plot(hist_w09, label=f"PSO w=0.9 (Régime transitoire, final: {fit_w09:.2e})", color="#ff7f0e", lw=1.8, linestyle="--")
plt.plot(hist_w12, label=f"PSO w=1.2 (Régime divergent, final: {fit_w12:.2e})", color="#d62728", lw=2)
plt.yscale("log")
plt.xlabel("Nombre d'évaluations de $f$")
plt.ylabel("Fitness (échelle log)")
plt.title("Sensibilité à l'inertie $w$ en PSO (Sphère $d=5$, budget=3000)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="upper right")
plt.savefig(os.path.join(output_dir, "pso_w_convergence_divergence.png"))
plt.close()

print(f"--- 5. Benchmark TSP (20 villes, GA-OX vs ACO Ant System vs Heuristique Gloutonne) ---")
coords_tsp, D_tsp = generer_instance_tsp(n_villes=20, seed=42)
tour_ppv, dist_ppv = plus_proche_voisin(D_tsp, depart=0)

budget_tsp = 4000
tour_ga, dist_ga, hist_ga = algo_genetique_tsp(D_tsp, budget=budget_tsp, seed=42)
tour_aco, dist_aco, hist_aco = aco_tsp(D_tsp, budget=budget_tsp, seed=42)

plt.figure(figsize=(7, 4.2), dpi=300)
plt.axhline(y=dist_ppv, color="black", linestyle=":", label=f"Plus Proche Voisin (Glouton: {dist_ppv:.1f})", lw=1.8)
plt.plot(hist_ga, label=f"GA TSP (Croisement OX, final: {dist_ga:.1f})", color="#1f77b4", lw=2)
plt.plot(hist_aco, label=f"ACO TSP (Système de Fourmis, final: {dist_aco:.1f})", color="#2ca02c", lw=2)
plt.xlabel("Nombre d'évaluations de tour")
plt.ylabel("Longueur totale du tour fermé")
plt.title("Optimisation combinatoire TSP (20 villes, budget=4000 évals)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="upper right")
plt.savefig(os.path.join(output_dir, "tsp_benchmark_convergence.png"))
plt.close()

print(f"--- 6. Boxplots comparatifs de robustesse (15 graines) ---")
fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0), dpi=300)

data_sp = [final_sphere[k] for k in ["GA", "PSO", "ACO", "CLONALG"]]
axes[0].boxplot(data_sp, tick_labels=["GA", "PSO", "ACO", "CLONALG"])
axes[0].set_yscale("log")
axes[0].set_title("Sphère ($d=5$, 15 graines)")
axes[0].set_ylabel("Fitness finale (log)")
axes[0].grid(True, linestyle="--", alpha=0.5)

data_rast = [final_rastrigin[k] for k in ["GA", "PSO", "ACO", "CLONALG"]]
axes[1].boxplot(data_rast, tick_labels=["GA", "PSO", "ACO", "CLONALG"])
axes[1].set_yscale("log")
axes[1].set_title("Rastrigin ($d=5$, 15 graines)")
axes[1].set_ylabel("Fitness finale (log)")
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.suptitle("Distribution des scores finaux (Boxplots sur 15 graines indépendantes)", fontsize=11)
plt.savefig(os.path.join(output_dir, "boxplots_benchmark_sphere_rastrigin.png"))
plt.close()

# Sauvegarde des statistiques numériques réelles sous forme JSON
stats = {}
for pb_name, data_dict in [("sphere", final_sphere), ("rastrigin", final_rastrigin), ("rosenbrock", final_rosenbrock)]:
    stats[pb_name] = {}
    for nom, vals in data_dict.items():
        arr = np.array(vals)
        stats[pb_name][nom] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "best": float(np.min(arr)),
            "worst": float(np.max(arr))
        }

stats["tsp"] = {
    "ppv": float(dist_ppv),
    "ga": float(dist_ga),
    "aco": float(dist_aco)
}

stats_path = os.path.join(output_dir, "empirical_stats.json")
with open(stats_path, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

print(f"[OK] Toutes les figures et statistiques générées avec succès dans {output_dir}")
