# Projet 5 : Optimisation Multi-Agents (Banc d'Essai)

Banc d'essai comparatif de métaheuristiques bio-inspirées et d'intelligence distribuée :
- **PSO** (*Particle Swarm Optimization* - Essaim Particulaire)
- **ACO** (*Ant Colony Optimization* - Colonie de Fourmis continue $\text{ACO}_\mathbb{R}$ et combinatoire Ant System)
- **CLONALG** (*Clonal Selection Algorithm* - Système Immunitaire Artificiel)
- **GA** (*Genetic Algorithm* - Algorithme Génétique continu BLX-$\alpha$ et combinatoire TSP OX)

---

## 👥 Équipe Projet
- **Membre A :** Chef de Projet / Théorie & Modélisation
- **Membre B :** Leila Khezaz (Développeur Quant / Python — Noyau d'Évaluation, Métaheuristiques & Benchmarks)
- **Membre C :** Sara Lamraidi (Développeur Agent IA / LangChain & Intégration Outils)
- **Membre D :** Testeur / Expérimentateur & Bancs d'Essai

---

## 📁 Structure du Dépôt

```
.
├── src/                          # Code source modulaire
│   ├── optim_core.py             # Fonctions continues, utilitaires TSP et CompteurEvaluations (budget strict)
│   ├── algos.py                  # Métaheuristiques continues (GA, PSO, ACO, CLONALG) et TSP (GA OX, ACO)
│   ├── optimizers.py             # Interfaces et dispatchers d'optimisation
│   ├── fitness_functions.py      # Fonctions objectifs analytiques (Sphere, Rastrigin, Rosenbrock)
│   ├── outils_optim.py           # Outils LangChain (comparer_optimiseurs, mann_whitney)
│   └── agent_tp_sma.py           # Module socle SMA et compatibilité descendante
├── tests_b/                      # Suite de validation Membre B (28 tests unitaires)
│   ├── test_algos.py             # Invariants GA, PSO, ACO, CLONALG et TSP
│   └── test_optim_core.py        # Minima globaux, arrêt strict au budget et calculs TSP
├── tests/                        # Tests d'intégration et tests quantitatifs
│   ├── test_quant.py             # Tests de validation fonctionnelle
│   └── test_hors_ligne.py        # Tests hors-ligne des outils agents
├── notebooks/                    # Notebooks Jupyter
│   ├── notebook_s2.ipynb         # Validation complète Semaine 2 de bout en bout
│   └── 02_outils_agent.ipynb     # Démonstration des outils agents
├── docs/                         # Documentation technique et notes d'équipe
│   ├── HANDOFF_NOTES.md          # Notes de transmission (hyperparamètres, formules, outils)
│   ├── convergence_sphere_dim5_medianes.png
│   └── pso_w_convergence_divergence.png
└── requirements.txt              # Dépendances logicielles du projet
```

---

## 🚀 Installation & Exécution

### 1. Installation des dépendances
```bash
pip install -r requirements.txt
```

### 2. Exécution des tests unitaires
```bash
pytest tests_b/
```

### 3. Exécution du banc d'essai Semaine 2
```bash
python build_and_run_notebook.py
```
