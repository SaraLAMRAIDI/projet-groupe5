# Livrables du Membre B & Notes de Transmission à l'Équipe

**Projet 5 — Optimisation multi-agents : essaim, fourmis, immunitaire et génétique (banc d'essai)**  
**Auteur :** Membre B (Quant / Python)

---

## 1. Transmission au Membre C (Développeur Agent)

### Signature Standardisée des Fonctions
Chaque algorithme dans `optimizers.py` suit strictement la même convention d'appel :
```python
def nom_algorithme(fitness_func, dim, bounds, budget, seed=42, **params):
    """
    Retourne :
        best_position (np.ndarray) : Coordonnées de la meilleure solution trouvée.
        best_fitness (float)       : Valeur de la fonction objectif à l'optimum.
        history (list[float])      : Historique de la meilleure fitness par itération (pour les courbes).
    """
```

### Dictionnaires de Routage (Dispatchers)
```python
from optimizers import ALGORITHMS
# ALGORITHMS = {"PSO": pso, "ACO": aco, "CLONALG": clonalg, "GA": ga}

from fitness_functions import FUNCTIONS, BOUNDS
# FUNCTIONS = {"sphere": sphere, "rastrigin": rastrigin, "rosenbrock": rosenbrock}
# BOUNDS = {"sphere": (-100, 100), "rastrigin": (-5.12, 5.12), "rosenbrock": (-30, 30)}
```

### Encapsulation dans un Outil LangChain (`@tool`)
```python
from langchain.tools import tool
from optimizers import ALGORITHMS
from fitness_functions import FUNCTIONS, BOUNDS

@tool
def comparer_optimiseurs(algo_name: str, func_name: str, dim: int, budget: int = 5000, seed: int = 42) -> dict:
    """Exécute une métaheuristique d'optimisation sur une fonction test de référence."""
    algo = ALGORITHMS[algo_name.upper()]
    func = FUNCTIONS[func_name.lower()]
    bounds = BOUNDS[func_name.lower()]
    
    best_pos, best_fit, history = algo(func, dim=dim, bounds=bounds, budget=budget, seed=seed)
    return {
        "algo": algo_name,
        "func": func_name,
        "best_fitness": float(best_fit),
        "best_position": best_pos.tolist(),
        "iterations": len(history)
    }
```

---

## 2. Note & Répertoire des Hyperparamètres pour le Membre D (Testeur / Expérimentateur)

Tous les algorithmes sont configurés pour garantir une comparaison rigoureuse et équitable sous un budget d'évaluations identique.

| Algorithme | Paramètre | Valeur | Rôle / Justification |
| :--- | :--- | :--- | :--- |
| **Commun** | `pop_size` | 50 | Taille de population uniforme pour toutes les métaheuristiques. |
| **Commun** | `budget` | 5000 (ou 10000) | Plafond d'évaluations strictement respecté (`eval_count <= budget`). |
| **Commun** | `bounds` | Depuis `BOUNDS` | Espace de recherche ; bornage systématique : $\text{clip}(x, x_{\min}, x_{\max})$. |
| **PSO** | $w$ | 0.7 | Poids d'inertie (régime convergent prouvé en cours, évitant la divergence à $w=1.2$). |
| **PSO** | $c_1$ | 1.4 | Facteur d'accélération cognitive (attraction vers la meilleure position personnelle). |
| **PSO** | $c_2$ | 1.4 | Facteur d'accélération sociale (attraction vers la meilleure position de l'essaim). |
| **PSO** | $v_{\max}$ | $0.2 \cdot (upper - lower)$ | Bornage des vitesses pour empêcher l'explosion de l'essaim. |
| **GA** | $p_c$ | 0.8 | Probabilité de croisement (croisement arithmétique / BLX-$\alpha$). |
| **GA** | $p_m$ | 0.1 | Probabilité de mutation par gène. |
| **GA** | $\sigma_{\text{mut}}$ | $0.1 \cdot (upper - lower)$ | Écart-type de la perturbation gaussienne. |
| **GA** | `tournament_size` | 3 | Pression de sélection (sélection par tournoi de taille 3). |
| **GA** | Élitisme | 1 meilleur | Conservation inconditionnelle du meilleur individu dans la génération suivante. |
| **CLONALG** | `clone_factor` | 0.2 | Coefficient de base pour le clonage : $N_c = \text{arrondi}(\beta \cdot N \cdot c / \text{rang})$. |
| **CLONALG** | $\beta$ | 1.0 | Facteur d'échelle du clonage. |
| **CLONALG** | $\sigma_{\text{hyp}}$ | $[0.01, 0.20] \cdot \Delta$ | Hypermutation inversement proportionnelle à l'affinité. |
| **CLONALG** | `replace_rate`| 0.10 | Édition des récepteurs : remplacement aléatoire des 10% pires anticorps. |
| **ACO** ($\text{ACO}_{\mathbb{R}}$)| $K$ | 50 | Taille de l'archive de solutions. |
| **ACO** ($\text{ACO}_{\mathbb{R}}$)| $q$ | 0.5 | Paramètre de localité : contrôle la décroissance du poids gaussien selon le rang. |
| **ACO** ($\text{ACO}_{\mathbb{R}}$)| $\xi$ | 0.85 | Paramètre d'évaporation / dispersion régulant l'écart-type d'échantillonnage. |

---

## 3. Formules Mathématiques pour le Membre A (Chef de Projet / Théorie & Rapport)

### 3.1 Essaim Particulaire (PSO)
- **Mise à jour des vitesses :**
  $$v_i^{(t+1)} = w \cdot v_i^{(t)} + c_1 r_1 \left(pbest_i - x_i^{(t)}\right) + c_2 r_2 \left(gbest - x_i^{(t)}\right)$$
  avec $r_1, r_2 \sim \mathcal{U}(0, 1)^d$.
- **Bornage des vitesses :**
  $$v_{i, d} \leftarrow \max\left(-v_{\max, d}, \min\left(v_{\max, d}, v_{i, d}\right)\right), \quad v_{\max, d} = 0.2 \cdot (x_{\max, d} - x_{\min, d})$$
- **Mise à jour des positions & bornage spatial :**
  $$x_i^{(t+1)} = \text{clip}\left(x_i^{(t)} + v_i^{(t+1)}, x_{\min}, x_{\max}\right)$$

### 3.2 Colonie de Fourmis Continue ($\text{ACO}_{\mathbb{R}}$)
- **Archive de Solutions :** $T$ contient $K$ solutions ordonnées telles que $f(x^1) \le f(x^2) \le \dots \le f(x^K)$.
- **Poids Gaussiens (analogue aux phéromones) :**
  $$w_l = \frac{1}{q K \sqrt{2\pi}} \exp\left(-\frac{(l-1)^2}{2 q^2 K^2}\right), \quad p_l = \frac{w_l}{\sum_{j=1}^K w_j}$$
- **Échantillonnage par Noyau Gaussien :**
  $$x_d \sim \mathcal{N}\left(\mu_d, \sigma_d^2\right), \quad \mu_d = x_d^l, \quad \sigma_d = \xi \sum_{j=1}^K \frac{|x_d^j - x_d^l|}{K - 1}$$

### 3.3 Sélection Clonale (CLONALG)
- **Taux de Clonage :**
  $$N_{\text{clones}}(i) = \max\left(1, \left\lfloor \frac{\beta \cdot N \cdot \text{clone\_factor}}{i} \right\rceil\right)$$
- **Hypermutation selon l'Affinité :**
  $$\sigma_i = \left(\sigma_{\min} + (\sigma_{\max} - \sigma_{\min}) \cdot \frac{i - 1}{N - 1}\right) \cdot (x_{\max} - x_{\min})$$
  $$x_{\text{clone}} = \text{clip}(x_i + \mathcal{N}(0, \sigma_i^2), x_{\min}, x_{\max})$$
- **Édition des Récepteurs (Diversité) :**
  Remplacement des $d = \lfloor \text{replace\_rate} \cdot N \rfloor$ anticorps de plus faible affinité par de nouveaux individus aléatoires uniformes.

### 3.4 Algorithme Génétique (GA)
- **Sélection :** Sélection par tournoi à 3 individus.
- **Croisement Arithmétique / BLX :**
  $$c_1 = \alpha x_1 + (1 - \alpha) x_2, \quad c_2 = \alpha x_2 + (1 - \alpha) x_1, \quad \alpha \sim \mathcal{U}(-0.1, 1.1)^d$$
- **Mutation Gaussienne :**
  $$x_d' = x_d + \mathcal{N}(0, \sigma^2) \quad \text{avec probabilité } p_m$$
- **Élitisme :** Le meilleur individu de la génération courante est préservé sans altération.

---

## 4. Fonctions Tests de Référence
1. **Sphère :**
   $$f_{\text{Sphère}}(x) = \sum_{i=1}^d x_i^2, \quad x \in [-100, 100]^d, \quad \min f(0, \dots, 0) = 0$$
2. **Rastrigin :**
   $$f_{\text{Rastrigin}}(x) = 10d + \sum_{i=1}^d \left(x_i^2 - 10 \cos(2\pi x_i)\right), \quad x \in [-5.12, 5.12]^d, \quad \min f(0, \dots, 0) = 0$$
3. **Rosenbrock :**
   $$f_{\text{Rosenbrock}}(x) = \sum_{i=1}^{d-1} \left[100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2\right], \quad x \in [-30, 30]^d, \quad \min f(1, \dots, 1) = 0$$
