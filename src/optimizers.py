"""
optimizers.py - Metaheuristic optimization algorithms for continuous functions.
Project: Multi-agent Optimization (PSO, ACO, CLONALG, GA)
Role: Member B (Quant / Python)

All algorithms share the standard signature:
    def algorithm_name(fitness_func, dim, bounds, budget, seed=42, **params):
        return best_position, best_fitness, history
"""

import numpy as np


def pso(fitness_func, dim, bounds, budget, seed=42, pop_size=50, w=0.7, c1=1.4, c2=1.4, **kwargs):
    """
    Particle Swarm Optimization (PSO).
    
    Update rules:
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        v = clip(v, -v_max, v_max) where v_max = 0.2 * (upper - lower)
        x = clip(x + v, lower, upper)
    
    Parameters:
        fitness_func: Callable objective function f(x) -> float
        dim: Problem dimensionality (int)
        bounds: Tuple of (lower, upper)
        budget: Maximum number of function evaluations (int)
        seed: Random seed for reproducibility (int)
        pop_size: Number of particles in the swarm (int)
        w: Inertia weight (default 0.7 for convergence)
        c1: Cognitive acceleration coefficient (default 1.4)
        c2: Social acceleration coefficient (default 1.4)
        
    Returns:
        best_position (np.ndarray): Best solution coordinates found
        best_fitness (float): Objective value at best solution
        history (list[float]): Best fitness per iteration
    """
    rng = np.random.default_rng(seed)
    lower, upper = bounds
    v_max = 0.2 * (upper - lower)
    eval_count = 0

    # 1. Initialize swarm positions and velocities
    pop = rng.uniform(lower, upper, size=(pop_size, dim))
    vel = rng.uniform(-v_max, v_max, size=(pop_size, dim))
    
    pbest = pop.copy()
    pbest_fit = np.full(pop_size, np.inf)

    # Initial evaluation
    for i in range(pop_size):
        if eval_count >= budget:
            break
        fit = float(fitness_func(pop[i]))
        eval_count += 1
        pbest_fit[i] = fit

    best_idx = np.argmin(pbest_fit)
    gbest = pbest[best_idx].copy()
    gbest_fit = pbest_fit[best_idx]
    history = [gbest_fit]

    # 2. Main swarm loop
    while eval_count < budget:
        for i in range(pop_size):
            if eval_count >= budget:
                break
            
            # Stochastic velocity update
            r1 = rng.uniform(0.0, 1.0, size=dim)
            r2 = rng.uniform(0.0, 1.0, size=dim)
            vel[i] = (w * vel[i] 
                      + c1 * r1 * (pbest[i] - pop[i]) 
                      + c2 * r2 * (gbest - pop[i]))
            
            # Clip velocity to prevent explosion
            vel[i] = np.clip(vel[i], -v_max, v_max)
            
            # Update position and clip within search space
            pop[i] = np.clip(pop[i] + vel[i], lower, upper)
            
            # Function evaluation with budget tracking
            fit = float(fitness_func(pop[i]))
            eval_count += 1
            
            # Update personal best
            if fit < pbest_fit[i]:
                pbest[i] = pop[i].copy()
                pbest_fit[i] = fit
                
                # Update global best
                if fit < gbest_fit:
                    gbest = pop[i].copy()
                    gbest_fit = fit

        history.append(gbest_fit)

    return gbest, gbest_fit, history


def ga(fitness_func, dim, bounds, budget, seed=42, pop_size=50, pc=0.8, pm=0.1, tournament_size=3, **kwargs):
    """
    Continuous Genetic Algorithm (GA).
    
    Uses:
        - Tournament selection
        - Arithmetic / BLX-alpha crossover
        - Gaussian mutation
        - Elitism (preserves the single best individual)
        - Boundary clipping: clip(x, lower, upper)
        
    Parameters:
        fitness_func: Callable objective function f(x) -> float
        dim: Problem dimensionality (int)
        bounds: Tuple of (lower, upper)
        budget: Maximum number of function evaluations (int)
        seed: Random seed for reproducibility (int)
        pop_size: Population size (int)
        pc: Crossover probability (float, default 0.8)
        pm: Mutation probability per gene (float, default 0.1)
        tournament_size: Size of tournament pool for selection (int, default 3)
        
    Returns:
        best_position (np.ndarray): Best solution coordinates found
        best_fitness (float): Objective value at best solution
        history (list[float]): Best fitness per iteration
    """
    rng = np.random.default_rng(seed)
    lower, upper = bounds
    sigma_mut = 0.1 * (upper - lower)
    eval_count = 0

    # 1. Initialize population
    pop = rng.uniform(lower, upper, size=(pop_size, dim))
    fitness = np.full(pop_size, np.inf)

    for i in range(pop_size):
        if eval_count >= budget:
            break
        fitness[i] = float(fitness_func(pop[i]))
        eval_count += 1

    best_idx = np.argmin(fitness)
    gbest = pop[best_idx].copy()
    gbest_fit = fitness[best_idx]
    history = [gbest_fit]

    def tournament_select():
        candidates = rng.choice(pop_size, size=tournament_size, replace=False)
        best_cand = candidates[np.argmin(fitness[candidates])]
        return pop[best_cand].copy()

    # 2. Main generational loop
    while eval_count < budget:
        new_pop = []
        new_fitness = []

        # Elitism: retain current best individual
        new_pop.append(gbest.copy())
        new_fitness.append(gbest_fit)

        # Generate offspring
        while len(new_pop) < pop_size and eval_count < budget:
            parent1 = tournament_select()
            parent2 = tournament_select()

            # Crossover (Arithmetic / Blend)
            if rng.uniform(0.0, 1.0) < pc:
                alpha = rng.uniform(-0.1, 1.1, size=dim)
                child1 = alpha * parent1 + (1.0 - alpha) * parent2
                child2 = alpha * parent2 + (1.0 - alpha) * parent1
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()

            for child in (child1, child2):
                if len(new_pop) >= pop_size or eval_count >= budget:
                    break

                # Mutation (Gaussian)
                mut_mask = rng.uniform(0.0, 1.0, size=dim) < pm
                if np.any(mut_mask):
                    child[mut_mask] += rng.normal(0.0, sigma_mut, size=np.sum(mut_mask))

                # Boundary constraint
                child = np.clip(child, lower, upper)

                # Evaluation
                fit = float(fitness_func(child))
                eval_count += 1

                new_pop.append(child)
                new_fitness.append(fit)

                if fit < gbest_fit:
                    gbest = child.copy()
                    gbest_fit = fit

        pop = np.array(new_pop)
        fitness = np.array(new_fitness)
        history.append(gbest_fit)

    return gbest, gbest_fit, history


def clonalg(fitness_func, dim, bounds, budget, seed=42, pop_size=50, clone_factor=0.2, beta=1.0, replace_rate=0.1, **kwargs):
    """
    Clonal Selection Algorithm (CLONALG) - Artificial Immune System for optimization.
    
    Mechanism:
        1. Antibodies represent candidate solutions in the search space.
        2. Antibodies are ranked by affinity (lower fitness = higher affinity).
        3. Top antibodies produce clones proportionally to their affinity rank.
        4. Clones undergo affinity-dependent hypermutation: higher affinity
           antibodies mutate with smaller step sizes (exploitation), while lower
           affinity antibodies mutate with larger step sizes (exploration).
        5. The best clone of each antibody replaces it if it has better fitness.
        6. Receptor editing: lowest affinity antibodies are replaced by new
           random antibodies to prevent premature convergence.
        
    Parameters:
        fitness_func: Callable objective function f(x) -> float
        dim: Problem dimensionality (int)
        bounds: Tuple of (lower, upper)
        budget: Maximum number of function evaluations (int)
        seed: Random seed for reproducibility (int)
        pop_size: Repertoire size (antibodies count, int)
        clone_factor: Base multiplier for clone count (float, default 0.2)
        beta: Scaling factor for rank-based cloning (float, default 1.0)
        replace_rate: Fraction of worst antibodies randomly replaced (float, default 0.1)
        
    Returns:
        best_position (np.ndarray): Best solution coordinates found
        best_fitness (float): Objective value at best solution
        history (list[float]): Best fitness per iteration
    """
    rng = np.random.default_rng(seed)
    lower, upper = bounds
    eval_count = 0
    space_range = upper - lower

    # 1. Initialize antibody population
    pop = rng.uniform(lower, upper, size=(pop_size, dim))
    fitness = np.full(pop_size, np.inf)

    for i in range(pop_size):
        if eval_count >= budget:
            break
        fitness[i] = float(fitness_func(pop[i]))
        eval_count += 1

    best_idx = np.argmin(fitness)
    gbest = pop[best_idx].copy()
    gbest_fit = fitness[best_idx]
    history = [gbest_fit]

    # 2. Main clonal selection loop
    while eval_count < budget:
        # Sort antibodies by affinity (ascending fitness)
        sort_order = np.argsort(fitness)
        pop = pop[sort_order]
        fitness = fitness[sort_order]

        # Process each antibody for cloning and hypermutation
        for i in range(pop_size):
            if eval_count >= budget:
                break

            # Number of clones proportional to rank
            rank = i + 1
            n_clones = max(1, int(round((beta * pop_size * clone_factor) / rank)))

            # Hypermutation step size inversely proportional to affinity
            # Rank 1 (best): small sigma (fine search)
            # Higher rank (worse): larger sigma (broad search)
            normalized_rank = i / max(1, pop_size - 1)
            sigma = (0.01 + 0.19 * normalized_rank) * space_range

            best_clone = pop[i].copy()
            best_clone_fit = fitness[i]

            for _ in range(n_clones):
                if eval_count >= budget:
                    break

                # Mutate clone
                clone = pop[i] + rng.normal(0.0, sigma, size=dim)
                clone = np.clip(clone, lower, upper)

                fit = float(fitness_func(clone))
                eval_count += 1

                if fit < best_clone_fit:
                    best_clone = clone.copy()
                    best_clone_fit = fit

            # Clonal selection replacement
            if best_clone_fit < fitness[i]:
                pop[i] = best_clone
                fitness[i] = best_clone_fit
                if best_clone_fit < gbest_fit:
                    gbest = best_clone.copy()
                    gbest_fit = best_clone_fit

        # Receptor editing: replace worst fraction with fresh random solutions
        num_replace = max(1, int(round(replace_rate * pop_size)))
        for k in range(pop_size - num_replace, pop_size):
            if eval_count >= budget:
                break
            new_ind = rng.uniform(lower, upper, size=dim)
            fit = float(fitness_func(new_ind))
            eval_count += 1
            pop[k] = new_ind
            fitness[k] = fit
            if fit < gbest_fit:
                gbest = new_ind.copy()
                gbest_fit = fit

        history.append(gbest_fit)

    return gbest, gbest_fit, history


def aco(fitness_func, dim, bounds, budget, seed=42, pop_size=50, q=0.5, xi=0.85, **kwargs):
    """
    Continuous Ant Colony Optimization (ACO_R).
    
    Maintains a solution archive of size K (pop_size) ranked by fitness.
    Solutions are assigned Gaussian weights based on rank:
        w_l = 1 / (q * K * sqrt(2*pi)) * exp(- (l-1)^2 / (2 * q^2 * K^2))
    New candidate positions are sampled using Gaussian kernel convolution around
    the selected guide solution with dispersion determined by distance to peers.
    
    Parameters:
        fitness_func: Callable objective function f(x) -> float
        dim: Problem dimensionality (int)
        bounds: Tuple of (lower, upper)
        budget: Maximum number of function evaluations (int)
        seed: Random seed for reproducibility (int)
        pop_size: Solution archive size K (int, default 50)
        q: Locality parameter (smaller q = stronger preference for top solutions, default 0.5)
        xi: Evaporation / convergence parameter controlling standard deviation (default 0.85)
        
    Returns:
        best_position (np.ndarray): Best solution coordinates found
        best_fitness (float): Objective value at best solution
        history (list[float]): Best fitness per iteration
    """
    rng = np.random.default_rng(seed)
    lower, upper = bounds
    eval_count = 0
    k_archive = pop_size

    # 1. Initialize solution archive
    archive = rng.uniform(lower, upper, size=(k_archive, dim))
    fitness = np.full(k_archive, np.inf)

    for i in range(k_archive):
        if eval_count >= budget:
            break
        fitness[i] = float(fitness_func(archive[i]))
        eval_count += 1

    # Sort archive
    sort_order = np.argsort(fitness)
    archive = archive[sort_order]
    fitness = fitness[sort_order]

    gbest = archive[0].copy()
    gbest_fit = fitness[0]
    history = [gbest_fit]

    # Precalculate selection probabilities based on ranks
    ranks = np.arange(k_archive)
    weights = (1.0 / (q * k_archive * np.sqrt(2.0 * np.pi))) * np.exp(- (ranks**2) / (2.0 * (q**2) * (k_archive**2)))
    probs = weights / np.sum(weights)

    # 2. Ant construction loop
    while eval_count < budget:
        new_solutions = []
        new_fitnesses = []

        # Number of new ants per generation equals archive size
        for _ in range(k_archive):
            if eval_count >= budget:
                break

            # Choose guide solution l based on pheromone weight distribution
            guide_idx = rng.choice(k_archive, p=probs)
            guide = archive[guide_idx]

            # Compute standard deviation per dimension based on archive spread
            if k_archive > 1:
                sigma = xi * np.sum(np.abs(archive - guide), axis=0) / (k_archive - 1)
            else:
                sigma = np.full(dim, 0.1 * (upper - lower))

            # Avoid degenerate zero sigma
            sigma = np.maximum(sigma, 1e-12)

            # Sample new solution from Gaussian kernel
            ant_pos = rng.normal(guide, sigma)
            ant_pos = np.clip(ant_pos, lower, upper)

            fit = float(fitness_func(ant_pos))
            eval_count += 1

            new_solutions.append(ant_pos)
            new_fitnesses.append(fit)

            if fit < gbest_fit:
                gbest = ant_pos.copy()
                gbest_fit = fit

        if new_solutions:
            # Combine archive and new solutions, then keep top k_archive
            combined_pop = np.vstack([archive, np.array(new_solutions)])
            combined_fit = np.concatenate([fitness, np.array(new_fitnesses)])

            best_indices = np.argsort(combined_fit)[:k_archive]
            archive = combined_pop[best_indices]
            fitness = combined_fit[best_indices]

        history.append(gbest_fit)

    return gbest, gbest_fit, history


# --- Dispatcher Dictionary (Essential for Member C & LangChain Integration) ---
ALGORITHMS = {
    "PSO": pso,
    "ACO": aco,
    "CLONALG": clonalg,
    "GA": ga,
}

