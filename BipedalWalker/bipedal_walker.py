# bipedal_walker.py

import random
import gymnasium as gym
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neat_module import create_initial_population, InnovationManager, Species, get_genetic_distance, neat_crossover, apply_mutations
import copy


def run(model, env, n_sim):
    
    all_rewards = np.zeros(n_sim)
    for idx_sim in range(n_sim):
        observation, info = env.reset() # Reset environment
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action_values = model.forward(observation)
            action = np.clip(action_values, -1, 1)
            observation, reward, terminated, truncated, _ = env.step(action)
            all_rewards[idx_sim] += reward
        
    return np.mean(all_rewards)

#**********Test pour un génome aléatoire

from gymnasium.wrappers import RecordVideo

trigger = lambda t: t % 5 == 0
env = gym.make(
    "BipedalWalker-v3",
    hardcore = False,
    render_mode="rgb_array", # None for intensive simulation
)
env = RecordVideo(env, video_folder="./videos", name_prefix="random_bipedal_walker",episode_trigger=trigger,video_length=600, disable_logger=True)
ann = random.choice(create_initial_population(10, 24, 4, inv_manager=InnovationManager()))
score = run(ann,env,5)
print(score)
env.close()


#**********Boucle principale d'entraînement

def run_neat_bipedal_walker(generations=100, pop_size=80):
    manager = InnovationManager()
    population = create_initial_population(pop_size, 24, 4, manager)

    env = gym.make("BipedalWalker-v3")

    for gen in range(generations):

        # 1. ÉVALUATION
        for genome in population:
            genome.fitness = run(genome, env, n_sim=5)

        # 2. SPÉCIATION
        species_list = []

        for genome in population:
            found = False
            for s in species_list:
                if get_genetic_distance(genome, s.representative) < 0.3:
                    s.add_member(genome)
                    found = True
                    break
            if not found:
                species_list.append(Species(genome))

        # 3. FITNESS AJUSTÉE
        for s in species_list:
            for g in s.members:
                g.adjusted_fitness = g.fitness / len(s.members)

        # 4. STATS
        best = max(population, key=lambda g: g.fitness)
        print(f"Gen {gen} | Best: {best.fitness:.2f} | Species: {len(species_list)}")

        if best.fitness > 300:
            print("--- Solution trouvée ! ---")
            break

        # 5. ÉLITISME (top 5%)
        sorted_pop = sorted(population, key=lambda g: g.fitness, reverse=True)
        elite_count = max(1, pop_size // 20)
        new_population = [copy.deepcopy(g) for g in sorted_pop[:elite_count]]

        # 6. REPRODUCTION
        while len(new_population) < pop_size:

            # Sélection d'espèce
            species_fitness = [
                sum(g.adjusted_fitness for g in s.members)
                for s in species_list
            ]

            # éviter division par zéro
            species_fitness = [max(f, 1e-6) for f in species_fitness]

            s = random.choices(species_list, weights=species_fitness)[0]

            # Trier les membres
            s.members.sort(key=lambda g: g.fitness, reverse=True)

            # Pool restreint (top 20%)
            cutoff = max(1, len(s.members) // 5)
            pool = s.members[:cutoff]

            # Reproduction
            if len(pool) > 1 and random.random() < 0.75:
                p1, p2 = random.sample(pool, 2)
                child = neat_crossover(p1, p2)
            else:
                child = copy.deepcopy(random.choice(pool))

            apply_mutations(child, manager)
            new_population.append(child)

        # 7. RESET ESPÈCES
        for s in species_list:
            s.representative = random.choice(s.members)
            s.members = []

        population = new_population

    return population



print("*************************")
print("*********START***********")

best_population = run_neat_bipedal_walker()

print("******Best Individual Test******")

test_env = gym.make(
    "BipedalWalker-v3",
    render_mode="rgb_array",
    )

test_env = RecordVideo(test_env, video_folder="./videos/", name_prefix = "best-individual", episode_trigger=trigger,video_length=600, disable_logger=True)

best_genome = best_population[np.argmax([g.fitness for g in best_population])]
score = run(best_genome, test_env, n_sim=5)
print(f"Best genome score: {score}")
test_env.close()

print("*********END***********")
print("*************************")

