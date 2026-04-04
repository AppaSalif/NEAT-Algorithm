import random
import copy
import numpy as np



# Fonction d'activation (sigmoïde)
def sigmoid(x):
    # Clip pour éviter les overflows avec exp
    x = np.clip(x, -500, 500)    
    return 1. / (1 + np.exp(-x))


# Innovation manager

class InnovationManager:
    def __init__(self):
        self.current_innovation = 0
        self.current_node_id = 0
        # Historique des connexions: {(in_node, out_node): innovation_id}
        self.connection_history = {}

    def get_innovation_conn(self, in_id, out_id):
        pair = (in_id, out_id)
        if pair not in self.connection_history:
            self.connection_history[pair] = self.current_innovation
            self.current_innovation +=1
        return self.connection_history[pair]

    def get_node_id(self):
        node_id = self.current_node_id
        self.current_node_id +=1
        return node_id


# Node d'un réseau
class Node:
    def __init__(self, id, type):
        self.id = id            # unique node id
        self.type = type        # "input", "hidden", "output", "bias"
        self.value = 0.0        


# Connexion entre deux nodes
class Connection:
    def __init__(self, in_node, out_node, weight, enabled=True, innovation=0):
        self.in_node = in_node      # Node source
        self.out_node = out_node    # Node destination
        self.weight = weight        # weight of the conn.
        self.enabled = enabled      # if conn. is activated
        self.innovation = innovation # unique id for crossover

# Genome : ensemble de nodes et connexions
class Genome:
    def __init__(self):
        self.nodes = {}            
        self.connections = {}      
        self.adj_list = set() # Pour verifier (in, out) en O(1)
        self.fitness = 0.0
        self.adjusted_fitness = 0.0

    def add_node(self, node_id, type):
        node = Node(node_id, type)
        self.nodes[node_id] = node
        return node

    def add_connection(self, in_id, out_id, weight, innovation, enabled=True):
        conn = Connection(in_id, out_id, weight, enabled, innovation)
        self.connections[innovation] = conn
        self.adj_list.add((in_id, out_id))
        return conn

    # Mutation des poids des connexions
    def mutate_weights(self, perturb_rate=0.8, step=0.5):
        for conn in self.connections.values():
            if random.random() < perturb_rate:
                conn.weight +=random.uniform(-step, step)
            else:
                # nouveau poids aléatoire
                conn.weight =random.uniform(-1., 1.)

    def mutate_add_connection(self, innovation_manager, max_attempts=20):
        nodes_list = list(self.nodes.values())

        for _ in range(max_attempts):
            n1 = random.choice(nodes_list)
            n2 = random.choice(nodes_list)
            
            # éviter cycles (simple règle DAG)
            if n1.id >= n2.id:
                continue
            
            if n1.type == "output" or n2.type == "input":
                continue

            if (n1.id, n2.id) in self.adj_list:
                continue

            # On récupère l'ID d'innovation global
            inv = innovation_manager.get_innovation_conn(n1.id, n2.id)
            weight = random.uniform(-1., 1.)
            self.add_connection(n1.id, n2.id, weight, inv)
            return True
        
        return False


    def mutate_add_node(self, innovation_manager):
        #choisir une connexion active au hasard
        enabled_cons = [c for c in self.connections.values() if c.enabled]

        if not enabled_cons:
            return False

        conn = random.choice(enabled_cons)
        conn.enabled = False  # On désactive l'ancienne connexion

        #créer le nouveau node
        new_node_id = innovation_manager.get_node_id()
        self.add_node(new_node_id, "hidden")

        # créer les deux nouvelles connexions 

        # A -> new_node (poids = 1)
        inv1 = innovation_manager.get_innovation_conn(conn.in_node, new_node_id)
        self.add_connection(conn.in_node, new_node_id, 1.0, inv1)
        
        # new_node -> B (poids = weight de l'ancienne connexion)
        inv2 = innovation_manager.get_innovation_conn(new_node_id, conn.out_node)
        self.add_connection(new_node_id, conn.out_node, conn.weight, inv2)
        
        return True
        
    def compute_layers(self):
        """Ordonne les nœuds pour que le signal circule de l'entrée vers la sortie."""
        # 1. Construire une liste d'adjacence simplifiée
        adj = {n_id: [] for n_id in self.nodes}
        in_degree = {n_id: 0 for n_id in self.nodes}
         
        for conn in self.connections.values():
            if conn.enabled:
                adj[conn.in_node].append(conn.out_node)
                in_degree[conn.out_node] += 1
            
        # 2. Algorithme de Kahn
        queue = [n_id for n_id in in_degree if in_degree[n_id] == 0]
        sorted_nodes = []
            
        while queue:
            u = queue.pop(0)
            sorted_nodes.append(u)
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
            
        return sorted_nodes


    def forward(self, input_vals, activation_func=sigmoid):
        input_nodes = sorted([n for n in self.nodes.values() if n.type=="input"], key=lambda x: x.id)
        bias_nodes = [n for n in self.nodes.values() if n.type=="bias"]


        if len(input_vals) !=len(input_nodes):
            raise ValueError(f"Attendu {len(input_nodes)} inputs, reçu {len(input_vals)}")
        

        # 1. Reset toutes les valeurs et assigner inputs/bias
        for node in self.nodes.values():
            node.value = 0.0

        for i, val in enumerate(input_vals):
            input_nodes[i].value = val

        for node in bias_nodes:
            node.value = 1.0      

        # 2. Calculer dans l'ordre topologique
        ordered_ids = self.compute_layers()

        for n_id in ordered_ids:
            node = self.nodes[n_id]
            
            if node.type not in ["input", "bias"]:
                node.value = activation_func(node.value)
            
            # Propager la valeur vers les nœuds suivants
            for conn in self.connections.values():
                if conn.enabled and conn.in_node == n_id:
                    self.nodes[conn.out_node].value += node.value * conn.weight

        # 3. Récupérer les sorties
        output_nodes = sorted([n for n in self.nodes.values() if n.type == "output"], key=lambda x: x.id)
        return [n.value for n in output_nodes]
        


class Species:
    def __init__(self, representative):
        self.representative = representative
        self.members = [representative]
        self.max_fitness = 0.0
        self.staleness = 0 # Compteur de stagnation

    def add_member(self, genome):
        self.members.append(genome)

    def update_stagnation(self):
            # On trouve le meilleur score actuel de l'espèce
            current_max = max([g.fitness for g in self.members])
            if current_max > self.max_fitness:
                self.max_fitness = current_max
                self.staleness = 0 # On remet à zéro, l'espèce progresse
            else:
                self.staleness += 1 # Pas de progrès cette génération

def get_genetic_distance(g1, g2, c1=1.0, c2=1.0, c3=0.4):
    innov1 = set(g1.connections.keys())
    innov2 = set(g2.connections.keys())
    
    all_innov = innov1.union(innov2)
    max_innov1 = max(innov1) if innov1 else 0
    max_innov2 = max(innov2) if innov2 else 0
    
    disjoint = 0
    excess = 0
    weight_diffs = []
    
    for i in all_innov:
        if i in innov1 and i in innov2:
            weight_diffs.append(abs(g1.connections[i].weight - g2.connections[i].weight))
        elif i in innov1:
            if i > max_innov2: excess += 1
            else: disjoint += 1
        elif i in innov2:
            if i > max_innov1: excess += 1
            else: disjoint += 1
            
    n = max(len(innov1), len(innov2), 1)
    
    avg_w = sum(weight_diffs) / len(weight_diffs) if weight_diffs else 0
    
    return (c1 * excess / n) + (c2 * disjoint / n) + (c3 * avg_w)



# Neat crossover

def neat_crossover(p1, p2):
    # L'enfant prend la topologie du parent le plus performant
    fitter, weaker = (p1, p2) if p1.fitness >= p2.fitness else (p2, p1)
    
    child = Genome()

    #copier les nodes
    for node in fitter.nodes.values():
        child.add_node(node.id, node.type)
    
    #crossover des connexions
    for innov, conn in fitter.connections.items():
        weight = conn.weight
        enabled = conn.enabled
        if innov in weaker.connections:
            if random.random() < 0.5: weight = weaker.connections[innov].weight
            if not conn.enabled or not weaker.connections[innov].enabled:
                if random.random() < 0.75: enabled = False # Probabilité de rester désactivé
        child.add_connection(conn.in_node, conn.out_node, weight, innov, enabled)        
    return child


def apply_mutations(genome, innovation_manager):
    if random.random() < 0.8: genome.mutate_weights(perturb_rate=0.8, step=0.5)
    if random.random() < 0.25: genome.mutate_add_connection(innovation_manager)
    if random.random() < 0.15: genome.mutate_add_node(innovation_manager)


def create_initial_population(size, nb_in, nb_out, inv_manager):
    pop = []

    # IDs GLOBAUX (créés une seule fois)
    in_ids = [inv_manager.get_node_id() for _ in range(nb_in)]
    out_ids = [inv_manager.get_node_id() for _ in range(nb_out)]
    bias_id = inv_manager.get_node_id()

    for _ in range(size):
        g = Genome()

        # mêmes IDs pour tout le monde
        for i_id in in_ids:
            g.add_node(i_id, "input")

        for o_id in out_ids:
            g.add_node(o_id, "output")

        g.add_node(bias_id, "bias")

        # connexions initiales
        for i_id in in_ids:
            for o_id in out_ids:
                inv = inv_manager.get_innovation_conn(i_id, o_id)
                g.add_connection(i_id, o_id, random.uniform(-1, 1), inv)
        
        for o_id in out_ids:
            inv = inv_manager.get_innovation_conn(bias_id, o_id)
            g.add_connection(bias_id, o_id, random.uniform(-1, 1), inv)

        pop.append(g)

    return pop


#******************EXEMPLE D'UTILISATION******************

# --- CONFIGURATION DU TEST XOR ---

def evaluate_fitness(genome):
    test_cases = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0),
    ]

    error = 0.0

    for inputs, expected in test_cases:
        output = genome.forward(inputs)[0]
        error += (output - expected) ** 2

    fitness = 4 - error  # max fitness = 4 (si erreur = 0)
    return fitness

def run_xor_test(generations=500, pop_size=150):
    manager = InnovationManager()
    population = create_initial_population(pop_size, 2, 1, manager)
    species_list = []
    
    print(f"Début de l'entraînement XOR sur {generations} générations...")
    
    for gen in range(generations):
        # 1. Évaluation et Spéciation
        for genome in population:
            genome.fitness = evaluate_fitness(genome)
            
            found = False
            for s in species_list:
                if get_genetic_distance(genome, s.representative) < 0.5:
                    s.add_member(genome)
                    found = True
                    break
            if not found:
                species_list.append(Species(genome))
        
        # Nettoyage des espèces vides
        species_list = [s for s in species_list if len(s.members) > 0]
        
        # 2. Stats
        best_genome = max(population, key=lambda g: g.fitness)
        if gen % 10 == 0 or best_genome.fitness > 3.9:
            print(f"Gen {gen} | Espèces: {len(species_list)} | Best Fitness: {best_genome.fitness:.4f}")
        
        #***debug****
        if gen % 50 == 0:
            print("Gen", gen, "best", best_genome.fitness)
            for x in [[0, 0], [0, 1], [1, 0], [1, 1]]:
                print(x, "->", best_genome.forward(x)[0])

        if best_genome.fitness > 3.95:
            print("--- Solution trouvée ! ---")
            break


        # 3. Reproduction
        ## 1. Mise à jour de la stagnation pour chaque espèce
        for s in species_list:
            s.update_stagnation()

        ## 2. Tri et filtrage des espèces (Staleness > 15 = suppression)
        # On garde toujours au moins 2 espèces pour éviter l'extinction totale
        if len(species_list) > 2:
            species_list = [s for s in species_list if s.staleness < 15]

        ## 3. Calcul de la reproduction (Fitness Partagée)
        # On s'assure que chaque membre a sa fitness ajustée
        for s in species_list:
            for g in s.members:
                g.adjusted_fitness = g.fitness / len(s.members)
            s.avg_fitness = sum(g.adjusted_fitness for g in s.members) / len(s.members)

        ## 4. Création de la nouvelle population
        total_avg_fitness = sum(s.avg_fitness for s in species_list)
        new_population = []
        new_population.append(copy.deepcopy(best_genome)) # Elitisme

        while len(new_population) < pop_size:
            if total_avg_fitness > 0:
                s = random.choices(species_list, weights=[s.avg_fitness for s in species_list])[0]
            else:
                s = random.choice(species_list)

            # 50% des meilleurs de l'espèce peuvent se reproduire
            s.members.sort(key=lambda g: g.fitness, reverse=True)
            pool = s.members[:max(1, len(s.members)//2)]
            
            if len(pool) > 1 and random.random() < 0.75:
                p1, p2 = random.sample(pool, 2)
                child = neat_crossover(p1, p2)
            else:
                child = copy.deepcopy(random.choice(pool))
            
            apply_mutations(child, manager)
            new_population.append(child)
            
        # Reset pour le tour suivant
        for s in species_list:
            s.representative = random.choice(s.members)
            s.members = []
        population = new_population

    # --- TEST FINAL DU CHAMPION ---
    print("\nRésultats du meilleur génome :")
    for x in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        res = best_genome.forward(x)[0]
        print(f"Entrée: {x} -> Sortie: {res:.4f} (Attendu: {x[0]^x[1]})")

if __name__ == "__main__":
    run_xor_test()    