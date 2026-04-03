import random
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx



# Fonction d'activation (sigmoïde)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Node d'un réseau
class Node:
    def __init__(self, id, type):
        self.id = id            # unique node id
        self.type = type        # "input", "hidden", "output"
        self.value = 0.0        #

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
        self.nodes = {}            # dict id: Node
        self.connections = {}      # dict innovation: Connection
        self.next_node_id = 0
        self.next_innovation = 0
        self.fitness = 0.0

    def add_node(self, type):
        node = Node(self.next_node_id, type)
        self.nodes[self.next_node_id] = node
        self.next_node_id += 1
        return node

    def add_connection(self, in_node, out_node, weight=None):
        if weight is None:
            weight = random.uniform(-1.0, 1.0)
        conn = Connection(in_node.id, out_node.id, weight, True, self.next_innovation)
        self.connections[self.next_innovation] = conn
        self.next_innovation += 1
        return conn

    # Mutation des poids des connexions
    def mutate_weights(self, perturb_rate=0.8, step=0.3):
        for conn in self.connections.values():
            if random.random() < perturb_rate:
                conn.weight +=random.uniform(-step, step)
            else:
                # nouveau poids aléatoire
                conn.weight =random.uniform(-1., 1.)

    def mutate_add_connection(self, max_attempts=10):
        nodes = list(self.nodes.values())

        for _ in range(max_attempts):
            n1 = random.choice(nodes)
            n2 = random.choice(nodes)

            if n1.type == "output" or n2.type == "input":
                continue

            if n1.id == n2.id:
                continue

            # Vérifier si la connexion existe déjà
            existing = False
            for conn in self.connections.values():
                if conn.in_node == n1.id and conn.out_node == n2.id:
                    existing = True
                    break
            
            if existing:
                continue

            # Ajouter la connexion
            self.add_connection(n1, n2)
            return True
        
        return False


    def mutate_add_node(self):
        #choisir une connexion active au hasard
        enabled_cons = [c for c in self.connections.values() if c.enabled]

        if not enabled_cons:
            return False

        conn = random.choice(enabled_cons)
        conn.enabled = False

        #créer le nouveau node
        new_node = self.add_node("hidden")

        # créer les deux nouvelles connexions 

        # A -> new_node (poids = 1)
        self.add_connection(self.nodes[conn.in_node], new_node, weight=1.0)
        
        # new_node -> B (poids = weight de l'ancienne connexion)
        self.add_connection(new_node, self.nodes[conn.out_node], weight=conn.weight)

        return True
        

    def forward(self, input_vals):
        input_nodes = [n for n in self.nodes.values() if n.type=="input"]
        bias_nodes = [n for n in self.nodes.values() if n.type=="bias"]
        hidden_nodes = [n for n in self.nodes.values() if n.type == "hidden"]
        output_nodes = [n for n in self.nodes.values() if n.type == "output"]


        if len(input_vals) !=len(input_nodes):
            raise ValueError("Nombre d'inputs incorrect")
        
        #Assign inputs
        for i, node in enumerate(input_nodes):
            node.value = input_vals[i]


        #Assign bias
        for node in bias_nodes:
            node.value = 1.0      

        # Reset hidden/output nodes
        for n in hidden_nodes + output_nodes:
            n.value = 0.0

        #------calcul hidden
        for conn in self.connections.values():
            if conn.enabled and self.nodes[conn.out_node].type == "hidden":
                in_val = self.nodes[conn.in_node].value
                self.nodes[conn.out_node].value += in_val*conn.weight

        for n in hidden_nodes:
            n.value = sigmoid(n.value)

        #-------calcul output
        for conn in self.connections.values():
            if conn.enabled and self.nodes[conn.out_node].type == "output":
                in_val = self.nodes[conn.in_node].value
                self.nodes[conn.out_node].value += in_val*conn.weight

        for n in output_nodes:
            n.value = sigmoid(n.value)                    
        

        return [n.value for n in output_nodes]



#**********premier test (Création d'un Genome)
genome = Genome()
n1 = genome.add_node("input")
n2 = genome.add_node("input")
b = genome.add_node("bias")
h1 = genome.add_node("hidden")
o1 = genome.add_node("output")

genome.add_connection(n1, h1, weight=0.5)
genome.add_connection(n2, h1, weight=-0.4)
genome.add_connection(b,  h1, 0.8) #bias

genome.add_connection(h1, o1, 1.2)
genome.add_connection(b, o1, weight=-0.3) #bias

inputs = [1.0, 0.5]
outputs = genome.forward(inputs)

print("Inputs :", inputs)
print("Outputs :", outputs)


#***********deuxième test (mutation des poids)
print("Avant mutation:")
for c in genome.connections.values():
    print(c.weight)

genome.mutate_weights()

print("\nAprès mutation:")
for c in genome.connections.values():
    print(c.weight)

print("Perturbations -> Outputs :", genome.forward(inputs) )

#***********Troisième test (resolution du probleme XOR)

# definition de la fonction fitness
def evaluate_fitness(genome):
    test_cases = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0),
    ]

    error = 0.

    for inputs, expected in test_cases:
        output = genome.forward(inputs)
        error += (output[0] - expected)**2

    fitness = 4 - error   # max fitness = 4 (si error = 0)
    return fitness


# Créer une population

def create_genome():
    g = Genome()

    n1 = g.add_node("input")
    n2 = g.add_node("input")

    b  = g.add_node("bias") 
    
    o  = g.add_node("output")

    # connexions directes
    g.add_connection(n1, o)
    g.add_connection(n2, o)
    g.add_connection(b, o)    

    return g


def create_population(size):
    return [create_genome() for _ in range(size)]



# GA OPERATORS
# ----------------------------
def tournament_selection(population, k):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(range(len(population)), k)
        winner = max(tournament, key=lambda i: population[i].fitness)
        selected.append(winner)
    return selected

def crossover(parent1, parent2):
    child = copy.deepcopy(parent1)

    for innov in child.connections:
        if innov in parent2.connections:
            if random.random() < 0.5:
                child.connections[innov].weight = parent2.connections[innov].weight

    return child

# Neat crossover

def neat_crossover(parent1, parent2):
    # L'enfant prend la topologie du parent le plus performant
    if parent1.fitness >= parent2.fitness:
        fitter = parent1
        weaker = parent2
    else:
        fitter = parent2
        weaker = parent1
    
    child = Genome()
    child.next_node_id = max(parent1.next_node_id, parent2.next_node_id)
    child.next_innovation = max(parent1.next_innovation, parent2.next_innovation)

    #copier les nodes
    for node_id, node in fitter.nodes.items():
        child.nodes[node_id] = Node(node.id, node.type)
    
    #crossover des connexions
    for innov, conn in fitter.connections.items():
        if innov in weaker.connections:
            # Matching gene, choisir aléatoirement les poids
            chosen_weight = random.choice([conn.weight, weaker.connections[innov].weight])
        else:
            # Disjoint/excess gene, prendre le poids du fitter
            chosen_weight = conn.weight
        
        child.connections[innov] = Connection(conn.in_node, conn.out_node, chosen_weight, conn.enabled, conn.innovation)        
    return child

# EVOLUTION
# ----------------------------
def evolve(population, generations=50, mutation_rate=0.2, crossover_rate=0.8, tournament_size=5):

    pop_size = len(population)

    for gen in range(generations):

        # Évaluation
        for genome in population:
            genome.fitness = evaluate_fitness(genome)

        # Stats
        best = max(population, key=lambda g: g.fitness)
        
        if gen%10 ==0:
            print(f"Génération {gen} - Best fitness: {best.fitness}")

        # Sélection
        parent_indices = tournament_selection(population, tournament_size)
        parents = [population[i] for i in parent_indices]

        # Nouvelle population
        new_population = []

        # Élitisme
        sorted_pop = sorted(population, key=lambda g: g.fitness, reverse=True)
        new_population.append(copy.deepcopy(sorted_pop[0]))
        new_population.append(copy.deepcopy(sorted_pop[1]))

        # Reproduction
        while len(new_population) < pop_size:
            p1 = random.choice(parents)
            p2 = random.choice(parents)

            # crossover NEAT
            if random.random() < crossover_rate:
                child = neat_crossover(p1, p2)
            else:
                child = copy.deepcopy(p1)

            # mutation des poids
            if random.random() < mutation_rate:
                child.mutate_weights()

            # Mutation structurelle rare: ajout d'une connexion
            if random.random() < 0.06:
                child.mutate_add_connection()

            # Mutation structurelle rare: ajout d'un node
            if random.random() < 0.02:
                child.mutate_add_node()


            new_population.append(child)

        population = new_population

    return population


pop = create_population(100)
pop = evolve(pop, generations=300)

