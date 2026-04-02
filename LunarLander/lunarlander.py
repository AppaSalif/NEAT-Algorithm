import random
import math
import copy


# Fonction d'activation (sigmoïde)
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


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
    def mutate_weights(self, perturb_rate=0.8, step=0.1):
        for conn in self.connections.values():
            if random.random() < perturb_rate:
                # petite variation
                #si les poids changaient grandement, le reseau serait instable -> difficile de faire une optimisation
                conn.weight +=random.uniform(-step, step)
            else:
                # nouveau poids aléatoire
                conn.weight =random.uniform(-1., 1.)


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
        for i, node in enumerate(bias_nodes):
            node.value = 1.0      

        # Reset hidden/output nodes
        for n in hidden_nodes + output_nodes:
            n.value = 0.0

        #------calcul hidden
        for conn in self.connections.values():
            if conn.enabled:
                if self.nodes[conn.out_node].type == "hidden":
                    in_val = self.nodes[conn.in_node].value
                    self.nodes[conn.out_node].value += in_val*conn.weight

        for n in hidden_nodes:
            n.value = sigmoid(n.value)

        #-------calcul output
        for conn in self.connections.values():
            if conn.enabled:
                if self.nodes[conn.out_node].type == "output":
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


# Créer une population (avec des génomes basiques)
# Genome = 
        # 2 inputs
        # 1 hidden
        # 1 bias
        # 1 output
        # connexions directes input -> output

def create_genome():
    g = Genome()

    n1 = g.add_node("input")
    n2 = g.add_node("input")

    b  = g.add_node("bias")

    h1 = g.add_node("hidden")
    o  = g.add_node("output")

    # input -> hidden
    g.add_connection(n1, h1)
    g.add_connection(n2, h1)
    g.add_connection(b,  h1)


    # hidden -> output
    g.add_connection(h1, o)

    # bias -> output
    g.add_connection(b, o)    

    return g

def create_population(size):
    return [create_genome() for _ in range(size)]

def copy_genome(genome):
    return copy.deepcopy(genome)

def evolve(population, generations=20):
    for gen in range(generations):
        # Évaluer fitness
        scored = [(genome, evaluate_fitness(genome)) for genome in population]

        # Trier
        scored.sort(key=lambda x: x[1], reverse=True)

        best_fitness = scored[0][1]
        print(f"Génération {gen} - Best fitness: {best_fitness}")

        # Garder les meilleurs
        survivors = [g for g, f in scored[:len(population)//2]]

        # Reproduire (cloner + muter)
        new_population = []
        for g in survivors:
            new_population.append(g)  # garder original

            child = copy_genome(g)
            child.mutate_weights()
            new_population.append(child)

        population = new_population

    return population

pop = create_population(20)
pop = evolve(pop, generations=30)    