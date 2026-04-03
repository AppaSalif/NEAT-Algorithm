# NEAT-Algorithm
Solving the RL problem using Neuroevolution

# NEAT (NeuroEvolution of Augmenting Topologies)

This project implements a basic version of **NEAT**, an evolutionary algorithm for neural networks that evolves both **weights** and **topologies**.

## Overview

NEAT is a type of genetic algorithm that evolves neural networks instead of fixed architectures. It allows networks to **start small** and **grow** more complex over generations. NEAT is particularly useful for problems where the best network structure is unknown.

### Key Features

1. **Evolving Weights**: NEAT adjusts the weights of the network connections through mutation.
2. **Evolving Topology**: NEAT can add new nodes or new connections over time.
3. **Crossover**: Offspring networks inherit connections and nodes from parents.
4. **Speciation**: (Not implemented in this basic version) NEAT usually groups similar networks into species to protect innovation.

## Components

- **Node**: Represents a neuron. Each node has a unique ID, a type (`input`, `hidden`, `output`, or `bias`), and a value.
- **Connection**: Represents a directed connection between two nodes with a weight, an enabled flag, and a unique innovation number.
- **Genome**: A network of nodes and connections. Includes mutation methods:
  - `mutate_weights()`: Randomly perturbs or resets weights.
  - `mutate_add_connection()`: Adds a new connection between nodes.
  - `mutate_add_node()`: Splits an existing connection to add a new node.

## How It Works

1. **Initialization**: Start with a population of simple networks.
2. **Evaluation**: Compute fitness for each genome based on a task (e.g., XOR).
3. **Selection**: Select parents using tournament selection.
4. **Crossover**: Combine parents to produce offspring. NEAT prefers keeping the topology of the fitter parent and randomly inherits matching genes.
5. **Mutation**: Apply structural and weight mutations to offspring.
6. **Next Generation**: Repeat evaluation, selection, crossover, and mutation.
