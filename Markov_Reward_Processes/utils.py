import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_probability_matrix_states(states, probabilities):

    matrix = np.zeros((len(states), len(states)))

    for state in states:
        for ns in probabilities[state]:
            matrix[state][ns] = probabilities[state][ns]

    return matrix


def select_next_state(state, probabilities):

    possible_moves = probabilities[state]
    next_state = np.random.choice(list(possible_moves.keys()), p=list(possible_moves.values()))
   
    return next_state

def select_first_state(grid):
    pos = list(range(len(grid)))
    state = np.random.choice(pos, p=grid)
    return state

def plot_heatmap(result):
    
    cmap = plt.get_cmap('viridis')
    normed_values = (result - np.min(result)) / (np.max(result) - np.min(result))
    fig, ax = plt.subplots(figsize=(7, 1))
    im = plt.imshow(normed_values.reshape(1, -1), cmap=cmap, aspect='auto', extent=[0, len(result), 0, 1])

    for i, value in enumerate(result):
        plt.text(i + 0.5, 0.5, f'{value:.3f}', color='red', ha='center', va='center')

    plt.axis('off')

    return im

def return_function(episode, rewards, discount_factor):
    total = 0
    for i in range(len(episode)):
        total += rewards[episode[i]] * (discount_factor ** i)
    return total

