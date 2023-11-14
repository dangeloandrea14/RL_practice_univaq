import numpy as np
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt


class state:
    def __init__(self, number, actions, reward):
        self.number = number
        self.actions = actions
        self.reward = reward

class simple_state:
    def __init__(self, number, next_states, reward):
        self.number = number
        self.next_states = next_states
        self.reward = reward


class world:
    def __init__(self, states, transitions):
        self.states = states
        self.transitions = transitions
        self.current_state = 0
        self.current_action = 0
        self.current_reward = 0
        self.current_done = False

def build_probability_matrix(states):

    matrix = np.zeros((len(states), len(states)))

    for state in states:
        for action in state.actions:
            matrix[state.number][action(state).number] = state.actions[action]

    return matrix


def build_probability_matrix_states(states):

    matrix = np.zeros((len(states), len(states)))

    for state in states:
        for ns in state.next_states:
            matrix[state.number][ns] = state.next_states[ns]

    return matrix


def select_next_move(state):
    action_list = state.actions
    action_probabilities = [action_list[action] for action in action_list]
    action = np.random.choice(list(action_list.keys()), p=action_probabilities)
    return action

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

