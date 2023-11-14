import utils
import numpy as np


class env:
    LAST_STATE = 6
    FIRST_STATE = 0
    DISCOUNT_FACTOR = 0.5


    def __init__(self, starting_grid = np.full((LAST_STATE-FIRST_STATE+1), 1/(LAST_STATE-FIRST_STATE+1))):
        self.states = [i for i in range(env.FIRST_STATE,env.LAST_STATE+1)]
        self.starting_grid = starting_grid
        self.starting_state = utils.select_first_state(self.starting_grid)
        self.probability_matrix = utils.build_probability_matrix_states(self.states, env.probabilities)
        self.rewards = env.build_rewards(self.states)
        print("Environment Ready.")        

    def state_right(state):
        return state + 1

    def state_left(state):
        return state - 1
    
    def stay(state):
        return state


    probabilities = {0: {1: 0.4, 0 : 0.6},
                     1: {0: 0.4, 2: 0.4, 1 : 0.2},
                    2: {1: 0.4, 3: 0.4, 2 : 0.2},
                    3: {2: 0.4, 4: 0.4, 3 : 0.2},
                    4: {3: 0.4, 5: 0.4, 4 : 0.2},
                    5: {4: 0.4, 6: 0.4, 5 : 0.2},
                    6: {5: 0.4, 6 : 0.6}}
    
    rewards = {0:1, 6:10}

    def build_rewards(states):
        return [env.rewards[state] if state in env.rewards else 0 for state in states]

    
    def sample_episode(self, length):
        state = self.starting_state
        episode = [state]

        while len(episode) < length:
            next_state = utils.select_next_state(state, env.probabilities)
            state = next_state
            episode.append(state)

        return episode

