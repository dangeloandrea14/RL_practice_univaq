import utils
import numpy as np


class env:
    LAST_STATE = 6
    FIRST_STATE = 0
    DISCOUNT_FACTOR = 0.5


    def __init__(self, starting_grid = np.full((LAST_STATE-FIRST_STATE+1), 1/(LAST_STATE-FIRST_STATE+1))):
        self.states = [env.build_state(i) for i in range(env.FIRST_STATE,env.LAST_STATE+1)]
        self.starting_grid = starting_grid
        self.starting_state = env.build_state(utils.select_first_state(self.starting_grid))
        self.probability_matrix = utils.build_probability_matrix_states(self.states)
        self.rewards = env.build_rewards(self.states)
        print("Environment Ready.")        

    def state_right(state):
        return env.build_state(state.number + 1)

    def state_left(state):
        return env.build_state(state.number - 1)
    
    def stay(state):
        return state
        
    def jump(state):
        return env.build_state(env.LAST_STATE)

    def build_state(state_number):
        next_states = {}
        for next_state in env.probabilities[state_number]:
            if env.probabilities[state_number][next_state] > 0:
                next_states[next_state] = env.probabilities[state_number][next_state]

        rew = env.rewards[state_number] if state_number in env.rewards else 0

        return utils.simple_state(state_number, next_states, rew)

    probabilities = {0: {1: 0.4, 0 : 0.6},
                     1: {0: 0.4, 2: 0.4, 1 : 0.2},
                    2: {1: 0.4, 3: 0.4, 2 : 0.2},
                    3: {2: 0.4, 4: 0.4, 3 : 0.2},
                    4: {3: 0.4, 5: 0.4, 4 : 0.2},
                    5: {4: 0.4, 6: 0.4, 5 : 0.2},
                    6: {5: 0.4, 6 : 0.6}}
    
    rewards = {0:1, 6:10}

    def build_rewards(states):
        return [env.rewards[state.number] if state.number in env.rewards else 0 for state in states]

    
    def sample_episode(self, length):
        state = self.starting_state.number
        episode = [state]

        while len(episode) < length:
            next_state = utils.select_next_state(state, env.probabilities)
            state = next_state
            episode.append(state)

        return episode

