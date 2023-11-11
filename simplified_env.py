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
        self.probability_matrix = utils.build_probability_matrix(self.states)
        self.rewards = env.build_rewards(self.states)
        print("Environment Ready.")        

    def move_right(state):
        return env.build_state(state.number + 1)

    def move_left(state):
        return env.build_state(state.number - 1)
    
    def stay(state):
        return state
        
    def jump(state):
        return env.build_state(env.LAST_STATE)

    def build_state(state_number):
        possible_moves = {}
        for move in env.probabilities[state_number]:
            if env.probabilities[state_number][move] > 0:
                possible_moves[move] = env.probabilities[state_number][move]

        rew = env.rewards[state_number] if state_number in env.rewards else 0

        return utils.state(state_number, possible_moves, rew)

    probabilities = {0: {move_left: 0.0, move_right: 0.4, stay : 0.6},
                     1: {move_left: 0.4, move_right: 0.4, stay : 0.2},
                    2: {move_left: 0.4, move_right: 0.4, stay : 0.2},
                    3: {move_left: 0.4, move_right: 0.4, stay : 0.2},
                    4: {move_left: 0.4, move_right: 0.4, stay : 0.2},
                    5: {move_left: 0.4, move_right: 0.4, stay : 0.2},
                    6: {move_left: 0.4, move_right: 0.0, stay : 0.6}}
    
    rewards = {0:1, 6:10}

    def build_rewards(states):
        return [env.rewards[state.number] if state.number in env.rewards else 0 for state in states]

    
    def sample_episode(self, length):
        state = self.starting_state
        episode = [state.number]

        while len(episode) < length:
            next_action = utils.select_next_move(state)
            state = next_action(state)
            episode.append(state.number)

        return episode

