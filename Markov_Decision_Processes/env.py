import utils
import numpy as np
import pandas as pd

class env:
    LAST_STATE = 6
    FIRST_STATE = 0
    DISCOUNT_FACTOR = 0.5


    def __init__(self, starting_grid = np.full((LAST_STATE-FIRST_STATE+1), 1/(LAST_STATE-FIRST_STATE+1))):
        
        ## STATES AND ACTIONS
        self.states = [env.build_state(i) for i in range(env.FIRST_STATE,env.LAST_STATE+1)]

        self.actions = []
        for action in env.actions:
            self.actions.append(env.build_action(env.actions[action], action))

        ## STARTING STATE

        self.starting_grid = starting_grid
        self.starting_state = env.build_state(utils.select_first_state(self.starting_grid))
        
        ## PROBABILITY MATRIX AND REWARDS

        self.probability_matrix, self.stateaction_dict = utils.build_probability_matrix(self.states, self.actions)
        self.probability_matrix_df = utils.build_df(self.probability_matrix, self.stateaction_dict)
        self.rewards = env.build_rewards(self.states)
        
        print("Environment Ready.")        

    def action_right(state):
        return {env.build_state(state.number + 1) : 1}

    def action_left(state):
        return {env.build_state(state.number - 1) : 1}
    
    def stay(state):
        return {state : 1}
        
    def jump(state):
        return { env.build_state(env.LAST_STATE) : 0.5, env.build_state(env.FIRST_STATE) : 0.5}
    
    def build_rewards(states):
        return [env.rewards[state] if state in env.rewards else 0 for state in states]
    
    def build_action(function, name):
        return utils.action(function, name)

    def build_state(state_number):
        action_list = []

        for action in env.actions:
            if env.actions[action] in env.probabilities[state_number] and env.probabilities[state_number][env.actions[action]] > 0:
                action_list.append( env.build_action( env.actions[action] ,action ) )


        rew = env.rewards[state_number] if state_number in env.rewards else 0

        return utils.state(state_number, action_list, rew)


    actions = {'left': action_left, 'right': action_right, 'stay': stay}


    probabilities = {0: {action_left: 0.0, action_right: 0.4, stay : 0.6},
                     1: {action_left: 0.4, action_right: 0.4, stay : 0.2},
                    2: {action_left: 0.4, action_right: 0.4, stay : 0.2},
                    3: {action_left: 0.4, action_right: 0.4, stay : 0.2},
                    4: {action_left: 0.4, action_right: 0.4, stay : 0.2},
                    5: {action_left: 0.4, action_right: 0.4, stay : 0.2},
                    6: {action_left: 0.4, action_right: 0.0, stay : 0.6}}
    

    
    rewards = {0:1, 6:10}

    
    def sample_episode(self, length):
        state = self.starting_state
        episode = [state.number]

        while len(episode) < length:
            next_action = utils.select_next_move(state)
            state = next_action(state)
            episode.append(state.number)

        return episode

