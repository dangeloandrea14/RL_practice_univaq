import utils
import numpy as np
import pandas as pd
from env import env

class agent:
    def __init__(self,starting_policy):
        self.policy = starting_policy



    
    actions = {'left': env.action_left, 'right': env.action_right, 'stay': env.stay}
