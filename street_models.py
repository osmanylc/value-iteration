from itertools import product
import numpy as np


class StreetGrid:
    """
    Let's define the MDP tuple (S, A, R, T):

    States (S): The states are the points where we can make a decision, so 
        they are the nodes in our street grid. Let's represent our grid by 
        a 2-D numpy array. Our states will therefore be pairs of ints.
    Actions (A): A(s) will be an int representing {left, right, up ,down}. 
        If we're at the edge of the grid, moving toward the edge leaves us 
        in the same place.
    Rewards (R): R(s, a, s') will be a negative value representing how long it 
        takes us to traverse a specific street.
    Transition(T): T(s, a, s') represents the probability of actually doing 
        what we want to do. There is a small probability that we will make 
        the wrong turn in some corners.


    This MDP will be undiscounted, since we want to find the best path 
    without the bias of discounting.
    """

    def __init__(self, size):
        # Reward parameters. We model traffic as a log-normal RV
        self.high_traffic = 5
        self.med_traffic = 3
        self.low_traffic = 1
        self.high_sd = 5
        self.med_sd = 3
        self.low_sd = 1

        # States
        self.grid_size = size
        self.states = self.init_states()
        self.lost_epsilon = .01

        # Actions
        self.a = self.init_s_to_a()

        # Adjacency list
        self.adj = {}

    def p(self, s, a):
        """
        p(s', r | s, a)
        """
        ss = self.next_state(s, a)
        pass
        
    
    def r(self, s, a, ss):
        """
        R(r | s, a, s')
        """
        pass

    def next_state(self, s, a):
        """
        """
        x, y = s
        xx, yy = x, y

        if a == Action.UP:
            yy += 1
        elif a == Action.RIGHT:
            xx += 1
        elif a == Action.DOWN:
            yy -= 1
        elif a == Action.LEFT:
            xx -= 1
        else:
            raise Exception('Not a valid action.')

        assert 0 <= yy < self.grid_size and 0 <= xx < self.grid_size
        
        return xx, yy

    def a(self, s):
        return self.a[s]

    def init_s_to_a(self):
        def s_to_a(s):
            x, y = s
            a = []

            if x + 1 < self.grid_size:
                a.append(Action.RIGHT)
            elif 0 <= x - 1:
                a.append(Action.LEFT)
            elif y + 1 < self.grid_size:
                a.append(Action.UP)
            elif 0 <= y - 1:
                a.append(Action.DOWN)

            return a

        return [s_to_a(s) for s in self.states]

    def init_states(self):
        return list(product(range(self.grid_size), repeat=2))

    def check_bounds(self, s):
        x, y = s
        assert 0 <= x < self.grid_size and 0 <= y < self.grid_size


class Action:
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
