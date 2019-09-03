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

    def __init__(self, size, seed=0):
        # Reward parameters. We model traffic as geometric RV 
        self.high_traffic = .5
        self.med_traffic = .7
        self.low_traffic = .9

        # States
        self.grid_size = size
        self.states = self.init_states()
        self.terminal_state = self.init_terminal_state()
        self.lost_epsilon = .01

        # Actions
        self.actions = self.init_s_to_a()

        # Reward function
        self.state_traffics = self.init_state_traffics()
        self.r = self.r_geo
        self.r_dist = self.build_dist(
            [self.high_traffic, self.med_traffic, self.low_traffic])
        self.discount = 1

        # Set random seed
        np.random.seed(seed)

    def build_dist(self, params, high=15):
        dist = {}

        for param in params:
            dist[param] = [StreetGrid.geometric_pdf(param, x) for x in range(high)]
            dist[param].append(1 - StreetGrid.geometric_cdf(param, high - 1))

        return dist

    def p(self, s, a):
        """
        p(s', r | s, a)
        """
        p_dist = {}
        ss = self.next_state(s, a)
        r_dist = self.r_geo(s, a, ss)

        for x in range(len(r_dist)):
            rw = -x
            p_dist[(ss, rw)] = r_dist[x]
        
        return p_dist

    @staticmethod
    def geometric_pdf(p, x):
        return p * (1 - p) ** x
    
    @staticmethod
    def geometric_cdf(p, x):
        return sum(StreetGrid.geometric_pdf(p, y) for y in range(x + 1))
    
    def r_geo(self, s, a, ss):
        """
        R(r | s, a, s')
        """
        assert ss == self.next_state(s, a)
        x, y = ss
        param = self.state_traffics[x, y]

        return self.r_dist[param]

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
        return self.actions[s]

    def init_s_to_a(self):
        def s_to_a(s):
            x, y = s
            a = []

            if x + 1 < self.grid_size:
                a.append(Action.RIGHT)
            if 0 <= x - 1:
                a.append(Action.LEFT)
            if y + 1 < self.grid_size:
                a.append(Action.UP)
            if 0 <= y - 1:
                a.append(Action.DOWN)

            return a

        return {s: s_to_a(s) for s in self.states}

    def init_states(self):
        return list(product(range(self.grid_size), repeat=2))

    def init_terminal_state(self):
        return (self.grid_size - 1, self.grid_size - 1)

    def init_state_traffics(self):
        # Get the number of rows and columns
        nrows, ncols = self.grid_size, self.grid_size

        # Populate the array with traffic values sampled uniformly at random
        traffics = [self.low_traffic, self.med_traffic, self.high_traffic]
        state_traffics = np.random.randint(len(traffics), size=(nrows, ncols))
        traffics_mapper = lambda x: traffics[x]
        arr_mapper = np.vectorize(traffics_mapper)

        state_traffics = arr_mapper(state_traffics)

        return state_traffics

    def check_bounds(self, s):
        x, y = s
        assert 0 <= x < self.grid_size and 0 <= y < self.grid_size


class Action:
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
