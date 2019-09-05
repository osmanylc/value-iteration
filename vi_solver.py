import math
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from street_models import Action


def value_iteration(mdp):
    # ::::: Initialize value and policy functions
    v = init_values(mdp)
    pi = {}
    tolerance = 1e-2
    residual = math.inf
    total_iterations = 10

    # ::::: Iterations of VI
    num_it = 0
    while residual > tolerance or num_it < total_iterations:
        residual = 0
        num_it += 1
        print(f'::::: Starting iteration {num_it}')

        # ::::: Do value backup in each state
        for s in v.keys():
            if s == mdp.terminal_state:
                continue

            v_old = v[s]
            v_new = -math.inf

            # Get all actions possible in this state
            acts = mdp.actions[s]
            # Loop over all actions and find one with highest val
            for act in acts:
                v_temp = 0

                # Loop over all (ss, r) transitions and accumulate vals
                for (ss, r), p_sa in mdp.p(s, act).items():
                    v_temp += p_sa * (r + mdp.discount * v[ss])
                
                # Max of this action val with current highest
                v_new = max(v_new, v_temp)
            
            # Assign max action-value state value
            v[s] = v_new
            # Record max residual
            residual = max(residual, abs(v_new - v_old))

        print(f'residual: {residual:10.4f}')
    
    # ::::: Build the greedy policy
    print('\n::::: Building policy...')
    for s in v.keys():
        acts = mdp.actions[s]
        v_max = -math.inf
        a_max = None

        for act in acts:
            v_temp = 0

            for (ss, r), p_sa in mdp.p(s, act).items():
                v_temp += p_sa * (r + mdp.discount * v[ss])
            
            if v_temp > v_max:
                v_max = v_temp
                a_max = act

        pi[s] = a_max
    print('Finished building policy.')

    return v, pi


def init_values(mdp):
    v = {}

    for s in mdp.states:
        v[s] = 0
    
    return v


def d_to_arr(d):
    max_coords = max(d)
    arr_shape = tuple(x + 1 for x in max_coords)
    d_arr = np.zeros(arr_shape)

    for (s1, s2), val in d.items():
        d_arr[s1, s2] = val
    
    return d_arr


def visualize_values(v, title, fig_file):
    v_arr = d_to_arr(v)
    x_ran, y_ran = v_arr.shape

    x = np.arange(x_ran)
    y = np.arange(y_ran)

    X, Y = np.meshgrid(x, y)

    f = plt.figure()
    ax = f.gca(projection='3d')
    surf = ax.plot_surface(X, Y, v_arr, cmap=cm.viridis)
    f.colorbar(surf)

    plt.title(title)
    plt.ylabel('y')
    plt.xlabel('x')
    f.savefig(fig_file)


def visualize_values_2d(v, title, fig_file):
    v_arr = d_to_arr(v)

    f, _ = plt.subplots()
    plt.imshow(v_arr, origin='lower', cmap='viridis')

    plt.title(title)
    plt.ylabel('y')
    plt.xlabel('x')
    f.savefig(fig_file)


def visualize_traffic(mdp, fig_file):
    """
    Color the nodes depending on how much traffic they get. 

    High = RED, Medium = YELLOW, Low = GREEN
    """
    # Convert traffics to integers for plotting
    success_prob_to_idx = {.9: 0, .7: 1, .5: 2}
    idxs = np.vectorize(lambda x: success_prob_to_idx[x])(mdp.state_traffics)

    # Plot the traffic values
    f, _ = plt.subplots()
    plt.imshow(idxs, cmap='RdYlGn', origin='lower')

    plt.title('Node Traffic Congestion')
    plt.xlabel('x')
    plt.ylabel('y')
    f.savefig(fig_file)


def visualize_policy(pi, mdp, fig_file):
    """
    Draw an arrow at every cell representing the turn we want to make.
    """
    a_to_vec = {Action.UP: (0,1), Action.RIGHT: (1,0), 
        Action.DOWN: (0,-1), Action.LEFT: (-1,0)}

    pi_arr = d_to_arr(pi)
    x_len, y_len = pi_arr.shape
    U = np.zeros_like(pi_arr)
    V = np.zeros_like(pi_arr)

    # Conver policy to arrow directions with (u,v)
    for x, y in product(range(x_len), range(y_len)):
        U[x,y] = a_to_vec[pi_arr[x,y]][0]
        V[x,y] = a_to_vec[pi_arr[x,y]][1]

    # Erase arrow for terminal state
    x_term, y_term = mdp.terminal_state
    U[x_term, y_term] = 0
    V[x_term, y_term] = 0
    
    X, Y = np.meshgrid(range(x_len), range(y_len), indexing='ij')

    # Traffic fig to overlay
    success_prob_to_idx = {.9: 0, .7: 1, .5: 2}
    idxs = np.vectorize(lambda x: success_prob_to_idx[x])(mdp.state_traffics)

    # Plot traffic + policy on top
    f, ax = plt.subplots()
    plt.imshow(idxs, cmap='RdYlGn', origin='lower')
    plt.quiver(X, Y, U, V)

    plt.title('Policy on Traffic Grid')
    plt.xlabel('x')
    plt.ylabel('y')
    f.savefig(fig_file)
