import math

def value_iteration(mdp):
    # ::::: Initialize value and policy functions
    v = init_values(mdp)
    pi = init_policy(mdp)
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
            v_old = v[s]
            v_new = -math.inf

            # Get all actions possible in this state
            acts = mdp.s_to_a(s)
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
    
    # ::::: Build the greedy policy
    for s in pi.keys():
        acts = mdp.s_to_a(s)
        v_max = -math.inf
        a_max = None

        for act in acts:
            v_temp = 0

            for (ss, r) , p_sa in mdp.p(s, act).items():
                v_temp += p_sa * (r + mdp.discount * v[ss])
            
            if v_temp > v_max:
                v_max = v_temp
                a_max = act

        pi[s] = a_max

    return v, pi


def init_values(mdp):
    pass


def init_policy(mdp):
    pass
