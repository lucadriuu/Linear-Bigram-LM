import numpy as np
import pickle 
from MCsampling import token_freq, u_j_greedy, sequence, emp_matrix
from EMPloss import evo



def EMPsingle(d, alpha, betas, eta, seq_len, t_min, t_max):

    pi = token_freq(d, alpha)
    eta = 1/pi[0]
    data = {}

    for i, beta in enumerate(betas):
        
        u_jg, p_plus, p_min = u_j_greedy(d, pi, beta)
        seq = sequence(seq_len, d, pi, u_jg, p_min, p_plus)
        jump_probs = emp_matrix(seq, d)
        dynamics, t_steps = evo(t_min, t_max, seq, eta, d, jump_probs)
        
        data[f"beta_{beta}"] = dynamics
        if "t_steps" not in data:
            data["t_steps"] = t_steps

    filepath = fr"/home/lucadriu/Desktop/uni/Tesi/Final/experiments/all/empirical/data/single/sim{alpha}_{d}_{seq_len}.pkl"
    # Open and write directly to the specified filepath
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
        
    return



def EMPmulti(d, alpha, betas, eta, seq_len, t_min, t_max, n_runs):
    pi = token_freq(d, alpha)
    data={}

    for i, beta in enumerate(betas):

        runs_dynamics = []

        for run in range(n_runs):
            u_jg, p_plus, p_min = u_j_greedy(d, pi, beta)
            
            seq = sequence(seq_len, d, pi, u_jg, p_min, p_plus)
            
            jump_probs = emp_matrix(seq, d)
            
            dynamics, t_steps = evo(t_min, t_max, seq, eta, d, jump_probs)
            
            runs_dynamics.append(dynamics)
        
        runs_dynamics = np.array(runs_dynamics)
        mean_dynamics = np.mean(runs_dynamics, axis=0)
        std_dynamics = np.std(runs_dynamics, axis=0)

        data[f"beta_{beta}_mean"] = mean_dynamics
        data[f"beta_{beta}_std"] = std_dynamics

        if "t_steps" not in data:
            data["t_steps"] = t_steps

    filepath = fr"/home/lucadriu/Desktop/uni/Tesi/Final/experiments/all/empirical/data/single/multi/sim{alpha}_{d}_{seq_len}_{n_runs}.pkl"

    with open(filepath, "wb") as f:
        pickle.dump(data, f)

    return filepath