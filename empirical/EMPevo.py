import numpy as np
import pickle 
from MCsampling import token_freq, u_j_greedy, sequence, bigram_stats
from EMPloss import emp_evo, pop_evo, emp_loss, make_t_steps



def EMPsingle(d, alpha, betas, eta, seq_len, t_min, t_max, n_seeds = 1, seed = None):

    pi = token_freq(d, alpha)
    rng = np.random.default_rng(seed)
    t_steps = make_t_steps(t_min, t_max)
    data = {"alpha": alpha, "d": d, "seq_len": seq_len, "betas": betas, "n_seeds": n_seeds,
            "t_steps": t_steps}

    for beta in betas:

        u_jg, p_plus, p_min = u_j_greedy(d, pi, beta)                               #deterministic: only the sequence changes across seeds
        runs = []

        for _ in range(n_seeds):
            seq = sequence(seq_len, d, pi, u_jg, p_min=p_min, p_plus=p_plus, rng=rng)
            marg, row_sq_sum = bigram_stats(seq, d)                                 #marg over the seq_len - 1 input tokens, no d x d matrix
            runs.append(emp_evo(t_steps, eta, marg, row_sq_sum))                    #same marg for eta and loss: eta*marg <= 1

        runs = np.array(runs)
        data[f"beta_{beta}"] = runs.mean(axis=0)
        data[f"std_{beta}"] = runs.std(axis=0)
        data[f"pop_{beta}"] = pop_evo(t_steps, eta, pi, u_jg, beta)              #population reference at this d, eta = 1/pi_1

    filepath = fr"/home/lucadriu/Desktop/uni/Tesi/Final/experiments/all/empirical/data/single/sim{alpha}_{d}_{seq_len}.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

    return filepath



def EMPdistance(d, alpha, beta, eta, ratios, t_min, t_max, n_seeds=1, seed=None):
    pi = token_freq(d, alpha)
    u_jg, p_plus, p_min = u_j_greedy(d, pi, beta)    
    t_steps = make_t_steps(t_min, t_max)
    rng = np.random.default_rng(seed)

    pop_dyn = pop_evo(t_steps, eta, pi, u_jg, beta)          #sequence independent: computed once

    data = {"alpha": alpha, "d": d, "beta": beta, "ratios": ratios, "n_seeds": n_seeds,
            "eta": eta, "t_steps": t_steps, "pop_dynamics": pop_dyn}

    for ratio in ratios:
        seq_len = int(ratio * d)
        runs = []

        for _ in range(n_seeds):
            seq = sequence(seq_len, d, pi, u_jg, p_min=p_min, p_plus=p_plus, rng=rng)
            marg, row_sq_sum = bigram_stats(seq, d)
            emp_dyn = emp_evo(t_steps, eta, marg, row_sq_sum)
            runs.append(np.abs(emp_dyn - pop_dyn))

        runs = np.array(runs)
        data[f"ratio_{ratio}"] = runs.mean(axis=0)
        data[f"std_{ratio}"] = runs.std(axis=0)

    filepath = fr"/home/lucadriu/Desktop/uni/Tesi/Final/experiments/all/empirical/data/distance/dist{alpha}_{d}_{beta}.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

    return filepath

