import numpy as np
import pickle 
from datastruc_test import struc


def tokens_struc(v_size, alpha, greedy, flip_mp_pl, flip_mp_hyb, n_passes, seed):
    data_struc = struc(v_size, alpha, seed)
    pi = data_struc.token_freq(v_size)

    if greedy == True:
        ui, _ = data_struc.u_j_greedy(v_size)                                #we can consider u_i sampled uniform or sampled by flipping                       
    if flip_mp_hyb == True:
        ui, cs_list = data_struc.u_j_flip_hyb(v_size, n_passes)    
    if flip_mp_pl == True:
        ui, cs_list = data_struc.u_j_flip_plain(v_size, n_passes)

    return pi, ui

def components(pi, ui, beta, eta, t = None):

    expo = np.exp(- (eta * pi * t))

    nu_t = (pi * ui) @ expo
    mu_t = 1 - (pi @ expo)

    lambda1_t = (mu_t*(1 + beta) + np.hypot(mu_t * (1 - beta), 2 * np.sqrt(beta) * nu_t))/2
    lambda2_t = (mu_t*(1 + beta) - np.hypot(mu_t * (1 - beta), 2 * np.sqrt(beta) * nu_t))/2

    comp2vec1 = (mu_t * nu_t) / (beta * mu_t - lambda1_t)
    comp1vec2 = (mu_t * beta * nu_t) / (mu_t - lambda2_t)

    norm1 = np.hypot(mu_t, comp2vec1)                           #normalization of the eigenevectors 
    norm2 = np.hypot(mu_t, comp1vec2)

    mu_t_norm1 = mu_t / norm1                                   #Second component of V+
    mu_t_norm2 = mu_t / norm2                                   #First component of V-

    comp2vec1_norm = comp2vec1 / norm1                          #First component of V+
    comp1vec2_norm = comp1vec2 / norm2                          #Second component of V-

    return comp2vec1_norm, comp1vec2_norm, mu_t_norm1, mu_t_norm2



def evolution(alpha, d, pi, ui, eta, b_step, b_min, b_max, t_in, t_fin, nsteps, init_norm, logspacing):
    c1v2n_bs = {}                                  # evolutions for multiple betas
    c2v1n_bs = {}
    mu_tn1_bs = {}
    mu_tn2_bs = {}

    if logspacing == True:
        t_ax = np.geomspace(t_in, t_fin)
    else:
        t_ax = np.arange(t_in, t_fin, nsteps)

    for b in np.arange (b_min, b_max, b_step):
        c1v2n_bf = []                               #beta fixed
        c2v1n_bf = []
        mu_tn1_bf = []
        mu_tn2_bf = []

        if init_norm == True:

            #initial values in order to normalize at t=0
            c2v1n_in, c1v2n_in, mu_tn1_in, mu_tn2_in = components(pi, ui, b, eta, t= t_in)

            for t in t_ax:
                c2v1n, c1v2n, mu_tn1, mu_tn2 = components(pi, ui, b, eta, t=t)
                c2v1n_bf.append(c2v1n / c2v1n_in)
                c1v2n_bf.append(c1v2n / c1v2n_in)
                mu_tn1_bf.append(mu_tn1 / mu_tn1_in)
                mu_tn2_bf.append(mu_tn2 / mu_tn2_in)

        else:

            for t in t_ax:
                c2v1n, c1v2n, mu_tn1, mu_tn2 = components(pi, ui, b, eta, t=t)
                c2v1n_bf.append(c2v1n)
                c1v2n_bf.append(c1v2n)
                mu_tn1_bf.append(mu_tn1)
                mu_tn2_bf.append(mu_tn2)


        c2v1n_bs[b] = c2v1n_bf
        c1v2n_bs[b] = c1v2n_bf
        mu_tn1_bs[b] = mu_tn1_bf
        mu_tn2_bs[b] = mu_tn2_bf

    sim_data = {
                'time': t_ax, 
                'comp2_vec1_norm': c2v1n_bs, 
                'comp1_vec2_norm': c1v2n_bs, 
                'mu1_t_norm': mu_tn1_bs, 
                'mu2_t_norm': mu_tn2_bs 
    }

    file_path = f"/home/lucadriu/Desktop/uni/Tesi/LM_bigram/eigen_dynam/dynam_test/evo_data/eigenvecs_evo({alpha}_{d}).pkl"

    # Save the dictionary in a file
    with open(file_path, 'wb') as file:
        pickle.dump(sim_data, file)
        
    return sim_data



def multi_evo(d, n_runs, alpha, eta, flip_mp_hyb, b_step, b_min, b_max, t_in, t_fin, nsteps, init_norm, logspacing):
    ss = np.random.SeedSequence(entropy = None)
    seeds = ss.generate_state(n_runs)


    all_c2v1n = []
    all_c1v2n = []

    b_ax = np.arange(b_min, b_max, b_step)

    if logspacing == True:
        t_ax = np.geomspace(t_in, t_fin)
    else:
        t_ax = np.arange(t_in, t_fin, nsteps)

    for i, seed in enumerate(seeds):
        data_struc = struc(d, alpha, seed = int(seed))
        pi = data_struc.token_freq(d)

        if flip_mp_hyb == True:

            ui, _ = data_struc.u_j_flip_hyb(d, n_passes = 10)   
        else:
            ui, _ = data_struc.u_j_flip_plain(d, n_passes = 10)

        if ui[0] == +1 and ui[1] == -1 and ui[2] == -1:                                                   #check the effect of considering \pm 1
            sim_data = evolution(alpha, d, pi, ui, eta, b_step, b_min, b_max, t_in, t_fin, nsteps, init_norm, logspacing)
            c2v1n_matrix = [sim_data['comp2_vec1_norm'][b] for b in b_ax]
            c1v2n_matrix = [sim_data['comp1_vec2_norm'][b] for b in b_ax]
            
            all_c2v1n.append(c2v1n_matrix)
            all_c1v2n.append(c1v2n_matrix)

    c2v1n_mean = np.mean(np.array(all_c2v1n), axis = 0)
    c2v1n_std = np.std(np.array(all_c2v1n), axis = 0)
    c1v2n_mean = np.mean(np.array(all_c1v2n), axis = 0)
    c1v2n_std = np.std(np.array(all_c1v2n), axis = 0)


    stats_data = {
        'time': t_ax,
        'beta': b_ax,
        'comp2_vec1_norm_mean': c2v1n_mean,
        'comp2_vec1_norm_std':  c2v1n_std,
        'comp1_vec2_norm_mean': c1v2n_mean,
        'comp1_vec2_norm_std':  c1v2n_std
    }
    
    if init_norm == True:
        file_path = f"/home/lucadriu/Desktop/uni/Tesi/LM_bigram/eigen_dynam/dynam_test/evo_data/test_norm_multi_eigenvecs_evo({alpha}_{d}).pkl"
    else:
        file_path = f"/home/lucadriu/Desktop/uni/Tesi/LM_bigram/eigen_dynam/dynam_test/evo_data/test_multi_eigenvecs_evo({alpha}_{d}).pkl"
    with open(file_path, 'wb') as file:
        pickle.dump(stats_data, file)

    return n_runs

























"""


pi , ui = tokens_struc(v_size=1000, alpha=1.5, greedy=False, flip_mp=True, n_passes=10, seed=42)

c2v1, c1v2, mu_tn1, mu_tn2, mu_t, norm1 = components(pi, ui, beta=0.5, eta=0.2, t=1e-5)

v1 = np.array([mu_tn1, c2v1])
v2 = np.array([mu_tn2, c1v2])

print(mu_t, '\n', norm1, '\n', mu_t / norm1)
"""