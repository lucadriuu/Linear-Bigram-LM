import numpy as np

def token_freq(d,alpha):
        i = np.arange(1,d + 1)
        pi = i**(-alpha) / np.sum(i**(- alpha))
        return pi

def u_j_greedy(d, pi, beta):
        u_jg = np.zeros(d)
        u_jg[0] = 1
        orto_sum = pi[0]

        for i in range(1, len(pi) ):
            if  orto_sum > 0:
                u_jg[i] = -1
            else:
                u_jg[i] = 1
            orto_sum = orto_sum + u_jg[i] * pi[i]

        p_plus = pi*(1 - beta * u_jg)
        p_min = pi*(1 + beta * u_jg)

        p_plus /= np.sum(p_plus)                #normalize in order to have summation to 1 when we call np.arandom.choice()
        p_min  /= np.sum(p_min)

        return u_jg, p_plus, p_min
    

def sequence(seq_len, d, pi, u, p_min, p_plus):
    seq = []
    vocab_indices = np.arange(d)
    current_token = np.random.choice(vocab_indices, p=pi)
    seq.append(current_token)
    
    for t in range(1, seq_len):
        if u[current_token] == 1:
            current_token = np.random.choice(vocab_indices, p=p_plus)
        else:
            current_token = np.random.choice(vocab_indices, p=p_min)
            
        seq.append(current_token)
    
    return seq


def emp_matrix(seq, d):
    counts = np.zeros((d, d), dtype=float)
    for t in range(len(seq)-1):
        j = seq[t]                      #current token
        i = seq[t + 1]                  #next token
        counts[j,i] += + 1.0

    row_sums = counts.sum(axis=1, keepdims=True)
    jump_probs = np.where(row_sums > 0, counts / row_sums, 0.0)

    #check = jump_probs @ pi


    return jump_probs



