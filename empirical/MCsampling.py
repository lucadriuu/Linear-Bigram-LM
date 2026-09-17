import numpy as np

def token_freq(d,alpha):
        i = np.arange(1,d + 1, dtype=float)     #float: integer ** negative integer alpha raises in numpy
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
    

def sequence(seq_len, d, pi, u, p_min, p_plus, rng):
    a = rng.choice(d, size=seq_len, p=p_plus).tolist()   # candidates if current u = +1
    b = rng.choice(d, size=seq_len, p=p_min).tolist()    # candidates if current u = -1
    up = (u > 0).tolist()                                 # group lookup table

    x = [0] * seq_len
    cur = int(rng.choice(d, p=pi))                        # stationary start
    x[0] = cur
    for t in range(1, seq_len):
        cur = a[t] if up[cur] else b[t]
        x[t] = cur

    return np.array(x, dtype=np.int32)



def bigram_stats(seq, d):
    cur = np.asarray(seq[:-1], dtype=np.int64)                 #input tokens (seq_len - 1 pairs)
    nxt = np.asarray(seq[1:], dtype=np.int64)                  #target tokens; int64 so cur*d + nxt does not overflow

    counts = np.bincount(cur, minlength=d)                                 #c_i
    keys, pair_counts = np.unique(cur * d + nxt, return_counts=True)       #c_ij for the observed pairs only
    sq_counts = np.bincount(keys // d, weights=pair_counts.astype(float)**2, minlength=d)     #sum_j c_ij^2

    marg = counts / len(cur)                                   #\pi^_i
    row_sq_sum = np.where(counts > 0, sq_counts / np.maximum(counts, 1)**2, 0.0)          #sum_j \pi^_{j|i}^2

    return marg, row_sq_sum



