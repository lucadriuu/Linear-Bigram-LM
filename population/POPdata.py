import numpy as np


def token_freq(d, alpha):
        i = np.arange(1, d + 1)
        pi = i**(-alpha) / np.sum(i**(- alpha))
        return pi

def u_jn(v_size, rng):
        u_j = rng.choice([1, -1], size=v_size)
        return u_j

def u_j_greedy(d, pi):
        u_jg = np.zeros(d)
        u_jg[0] = 1
        orto_sum = pi[0]

        for i in range(1, len(pi) ):
            if  orto_sum > 0:
                u_jg[i] = -1
            else:
                u_jg[i] = 1
            orto_sum = orto_sum + u_jg[i] * pi[i]

        return u_jg

def u_j_flip_mp(d, pi, n_passes, rng):
        u_jf = rng.choice([1, -1], size=d)

        cs_list = []
        current_sum = pi @ u_jf
        cs_list.append(current_sum)

        for p in range(n_passes):

            if p == 0:
                shuffled_indices = np.arange(d)
            else:
                shuffled_indices = rng.permutation(d)

            for k in shuffled_indices:

                old_val = u_jf[k]

                u_jf[k] = - old_val
                new_sum = current_sum + pi[k] * (u_jf[k] - old_val)

                if abs(new_sum) < abs(current_sum):
                    current_sum = new_sum
                else:
                    u_jf[k] = old_val
            current_sum = pi @ u_jf
            cs_list.append(current_sum)

        return u_jf


def build_u(method, d, pi, n_passes, rng):
        if method == 'greedy':
            return u_j_greedy(d, pi)
        elif method == 'random':
            return u_jn(d, rng)
        elif method == 'flip':
            return u_j_flip_mp(d, pi, n_passes, rng)
        else:
            raise ValueError('unknown method: ' + method)
