import numpy as np

class struc: 
    def __init__(self, v_size, alpha, seed = None):
        self.v_size = v_size
        self.alpha = alpha
        self.rng = np.random.default_rng(seed) 
        
    #Frequencies of tokens     
    def token_freq(self, v_size):
        i = np.arange(1, v_size + 1)
        pi = i**(-self.alpha) / np.sum(i**(- self.alpha))
        return pi

    #Components u_i uniformly distributed
    def u_j(self, v_size):
        u_j = self.rng.choice([1, -1], size=v_size)
        return u_j
    
    #Components u_i GREEDY
    def u_j_greedy(self, v_size):
        pi = self.token_freq(v_size)
        u_jg = np.zeros(v_size)
        u_jg[0] = 1
        sum = pi[0]

        for i in range(1, len(pi)):
            if  sum > 0:
                u_jg[i] = -1
            else:
                u_jg[i] = 1

            sum = sum + u_jg[i] * pi[i]

        return u_jg, sum
    


     #Components u_i flipped with mutliple passes
    def u_j_flip_plain(self, v_size, n_passes):
    
        pi = self.token_freq(v_size)
        u_jf = self.rng.choice([1, -1], size=v_size)

        cs_list = []
        #u_jf = np.ones(v_size)
        current_sum = pi @ u_jf 
        cs_list.append(current_sum)
        
        for p in range(n_passes):
            
            shuffled_indices = self.rng.permutation(v_size)

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

        return u_jf, cs_list




    #Components u_i flipped with the first sequential pass and the other shuffling the indices
    def u_j_flip_hyb(self, v_size, n_passes):
    
        pi = self.token_freq(v_size)
        u_jf = self.rng.choice([1, -1], size=v_size)

        cs_list = []
        #u_jf = np.ones(v_size)
        current_sum = pi @ u_jf 
        cs_list.append(current_sum)
        
        for p in range(n_passes):
            
            if p == 0:
                shuffled_indices = np.arange(v_size)
            else:
                shuffled_indices = self.rng.permutation(v_size)

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

        return u_jf, cs_list

