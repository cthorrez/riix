"""
An implementation of the Glicko rating system
http://www.glicko.net/glicko/glicko.pdf
http://www.glicko.net/research/glicko.pdf
"""
import math
import time
import numpy as np
import pandas as pd
from raters import Rater
from utils import sigmoid
from preprocessing import (index_players, 
                          Virtual1v1Processor)


# constants
three_over_pi_squared = 3 / (math.pi ** 2)
# index for a non existant player
DUMMY = -1

from glicko import q2

def g(phis):
    """g function idk why it's called that or what it really does
    Args:
        RDs: ndarray: array of "Rating Deviations"
    Outputs: 
        ndarray, transformed RDs, same shape as input
    """
    return 1 / np.sqrt(1 + three_over_pi_squared * np.square(phis))
    
def E(mu1s, mu2s, gphi2s):
    return sigmoid(gphi2s * (mu1s-mu2s))


class Glicko2(Rater):
    def __init__(self, num_players, initial_mu=1500, initial_phi=350, initial_sigma=0.06, 
                 tau=0.5, eps=1e-6, mode='1v1', max_iter=1000, **kwargs):
        super().__init__(**kwargs)
        self.num_players = num_players
        self.mus = np.zeros(num_players+1) + ((initial_mu - 1500) / 173.7178)
        self.phis = np.zeros(num_players+1) + initial_phi / 173.7178
        self.sigmas = np.zeros(num_players+1) + initial_sigma
        self.has_played_mask = np.zeros(num_players+1, dtype=bool)
        self.tau = tau
        self.tau2 = tau ** 2
        self.eps = eps
        self.virtual_1v1_processor = None
        if mode != '1v1' : self.virtual_1v1_processor = Virtual1v1Processor(mode)
        self.max_iter = max_iter
        self.initial_phi = initial_phi
        

    def f(self, x, delta2, phi, phi2, v, a):
        ex = np.exp(x)
        term1_num = ex * (delta2 - phi2 - v - ex)
        term1_denom = 2*np.square(phi2 + v +ex)
        term2 = (x - a) / self.tau2
        return (term1_num / term1_denom) - term2

    def calculate_sigma_prime(self, sigma, delta2, phi, phi2, v, a):
        A = a.copy()
        B = np.empty(sigma.shape[0])
        mask1 = delta2 <= (phi2 + v)
        B[~mask1] = np.log((delta2 - phi2 - v)[~mask1])
        k = np.ones(sigma.shape[0])
        if mask1.sum() > 0:
            mask2 = self.f(a - k*self.tau, delta2, phi, phi2, v, a) < 0
            iters = 0
            while mask2[mask1].sum() > 0:
                k[mask2] += 1
                mask2 = self.f(a - k*self.tau, delta2, phi, phi2, v, a) < 0
                iters += 1
                if iters >= self.max_iter:
                    print('first iteration loop hit max number of iterations without converging')
                    break

        B[mask1] = A[mask1] - k[mask1]*self.tau

        fA = self.f(A, delta2, phi, phi2, v, a)
        fB = self.f(B, delta2, phi, phi2, v, a)
        A_minus_B = A - B
        iters = 0
        while np.any(np.abs(A_minus_B) > self.eps):
            C = A + (A_minus_B * fA)/(fB - fA)
            fC = self.f(C, delta2, phi, phi2, v, a)
            fA = fA / 2
            swap_mask = fC * fB < 0
            fA[swap_mask] = fB[swap_mask]
            A[swap_mask] = B[swap_mask]
            B = C
            fB = fC
            A_minus_B = A - B
            iters += 1
            if iters >= self.max_iter:
                print('second iteration loop hit max number of iterations without converging')
                break
        sigma_prime = np.exp(A/2)
        return sigma_prime


    def fit_predict_batch(self, batch):
        idxs = batch[self.team1_cols + self.team2_cols].values
        team1_idxs, team2_idxs = np.hsplit(idxs, 2)
        players_per_team = team1_idxs.shape[1]
        scores = batch[self.score_col].values
        
        if self.virtual_1v1_processor:
            idxs, scores, cache = self.virtual_1v1_processor.processes_inputs(idxs, scores)
    
        # print('mean phi:', self.phis.mean())
        active_in_period = np.unique(idxs)
        self.has_played_mask[active_in_period] = True
        inactive = np.setdiff1d(np.arange(self.num_players), active_in_period)
        inactive_mask = np.zeros(self.num_players+1, dtype=bool)
        inactive_mask[inactive] = True
        increase_phi_mask = inactive_mask & self.has_played_mask
        self.phis[increase_phi_mask] = np.sqrt(np.square(self.phis[increase_phi_mask]) + np.square(self.sigmas[increase_phi_mask]))
        
        mus = self.mus[idxs]
        phis = self.phis[idxs]
        sigmas = self.sigmas[idxs]

        if self.virtual_1v1_processor:
            mus, phis = self.virtual_1v1_processor.process_params((mus, phis), cache)
    

        gphis = g(phis)
        active_in_period = active_in_period[active_in_period!=DUMMY]
        masks = np.equal(idxs[:,:,None], active_in_period[None,:])

        E1s = E(mus[:,0], mus[:,1], gphis[:,1])
        E2s = E(mus[:,1], mus[:,0], gphis[:,0])


        # for predictions, use variances of both players
        pred_Es = E(mus[:,0], mus[:,1], g(phis[:,0] + phis[:,1]))

        if self.virtual_1v1_processor:
            pred_Es = self.virtual_1v1_processor.process_preds(pred_Es, cache)

        # tmp = (E1s * (1 - E1s))[:,None] * np.square(gphis)[:,[1,0]]
        tmp = np.stack([E1s * (1 - E1s), E2s * (1-E2s)]).T * np.square(gphis)[:,[1,0]]
        vs = 1 / (tmp[:,:,None] * masks).sum(axis=(0,1))
        
        scores = np.hstack([scores[:,None], 1-scores[:,None]])
        Es = np.hstack([E1s[:,None], E2s[:,None]])

        diffs = ((gphis[:,[1,0]] * (scores - Es))[:,:,None] * masks).sum(axis=(0,1))
        deltas = vs * diffs
        delta2 = np.square(deltas)

        sigma = self.sigmas[active_in_period]
        phi = self.phis[active_in_period]
        phi2 = np.square(phi)
        a = np.log(np.square(sigma))

        sigma_prime = self.calculate_sigma_prime(sigma, delta2, phi, phi2, vs, a)

        phi_star = np.sqrt(phi2 + np.square(sigma_prime))
        phi_prime = 1 / np.sqrt((1 / np.square(phi_star)) + (1/vs))
        phi_prime2 = np.square(phi_prime)

        mu_prime = self.mus[active_in_period] + phi_prime2 * diffs

        self.mus[active_in_period] = mu_prime
        self.phis[active_in_period] = phi_prime
        self.sigmas[active_in_period] = sigma_prime

        return pred_Es

    def rank(self, n=5):
        print(self.sigmas.mean(), self.sigmas.min(), self.sigmas.max())
        idxs = np.argsort(self.mus - 2.5 * self.phis)[::-1]
        print(f'{"player": <28}{"mu": <12}{"phi": <12}{"sigma": <12}')
        up_mus = (self.mus * 173.7178) + 1500
        up_phis = self.phis * 173.7178

        for i in range(n):
            out = f'{self.idx_to_player[idxs[i]]: <28}{round(up_mus[idxs[i]],4): <12}'
            out += f'{round(up_phis[idxs[i]],4): <12}{round(self.sigmas[idxs[i]],4)}'
            print(out)

if __name__ == '__main__':
    model = Glicko2(num_players=4, tau=0.5, initial_sigma=0.06)
    model.mus[:-1] = (np.array([1500,1400,1550,1700]) - 1500) / 173.7178
    model.phis[:-1] = np.array([200, 30, 100, 300]) / 173.7178

    print('mus:', model.mus[:-1])
    print('phis:', model.phis[:-1])

    data = [
        {'player1' : '0', 'player2' : '1', 'score' : 1},
        {'player1' : '0', 'player2' : '2', 'score' : 0},
        {'player1' : '0', 'player2' : '3', 'score' : 0}
    ]
    df = pd.DataFrame(data)
    preds = model.fit_predict(df, ['player1'], ['player2'], 'score', 'minibatch', batch_size=3)
    print((model.mus[0] * 173.7178) + 1500)
    print(model.phis[0] * 173.7178)
    print(model.sigmas[0])



        

    
            