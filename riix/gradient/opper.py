import math
import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax
from jax import jacfwd, jacrev, grad, value_and_grad
from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
from tqdm import tqdm

def jax_print(val):
    if hasattr(val, 'primal'):
        if hasattr(val.primal, 'primal'):
            jax_print(val.primal)
        else:
            print(val.primal)
    else:
        print('hmm')

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def logistic_prob(mus, sigma2s, sigma2_0, label):
    c = jnp.sqrt(2*sigma2_0 + sigma2s[0] + sigma2s[1])
    p = sigmoid((mus[0]-mus[1])/(c))
    p = (p * label) + ((1. - p) * (1 - label)) 
    return p

def logistic_log_likelihood(mus, sigma2s, sigma2_0, label):
    p = logistic_prob(mus, sigma2s, sigma2_0, label)
    loss = jnp.log(p)
    return loss

# gradient of log loss with respect to mu_1 and mu_2
logistic_log_likelihood_and_grad_fn = value_and_grad(logistic_log_likelihood, argnums=0)

logistic_hess_fn = jacfwd(jacrev(logistic_log_likelihood, argnums=0))



# def log_loss_mu_grad_update(mu_w, mu_l, alpha=1.):
#     grad_fn = grad(log_loss_mu)
#     gradient = grad_fn(mu_w, mu_l, alpha)
#     update = -gradient
#     return update

# def log_loss_mu_hess_update(mu_w, mu_l, alpha=1.0):
#     grad_fn = grad(log_loss_mu)
#     hess_fn = grad(grad_fn)
#     gradient = grad_fn(mu_w, mu_l, alpha=alpha)
#     hessian = hess_fn(mu_w, mu_l, alpha=alpha)
#     print('grad mu hess', gradient)
#     print('hessian mu hess', hessian)
#     update = -gradient / (hessian / alpha)
#     return update

# def log_loss_mu_sigma_grad_update(w, l, alpha=1.0):
#     grad_fn = grad(log_loss_mu_sigma, argnums=(0,1))
#     gradient = grad_fn(w[0], w[1], l[0], l[1], alpha=alpha)
#     gradient = jnp.array(gradient)
#     print('gradient', gradient[1])
#     update = -gradient
#     return update

# def log_loss_mu_sigma_hess_update(w, l, alpha=1.0):
#     grad_fn = grad(log_loss_mu_sigma, argnums=(0,1))
#     hess_fn = jacrev(grad_fn, argnums=(0,1))
#     gradient = grad_fn(w[0], w[1], l[0], l[1], alpha=alpha)
#     gradient = jnp.array(gradient)
#     print('gradient', gradient)
#     hessian = hess_fn(w[0], w[1], l[0], l[1], alpha=alpha)
#     hessian = jnp.array(hessian)
#     print('hessian', hessian)
#     print('diag update', gradient/jnp.diag(hessian) / alpha)
#     update = -jnp.dot(gradient, jnp.linalg.inv(hessian) / alpha)
#     return update

class Opperate:
    def __init__(
        self,
        num_players: int,
        theta_0: float = 0.0,
        sigma2_0: float = 1.0,
    ):
        self.num_players = num_players
        self.sigma2_0 = sigma2_0
        self.theta = np.zeros(num_players, dtype=np.float64) + theta_0
        self.C = np.zeros((num_players, ), dtype=np.float64) + sigma2_0
        self.active_mask = np.zeros(num_players, dtype=np.bool_)
        self.pid_to_idx = {}
        self.probs = []


    def update_players(self, p1_id, p2_id, result):
        # update variance of the players for passage of time

        if p1_id not in self.pid_to_idx:
            p1_idx = len(self.pid_to_idx)
            self.pid_to_idx[p1_id] = p1_idx
        else:
            p1_idx = self.pid_to_idx[p1_id]

        if p2_id not in self.pid_to_idx:
            p2_idx = len(self.pid_to_idx)
            self.pid_to_idx[p2_id] = p2_idx
        else:
            p2_idx = self.pid_to_idx[p2_id]


        idxs = [p1_idx, p2_idx]
        thetas = self.theta[idxs]
        Cs = self.C[idxs]

        log_likelihood, grad = logistic_log_likelihood_and_grad_fn(
            thetas,
            Cs,
            self.sigma2_0,
            result
        )

        hess = logistic_hess_fn(
            thetas,
            Cs,
            self.sigma2_0,
            result
        )

        prob = jnp.exp(log_likelihood)
        if result == 0:
            prob = 1. - prob
        

        thetas = thetas + Cs * grad
        Cs = Cs + jnp.square(Cs) * jnp.diag(hess)

        self.theta[idxs] = thetas
        self.C[idxs] = Cs


        return prob

    def play_game(self, p1_id, p2_id, result):

        prob = self.update_players(p1_id, p2_id, result)
        self.probs.append(prob)


    def run_schedule(self, games):
        for (p1_id, p2_id, result) in tqdm(games):
            self.play_game(p1_id, p2_id, result)
        return np.array(self.probs)
    


    def topk(self, k):
        sorted_players = sorted(
            [(id, self.theta[idx], self.C[idx]) for id, idx in self.pid_to_idx.items()],
            key=lambda x: x[1] - 3*x[2],
            reverse=True
        )
        for idx in range(1, k+1):
            row = sorted_players[idx-1]
            print(f'{idx:<3}{row[0]:<20}{row[1]:.6f}    {math.sqrt(row[2]):.4f}')


if __name__ == '__main__':
    from riix.datasets import get_sc2_dataset, get_melee_dataset

    df, date_col, score_col, team1_cols, team2_cols = get_sc2_dataset(path='../data/sc2_matches_5-27-2023.csv')
    # df, date_col, score_col, team1_cols, team2_cols = get_melee_dataset(tier=1)

    n = 500000
    df = df.head(n)[[*team1_cols, *team2_cols, score_col]]
    num_players = len(pd.unique(df[[*team1_cols, *team2_cols]].values.ravel('K')))
    print(f'{num_players} unique players')
    games = list(df.itertuples(index=False, name=None))

    # beta = 0.99999
    # epsilon = 1 - beta ** 2

    beta = 1.0
    epsilon = 1e-5

    model = Opperate(
        num_players=num_players,
        theta_0=0.,
        sigma2_0=1.,
    )
    probs = model.run_schedule(games)
    acc = ((probs >= 0.5) == df[score_col]).mean()

    model.topk(25)
    print(f'accuracy: {acc}')
    print(f'score mean: {df[score_col].mean()}')
        