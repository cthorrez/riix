from math import log, exp, pow
import jax.numpy as jnp
import jax
from jax import jacfwd, jacrev, grad
from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')


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

def elo_update(mu_w, mu_l, k):
    p = 1 / (1 + pow(10, (mu_l - mu_w)/400))
    print(p)

    update = k * (1 - p)
    return update

def weng_lin_bt_update(mu_w, sigma2_w, mu_l, sigma2_l, beta2=0, alpha=1.0):
    c = jnp.sqrt(sigma2_w + sigma2_l)
    gamma = jnp.sqrt(sigma2_l) / c
    gamma = 1
    p = sigmoid(alpha*(mu_w - mu_l)/c)
    print(p)
    delta = (sigma2_w / c) * (1-p)

    g = (mu_w-mu_l)*(1-p)/(2*c**3)
    print('gradient wrt s2 in bt', g)

    d = (mu_l - mu_w)
    d2 = d**2
    h = -d2*(1-p)/(4*c**6) + d2*(1-p)**2/(4*c**6) - 3*d*(1-p)/(4*c**5)
    h2 = d*(1-p)*(-d*p-3*c)/(4*c**6)
    print('hess wrt sw in bt', h, h2)

    print('hess s2 update', g/h2)

    eta = gamma * (sigma2_w / c**2) * (p * (1-p))
    mu_update = delta
    sigma2_update = -sigma2_w * eta
    return jnp.array([mu_update, sigma2_update])

def constant_variance_glicko_update(mu_w, mu_l, sigma2, b=1.0):
    # b = alpha = log10/400 
    k_num = b / 2
    sigma2_d =  2*sigma2
    mu_d = mu_w - mu_l
    p = sigmoid(b*mu_d)
    k_denom = (1/sigma2_d) + (b**2)*p*(1-p)
    k = k_num / k_denom
    update = k*(1-p)
    return update


def ingram_loss(delta, mu_w, mu_l, sigma2, b=1.0):
    term1 = 0.5*jnp.power(delta-(mu_w-mu_l),2)/(2*sigma2)
    p = sigmoid(b*(delta))
    term2 = -jnp.log(p)
    return term1 + term2


def log_loss_mu(mu_w, mu_l, alpha=1.):
    p = sigmoid(alpha*(mu_w-mu_l))
    jax_print(p)
    loss = -jnp.log(p)
    return loss / alpha

def log_loss_mu_sigma(mu_w, sigma2_w, mu_l, sigma2_l, alpha=1.):
    c = jnp.sqrt(sigma2_w + sigma2_l)
    p = sigmoid(alpha*(mu_w-mu_l)/(c))
    jax_print(p)
    loss = -jnp.log(p)
    return loss / alpha


def log_loss_mu_grad_update(mu_w, mu_l, alpha=1.):
    grad_fn = grad(log_loss_mu)
    gradient = grad_fn(mu_w, mu_l, alpha)
    update = -gradient
    return update

def log_loss_mu_hess_update(mu_w, mu_l, alpha=1.0):
    grad_fn = grad(log_loss_mu)
    hess_fn = grad(grad_fn)
    gradient = grad_fn(mu_w, mu_l, alpha=alpha)
    hessian = hess_fn(mu_w, mu_l, alpha=alpha)
    print('grad mu hess', gradient)
    print('hessian mu hess', hessian)
    update = -gradient / (hessian / alpha)
    return update

def log_loss_mu_sigma_grad_update(w, l, alpha=1.0):
    grad_fn = grad(log_loss_mu_sigma, argnums=(0,1))
    gradient = grad_fn(w[0], w[1], l[0], l[1], alpha=alpha)
    gradient = jnp.array(gradient)
    print('gradient', gradient[1])
    update = -gradient
    return update

def log_loss_mu_sigma_hess_update(w, l, alpha=1.0):
    grad_fn = grad(log_loss_mu_sigma, argnums=(0,1))
    hess_fn = jacrev(grad_fn, argnums=(0,1))
    gradient = grad_fn(w[0], w[1], l[0], l[1], alpha=alpha)
    gradient = jnp.array(gradient)
    print('gradient', gradient)
    hessian = hess_fn(w[0], w[1], l[0], l[1], alpha=alpha)
    hessian = jnp.array(hessian)
    print('hessian', hessian)
    print('diag update', gradient/jnp.diag(hessian) / alpha)
    update = -jnp.dot(gradient, jnp.linalg.inv(hessian) / alpha)
    return update

def ingram_loss_update(mu_w, mu_l, sigma2, b=1.0):
    grad_fn = grad(ingram_loss)
    hess_fn = grad(grad_fn)
    delta = mu_w-mu_l
    gradient = grad_fn(delta, mu_w, mu_l, sigma2, b)
    hessian = hess_fn(delta, mu_w, mu_l, sigma2, b)
    print('ingram grad', gradient)
    print('ingram hess', hessian)
    update = -gradient / (hessian / b)

    return update / 2# update to mu_w is half of update to mu_delta


if __name__ == '__main__':
    mu_w = 1600.
    mu_l = 1400.
    sigma2_w = 50.
    sigma2_l = 60.
    # sigma2_w = sigma2_l = 0.5

    mu_w = 1.0
    mu_l = 0.0
    sigma2_w = 0.4
    sigma2_l = 0.4

    elo_alpha = log(10) / 400

    a = elo_update(mu_w, mu_l, k=1)
    print('elo update', a)

    b = log_loss_mu_grad_update(mu_w, mu_l, elo_alpha)
    print('log loss grad update', b)


    elo_alpha = 1.

    c = log_loss_mu_hess_update(mu_w, mu_l, elo_alpha)
    print('log loss mu hess update', c)


    d = log_loss_mu_sigma_grad_update(jnp.array([mu_w, sigma2_w]), jnp.array([mu_l, sigma2_l]), elo_alpha)
    print('log loss mu sigma grad update', d)

    e = log_loss_mu_sigma_hess_update(jnp.array([mu_w, sigma2_w]), jnp.array([mu_l, sigma2_l]), elo_alpha)
    print('log loss mu sigma hess update', e)

    c = jnp.sqrt(sigma2_w + sigma2_l)
    gamma = jnp.sqrt(sigma2_l) / c
    gamma = 1
    f = weng_lin_bt_update(mu_w, sigma2_w, mu_l, sigma2_l, alpha=elo_alpha)
    print('weng lin bt update', f)

    g = constant_variance_glicko_update(mu_w, mu_l, sigma2_w, b=elo_alpha)
    print('contstant variance glicko update:', g)

    h = ingram_loss_update(mu_w, mu_l, sigma2_w, b=elo_alpha)
    print('ingram loss update:', h)









