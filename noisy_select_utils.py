from ges.utils import pa, reachable

import numpy as np
from numpy.random import laplace

from statsmodels.regression.linear_model import OLS
from statsmodels.stats.weightstats import _zconfint_generic

from scipy.optimize import minimize

def huber_lin_reg(X, y, delta=np.inf):
    """Minimizes the Huber loss.

    Args:
        X (ndarray): Covariates.
        y (ndarray): Labels.
        delta (float): The delta parameter in the Huber score.

    Returns:
        The resulting loss (resulting value of objective function).
    """
    
    n, d = X.shape
    
    def f(theta):
        y_curr = X @ theta
        resid = y - y_curr
        
        clip_idx = np.where(np.abs(resid) > delta)[0]
        
        losses = 1/2 * resid**2
        losses[clip_idx] = delta * (np.abs(resid[clip_idx]) - delta / 2)
        
        return np.mean(losses)
        
    def g(theta):
        y_curr = X @ theta
        resid = y - y_curr
        
        clip_idx = np.where(np.abs(resid)  > delta)[0]
        resid[clip_idx] = delta * np.sign(resid[clip_idx])
        
        g_theta = - (1/n) * X.T @ resid
        
        return g_theta
    
    res = minimize(fun=f, x0=np.zeros(d), jac=g)
    return res.fun

def local_score(data, i, G, lambd=0.5, delta=np.inf):
    """Computes the local score.

    Args:
        data (ndarray): Dataset.
        i (int): Node.
        G (ndarray): Graph.
        lambd (float): The regularization strength.
        delta (float): The delta parameter in the Huber score. 
                       Defaults to infinity (reverting to BIC score).

    Returns:
        The resulting local score.
    """
    n, d = data.shape
    parents = pa(i, G)
    y = data[:, i].copy() 
    
    l0_term = np.log(n) / n * (len(parents) + 1)
    huber_loss = huber_lin_reg(data[:, list(parents)].copy(), y, delta=delta) if len(parents) > 0 \
                                else np.mean(np.minimum(y**2, delta*np.abs(y)))

    return huber_loss + lambd * l0_term

def score(data, G, lambd=0.5, delta=np.inf):
    """Computes the local score.

    Args:
        data (ndarray): Dataset.
        G (ndarray): Graph.
        lambd (float): The regularization strength.
        delta (float): The delta parameter in the Huber score. 
                       Defaults to infinity (reverting to BIC score).

    Returns:
        The resulting full Huber score.
    """
    
    _, d = data.shape
    full_score = 0
    
    for i in range(d):
        full_score += local_score(data, i, G, lambd=lambd, delta=delta)
        
    return full_score

def _ols(X, Y, return_se=False):
    """Computes the ordinary least squares coefficients.

    Args:
        X (ndarray): Covariates.
        Y (ndarray): Labels.
        return_se (bool, optional): Whether to return the standard errors of the coefficients.

    Returns:
        theta (ndarray): Ordinary least squares estimate of the coefficients.
        se (ndarray): If return_se==True, return the standard errors of the coefficients.
    """
    regression = OLS(Y, exog=X).fit()
    theta = regression.params
    if return_se:
        return theta, regression.HC0_se
    else:
        return theta
    
def classical_ols_ci(X, Y, w=None, alpha=0.1, alternative="two-sided"):
    """Confidence interval for the OLS coefficients using the classical method.

    Args:
        X (ndarray): Labeled features.
        Y (ndarray): Labeled responses.
        w (ndarray, optional): Sample weights for the data set. Must be positive
                               and will be normalized to sum to the size of the dataset.
        alpha (float, optional): Error level. Confidence interval will target a coverage
                                 of 1 - alpha. Defaults to 0.1. Must be in (0, 1).
        alternative (str, optional): One of "two-sided", "less", or "greater". Defaults
                                     to "two-sided".

    Returns:
        tuple: (lower, upper) confidence interval bounds.
    """
    n = Y.shape[0]
    if w is None:
        pointest, se = _ols(X, Y, return_se=True)
    else:
        w = w / w.sum() * Y.shape[0]
        pointest, se = _wls(X, Y, w, return_se=True)
    return _zconfint_generic(pointest, se, alpha, alternative)

def get_CI_width(data, G, err_lvl):
    """Computes the confidence interval width from inference at level
    err_lvl on data conducted on a randomly selected edge in G.

    Args:
        data (ndarray): Dataset.
        G (ndarray): Graph.
        err_lvl (float): Error level for inference.

    Returns:
        A float representing the width associated to the CI interval
        for a randomly selected edge in G.
    """
    
    n = data.shape[0]
    edges = np.argwhere(np.transpose(G) > 0)
    
    edge = edges[np.random.randint(len(edges))]
    backdoor = [k for k in reachable(edge[1], G) if k in reachable(edge[0], G)] # check if needs backdoor adj
    
    if len(backdoor) != 0: # regress on A and its backdoor (adjustment set)
        A = np.concatenate((data[:, backdoor], data[:, edge[0]].reshape((n,1))), axis=1)
    else: # if backdoor is empty, regress b on all its parents and read off A's contribution
        A = np.concatenate((data[:, list(pa(edge[1], G))], data[:, edge[0]].reshape((n,1))), axis=1)
        
    b = data[:, edge[1]]    
    
    ci = (classical_ols_ci(A, b, alpha=err_lvl)[0][-1], classical_ols_ci(A, b, alpha=err_lvl)[1][-1])
    return ci[1] - ci[0]

def get_sensitivity(data, G_options, lambd=0.5, delta=np.inf, iterations=10): # 4. Estimate sensitivity
    """Estimates the score sensitivity for the given graph options and data.

    Args:
        data (ndarray): Dataset.
        G_options (list): A list of graph variants.
        lambd (float): The regularization strength.
        delta (float): The delta parameter in the Huber score. 
                       Defaults to infinity (reverting to BIC score).
        iterations (int): The number of iterations for which we generate
                          a copy of the original data with one random entry
                          replaced.

    Returns:
        A sensitivity estimate.
    """
    
    max_change = 0
    
    for _ in range(iterations):
        idx_1 = np.random.randint(data.shape[0])
        idx_2 = np.random.randint(data.shape[0])
        
        data_prime = data.copy()
        data_prime[idx_1] = data[idx_2]
        
        for G in G_options:
            score_og = score(data, G, lambd=lambd, delta=delta)
            score_prime = score(data_prime, G, lambd=lambd, delta=delta)
            
            max_change = max(max_change, np.abs(score_og - score_prime))
            
    return max_change

noise_scale = lambda eps, tau: 2 * tau / eps # Get noise scale given sensitivity (tau) and privacy (eps) parameters.
max_info = lambda eps, gamma, n: n * eps**2/2 + eps * np.sqrt(n*np.log(2/gamma)/2) # compute maximum information.

def alpha_tilde(err_lvl, eps, n, start=1e-5, disc=1e4):
    """Computed the adjusted error level given a privacy level.

    Args:
        err_lvl (float): The original/desired error level.
        eps (float): The privacy parameter.
        n (integer): The number of datapoints.

    Returns:
        The adjusted error level for noisy-select.
    """
    gammas = np.linspace(start, err_lvl, int(disc))
    alpha_tilde = [(err_lvl-gamma)*np.exp(-max_info(eps, gamma, n)) for gamma in gammas]
    
    return np.max(alpha_tilde)

def subsample(data, frac):
    """Subsample a fraction of the data.

    Args:
        data (ndarray): the full dataset.
        frac (float): the fraction of the data to be subsampled.
        
    Returns:
        A tuple of two disjoint random subsamples of (frac%, (1-frac)%) of the full dataset.
    """
    
    n, _ = data.shape
    
    all_idx = list(range(n))
    select_idx = np.random.choice(n, size=int(n*frac), replace=False)
    infer_idx = []
    
    for idx in all_idx:
        if idx not in select_idx:
            infer_idx.append(idx)
    
    return data[select_idx], data[infer_idx]

def generate_graphs(G, no_graphs, p_remove, p_add):
    """ Generate random variants of the original graph.

    Args:
        G (ndarray): The original graph.
        no_graphs (integer): Number of graphs in list.
        p_remove (float): Probability of removing an edge.
        p_add (float): Probability of adding an edge.
        
    Returns:
        A list of graph variants.
    """
        
    d, _ = G.shape
    G_options = [G]
    
    while len(G_options) < no_graphs:
        
        # remove edges
        G_perturbed = (G + np.random.binomial(n=1, p=1-p_remove, size=[d*d]).reshape(d,d) >= 2).astype(int)
        
        # add edges
        G_perturbed = (G_perturbed + np.random.binomial(n=1, p=p_add, size=[d*d]).reshape(d,d) >= 1).astype(int)
        
        if np.any(G_perturbed - G): # add to list of graphs if different from G
            G_options.append(G_perturbed)
            
    return G_options