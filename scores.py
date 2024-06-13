import copy
import ges.utils as utils
import numpy as np
import scipy

class DecomposableScore():

    def __init__(self, data, cache=True, debug=0):
        self._data = copy.deepcopy(data)
        self._cache = {} if cache else None
        self._debug = debug
        self.p = None

    def local_score(self, x, pa):
        """
        Return the local score of a given node and a set of
        parents. If self.cache=True, will use previously computed
        score if possible.

        Parameters
        ----------
        x : int
            a node
        pa : set of ints
            the node's parents

        Returns
        -------
        score : float
            the corresponding score

        """
        if self._cache is None:
            return self._compute_local_score(x, pa)
        else:
            key = (x, tuple(sorted(pa)))
            try:
                score = self._cache[key]
                print("score%s: using cached value %0.2f" %
                      (key, score)) if self._debug >= 2 else None
            except KeyError:
                score = self._compute_local_score(x, pa)
                self._cache[key] = score
                print("score%s = %0.2f" % (key, score)) if self._debug >= 2 else None
            return score

    def _compute_local_score(self, x, pa):
        """
        Compute the local score of a given node and a set of
        parents.

        Parameters
        ----------
        x : int
            a node
        pa : set of ints
            the node's parents

        Returns
        -------
        score : float
            the corresponding score

        """
        return 0

class HuberScore(DecomposableScore):
    """
    Implements a cached l0-penalized general score.

    """

    def __init__(self, data, lambd=0.5, delta=np.inf, cache=False, debug=0):
        super().__init__(data, cache=cache, debug=debug)

        self.n, self.p = data.shape
        self.data = data
        self.lambd = lambd
        self.delta = delta

    def _huber_lin_reg(self, X, y, delta=np.inf):
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

        res = scipy.optimize.minimize(fun=f, x0=np.zeros(d), jac=g)
        return res.fun

    def _compute_local_score(self, x, pa):
        """
        Compute the local score of a given node and a set of
        parents.

        Parameters
        ----------
        x : int
            a node
        pa : set of ints
            the node's parents

        Returns
        -------
        score : float
            the corresponding score

        """
        l0_term = np.log(self.n) / self.n * (len(pa) + 1)

        y = self.data[:, x].copy()

        mean_rss = self._huber_lin_reg(self.data[:, list(pa)].copy(), y, delta=self.delta) if len(pa) > 0 else \
                np.mean(np.minimum(y**2, self.delta*np.abs(y)))

        return -mean_rss - self.lambd * l0_term

    def _compute_full_score(self, G):
        full_score = 0

        for i in range(self.p):
            full_score += self._compute_local_score(i, utils.pa(i, G))

        return full_score