"""
This module provides classes for numerical integration rules in 2D (triangles).
"""

from abc import ABC, abstractmethod
import numpy as np
from numpy import array
from methodsnm.intrule import IntRule

class IntRuleTesserakt(IntRule):
    """
    Abstract base class for numerical integration rules on Tesserakt.
    """
    def __init__(self):
        pass

class EdgeMidPointRule4D(IntRuleTesserakt):
    """
    Class for the midpoint rule for 4D numerical integration.
    """
    def __init__(self):
        """
        Initializes the midpoint rule with the given interval.

        """
        self.nodes = array([[0.5,0.5,0.5,0.5]])
        self.weights = array([1.0])
        self.exactness_degree = 1


import numpy as np
from numpy.polynomial.legendre import leggauss
from itertools import product
class IntRule4D(IntRuleTesserakt):
    @staticmethod
    def get_1d_rule_unit_interval(order):
        nodes, weights = leggauss(order)
        # transform from [-1,1] to [0,1]
        nodes = 0.5 * (nodes + 1)
        weights = 0.5 * weights
        return nodes, weights
    @staticmethod
    def get_4d_integration_rule(order):
        """
        Gives the weights and nodes for the integration rule in 4D on [0,1]^4

        Returns:
            nodes_4d: ndarray shape (n^4, 4)
            weights_4d: ndarray shape (n^4,)
        """
        nodes_1d, weights_1d = IntRule4D.get_1d_rule_unit_interval(order)
        nodes_4d = []
        weights_4d = []

        for x, y, z, t in product(range(order), repeat=4):
            point = [nodes_1d[x], nodes_1d[y], nodes_1d[z], nodes_1d[t]]
            weight = weights_1d[x] * weights_1d[y] * weights_1d[z] * weights_1d[t]
            nodes_4d.append(point)
            weights_4d.append(weight)

        return np.array(nodes_4d), np.array(weights_4d)
    
    def __init__(self, order):
        """
        Initializes the integration rule with the given order.

        Parameters:
        order (int): The order of the integration rule.
        """

        self.nodes, self.weights = IntRule4D.get_4d_integration_rule(order)
        
def _simplex_unit_volume(m: int) -> float:
    """
    Volume of m-dimensional Unit-Simplex
    { x_i >= 0, sum x_i <= 1 } = 1 / m!
    """
    v = 1.0
    for i in range(1, m+1):
        v /= i
    return v


def _compositions(n: int, k: int):
    """
    Generator for compositions of n into k nonnegative parts.
    Corresponds to comp_next from Burkardt.
    """
    if k == 1:
        yield (n,)
        return
    for i in range(n+1):
        for tail in _compositions(n - i, k - 1):
            yield (i,) + tail


def _gm_rule_size(rule: int, m: int) -> int:
    """
    Number of nodes N for GM-Rule 'rule' in the m-simplex:
    N = C(m + rule + 1, rule)
    """
    from math import comb
    return comb(m + rule + 1, rule)


def _gm_unit_rule_set(rule: int, m: int):
    """
    Python port of gm_unit_rule_set (Burkardt) for the
    m-dimensional unit simplex.

    Parameters
    ----------
    rule : int
        GM index s >= 0 (exactness = 2*s + 1)
    m : int
        Dimension

    Returns
    -------
    w : (N,) array
        Weights
    x : (m, N) array
        Nodes in the unit simplex
    """
    s = rule
    d = 2 * s + 1
    n = _gm_rule_size(rule, m)

    w = np.zeros(n)
    x = np.zeros((m, n))

    k = 0
    one_pm = 1.0

    for i in range(0, s + 1):
        weight = one_pm
        upper = max(m, d, d + m - i)

        for j in range(1, upper + 1):
            if j <= m:
                weight *= float(j)
            if j <= d:
                weight *= float(d + m - 2 * i)
            if j <= 2 * s:
                weight /= 2.0
            if j <= i:
                weight /= float(j)
            if j <= d + m - i:
                weight /= float(j)

        one_pm = -one_pm
        beta_sum = s - i

        for beta in _compositions(beta_sum, m + 1):
            if k >= n:
                raise RuntimeError("Too many GM points generated")
            x[:, k] = (2 * np.array(beta[1:], dtype=float) + 1.0) / float(d + m - 2 * i)
            w[k] = weight
            k += 1

    if k != n:
        raise RuntimeError(f"GM rule size mismatch: got {k}, expected {n}")

    vol = _simplex_unit_volume(m)
    w *= vol
    return w, x


class IntRuleSimplex4D(IntRule):
    """
    Abstract base class for 4D simplex integration rules.
    """
    def __init__(self):
        pass

class IntRulePentatope(IntRuleSimplex4D):
    """
    Integration rule for the 4D reference simplex with selectable order.
    Supported orders: 1–5.
    """
    _cache = {}

    def __init__(self, order: int):
        if order <= 1:
            self._init_order_1()
        elif order == 2:
            self._init_order_2()
        elif order == 3:
            self._init_order_3()
        elif order == 4:
            self._init_order_4()
        elif order == 5:
            self._init_order_5()
        else:
            raise NotImplementedError(f"Pentatope rule for order {order} not implemented.")
        self._cache[order] = (self.nodes.copy(),self.weights.copy(),self.exactness_degree)
        
    def _init_from_gm(self, s: int):
        m = 4
        w, x = _gm_unit_rule_set(s, m)
        
        self.nodes = x.T.copy()
        self.weights = w.copy()
        self.exactness_degree = 2 * s + 1

    def _init_order_1(self):
        self.nodes = np.array([[0.2, 0.2, 0.2, 0.2]])
        self.weights = np.array([1.0 / 24.0])
        self.exactness_degree = 1

    def _init_order_2(self):
        self._init_from_gm(s=1)        

    def _init_order_3(self):
        self._init_from_gm(s=2)

    def _init_order_4(self):
        self._init_from_gm(s=3)

    def _init_order_5(self):
        self._init_from_gm(s=3)
