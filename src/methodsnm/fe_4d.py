from abc import ABC, abstractmethod
import numpy as np
from numpy import array
from methodsnm.fe import *

class FE_4D(FE):
    """
    Abstract base class for finite elements in 4D.
    It implements a derivative evaluation using numerical differentiation.
    """    

    num_diff_warned = False

    def __init__(self):
        self.dim = 4

    def _evaluate_id(self, ip):
        raise Exception("Not implemented - Base class should not be used")

    def _evaluate_deriv(self, ip):
        # numerical differentiation - should be overwritten by subclasses
        # for proper accuracy and performance
        if not FE_4D.num_diff_warned:
            print("Warning: Using numerical differentiation for deriv evaluation in " + str(type(self)) + " object.")
            FE_4D.num_diff_warned = True
        eps = 1e-8
        left = ip.copy() - array([eps,0,0,0])
        right = ip.copy() + array([eps,0,0,0])
        left2= ip.copy() - array([0,eps,0,0])
        right2 = ip.copy() + array([0,eps,0,0])
        left3= ip.copy() - array([0,0,eps,0])
        right3 = ip.copy() + array([0,0,eps,0])
        left4= ip.copy() - array([0,0,0,eps])
        right4 = ip.copy() + array([0,0,0,eps])
        return array([(self._evaluate_id(right) - self._evaluate_id(left))/(2*eps), 
                      (self._evaluate_id(right2) - self._evaluate_id(left2))/(2*eps),
                      (self._evaluate_id(right3) - self._evaluate_id(left3))/(2*eps),
                      (self._evaluate_id(right4) - self._evaluate_id(left4))/(2*eps)])


class TesseraktFE(FE_4D):
    """
    Abstract base class for finite elements on tesserakt.
    """    

    def __init__(self):
        super().__init__()
        self.eltype = "tesserakt"

    @abstractmethod
    def _evaluate_id(self, ip):
        raise Exception("Not implemented - Base class should not be used")
class P1_Tesserakt_FE(TesseraktFE,Lagrange_FE):
    def _evaluate_deriv(self, ip):
        return super()._evaluate_deriv(ip)
    """
    This class represents a P1 Tesserakt finite element.
    """
    ndof = 2**4
    order = 1
    def __init__(self):
        super().__init__()
        self.nodes = []
        for l in [0,1]:
            for k in [0,1]:
                for j in [0,1]:
                    for i in [0,1]:
                        self.nodes.append(np.array([i,j,k,l]))

    def _evaluate_id(self, ip):
        """
        Evaluates the P1 Tesserakt finite element at the given integration point.

        Parameters:
        ip (numpy.ndarray): The integration point at which to evaluate the finite element.

        Returns:
        numpy.ndarray: The values of the P1 Tesserakt finite element at the given integration point.
        """

        x, y, z, w = ip
        return np.array([
            (1-x)*(1-y)*(1-z)*(1-w),
                x*(1-y)*(1-z)*(1-w),
            (1-x)*    y*(1-z)*(1-w),
                x*    y*(1-z)*(1-w),
            (1-x)*(1-y)*    z*(1-w),
                x*(1-y)*    z*(1-w),
            (1-x)*    y*    z*(1-w),
                x*    y*    z*(1-w),
            (1-x)*(1-y)*(1-z)*   w,
                x*(1-y)*(1-z)*   w,
            (1-x)*    y*(1-z)*   w,
                x*    y*(1-z)*   w,
            (1-x)*(1-y)*    z*   w,
                x*(1-y)*    z*   w,
            (1-x)*    y*    z*   w,
                x*    y*    z*   w
        ])
    
    def _evaluate_deriv(self, ip):
        x, y, z, w = ip

        B = [
            [1-x, 1-y, 1-z, 1-w],   # B0(t) = 1 - t
            [  x,   y,   z,   w]    # B1(t) = t
        ]

        dB = [
            [-1, -1, -1, -1],   # B0'(t) = -1
            [ 1,  1,  1,  1]     # B1'(t) = +1
        ]

        grads = np.zeros((4, 16))
        idx = 0
        for i in [0,1]:
            for j in [0,1]:
                for k in [0,1]:
                    for l in [0,1]:
                        # phi = B_i(x) B_j(y) B_k(z) B_l(w)
                        # derivative wrt x:
                        grads[0, idx] = dB[i][0] * B[j][1] * B[k][2] * B[l][3]
                        # derivative wrt y:
                        grads[1, idx] = B[i][0] * dB[j][1] * B[k][2] * B[l][3]
                        # derivative wrt z:
                        grads[2, idx] = B[i][0] * B[j][1] * dB[k][2] * B[l][3]
                        # derivative wrt w:
                        grads[3, idx] = B[i][0] * B[j][1] * B[k][2] * dB[l][3]

                        idx += 1

        return grads



    def __str__(self):
        return "P1 Tesserakt Finite Element"
    

class TriangleFE(FE_4D):
    """
    Abstract base class for finite elements on triangles.
    """    

    def __init__(self):
        super().__init__()
        self.eltype = "hypertriangle"

    @abstractmethod
    def _evaluate_id(self, ip):
        raise Exception("Not implemented - Base class should not be used") 

class P1_Hypertriangle_FE(TriangleFE,Lagrange_FE):
    """
    Linear (P1) finite element on the reference 4D simplex.
    """
    ndof = 5
    order = 1

    def __init__(self):
        super().__init__()
        # Knoten: die 5 Ecken des 4D-Simplex
        self.nodes = [
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
        ]

    def _evaluate_id(self, ip):
        """
        Evaluates the P1 basis functions at a given point inside the reference 4-simplex.

        Parameters:
            ip (np.ndarray): Point in 4D simplex (length 4)

        Returns:
            np.ndarray: Vector of length 5 (values of basis functions)
        """
        x0, x1, x2, x3 = ip
        lamb = [1 - x0 - x1 - x2 - x3, x0, x1, x2, x3]
        return np.array(lamb)
    
    def _evaluate_deriv(self, ip):
        """
        Returns the gradients of the 5 P1 shape functions on the reference 4-simplex.

        Output shape: (4, 5)
        grads[:, i] = ∇φ_i
        """

        # Gradients in the 4D REFERENCE simplex
        grads_ref = np.array([
            [-1.0, -1.0, -1.0, -1.0],  # grad φ0
            [ 1.0,  0.0,  0.0,  0.0],  # grad φ1
            [ 0.0,  1.0,  0.0,  0.0],  # grad φ2
            [ 0.0,  0.0,  1.0,  0.0],  # grad φ3
            [ 0.0,  0.0,  0.0,  1.0],  # grad φ4
        ]).T   # shape -> (4,5)

        return grads_ref


    def __str__(self):
        return "P1 4D-Simplex Finite Element"  
    
class P2_Hypertriangle_FE(TriangleFE, Lagrange_FE):
    """
    Quadratic (P2) Lagrange element on the reference 4D simplex.
    Nodes: 5 vertices + 10 edge midpoints.
    """
    ndof = 15
    order = 2

    def __init__(self):
        super().__init__()
        # vertices of reference 4-simplex
        v0 = np.array([0.0, 0.0, 0.0, 0.0])
        v1 = np.array([1.0, 0.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0, 0.0])
        v3 = np.array([0.0, 0.0, 1.0, 0.0])
        v4 = np.array([0.0, 0.0, 0.0, 1.0])

        self.vertices = [v0, v1, v2, v3, v4]

        # edge list in terms of vertex indices
        self.edge_pairs = [
            (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 2), (1, 3), (1, 4),
            (2, 3), (2, 4),
            (3, 4),
        ]

        mids = [(self.vertices[i] + self.vertices[j]) * 0.5
                for (i, j) in self.edge_pairs]

        # 5 vertex nodes + 10 edge-midpoint nodes
        self.nodes = self.vertices + mids

    def _evaluate_id(self, ip):
        """
        Evaluate P2 basis at point ip (x0,x1,x2,x3) in reference 4-simplex.
        Uses barycentric coordinates lambda_0,...,lambda_4.
        """
        x0, x1, x2, x3 = ip
        lamb = np.array([1.0 - x0 - x1 - x2 - x3,
                         x0, x1, x2, x3])
        ret = np.empty(self.ndof)

        for i in range(5):
            ret[i] = lamb[i] * (2.0 * lamb[i] - 1.0)

        for k, (i, j) in enumerate(self.edge_pairs):
            ret[5 + k] = 4.0 * lamb[i] * lamb[j]

        return ret
    
    def _evaluate_deriv(self, ip):
        """
        Return derivatives of all 15 shape functions at reference point ip.
        Output shape: (4, 15)
        """
        x0, x1, x2, x3 = ip

        # barycentric coordinates
        lamb = np.array([
            1.0 - x0 - x1 - x2 - x3,
            x0, x1, x2, x3
        ])

        # gradients of barycentric coordinates
        grad_lamb = np.array([
            [-1.0, -1.0, -1.0, -1.0],
            [ 1.0,  0.0,  0.0,  0.0],
            [ 0.0,  1.0,  0.0,  0.0],
            [ 0.0,  0.0,  1.0,  0.0],
            [ 0.0,  0.0,  0.0,  1.0],
        ])  # shape (5,4)

        grads = np.zeros((4, self.ndof))

        # vertex dofs: φ_i = λ_i (2 λ_i - 1)
        for i in range(5):
            grads[:, i] = (4.0 * lamb[i] - 1.0) * grad_lamb[i]

        # edge dofs: φ_ij = 4 λ_i λ_j
        for k, (i, j) in enumerate(self.edge_pairs):
            grads[:, 5 + k] = 4.0 * (lamb[j] * grad_lamb[i] + lamb[i] * grad_lamb[j])

        return grads
    
    def _evaluate_laplace(self, ip):
        """
        Compute the reference-Laplacian (sum of second ξ-derivatives)
        of all 15 P2 shape functions at point ip.
        Output: array shape (15,)
        """
        x0, x1, x2, x3 = ip

        lamb = np.array([1 - x0 - x1 - x2 - x3,
                        x0, x1, x2, x3])

        grad_lamb = np.array([
            [-1.0, -1.0, -1.0, -1.0],
            [ 1.0,  0.0,  0.0,  0.0],
            [ 0.0,  1.0,  0.0,  0.0],
            [ 0.0,  0.0,  1.0,  0.0],
            [ 0.0,  0.0,  0.0,  1.0],
        ])

        lap = np.zeros(self.ndof)

        # vertex dofs φ_i = 2λ_i^2 - λ_i
        for i in range(5):
            # Δ φ_i = 4 * ||grad λ_i||^2
            lap[i] = 4.0 * np.dot(grad_lamb[i], grad_lamb[i])

        # edge dofs φ_ij = 4 λ_i λ_j
        for k, (i, j) in enumerate(self.edge_pairs):
            # Δ φ_ij = 4 * 2 * (grad λ_i · grad λ_j)
            lap[5 + k] = 8.0 * np.dot(grad_lamb[i], grad_lamb[j])

        return lap


    def __str__(self):
        return "P2 4D-Simplex Finite Element"
