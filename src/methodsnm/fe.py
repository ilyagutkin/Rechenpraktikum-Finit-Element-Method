from abc import ABC, abstractmethod
import numpy as np
from numpy import array

class FE(ABC):
    """
    This is the base class for all **scalar** finite element classes. 
    It provides a template for the evaluate method.
    """
    ndof = None # number of degrees of freedom
    order = None # polynomial order
    eltype = None # e.g. "segment", "triangle", ...
    dim = None # dimension of the domain
    def __init__(self):
        pass

    @abstractmethod
    def _evaluate_id(self, ip):
        """
        Evaluates the finite element at the given integration point.

        Parameters:
        ip (numpy.array): The integration point at which to evaluate the finite element.

        Returns:
        numpy.array: The values of the finite element basis fcts. at the given integration point.
        """
        raise Exception("Not implemented - Base class should not be used")

    def _evaluate_id_array(self, ips):
        """
        Evaluates the finite element at multiple integration points at once.
        Base class implementation is a simple loop over the integration points.
        Performance gains can only be obtained by overwriting this method.

        Parameters:
        ips (numpy.array): The integration point at which to evaluate the finite element.

        Returns:
        numpy.ndarray: The values of the finite element at the given integration points.
                       shape: (len(ips), ndof)
        """
        ret = np.empty((len(ips), self.ndof ))
        for i in range(len(ips)):
            ret[i,:] = self._evaluate_id(ips[i])
        return ret

    def _evaluate_deriv_array(self, ips ,direction=None):
        """
        Evaluates the derivative of finite element at multiple integration points at once.
        Base class implementation is a simple loop over the integration points.
        Performance gains can only be obtained by overwriting this method.

        Parameters:
        ips (numpy.array): The integration point at which to evaluate the finite element.

        Returns:
        numpy.ndarray: The values of the finite element at the given integration points.
                       shape: (len(ips), dim, ndof)
        """
        
        if direction is not None:
            ret = np.empty((len(ips),self.ndof))
            for i in range(len(ips)):
                ret[i,:] = self._evaluate_deriv(ips[i])[direction]
            return ret
        else:
            ret = np.empty((len(ips), self.dim, self.ndof))
            for i in range(len(ips)):
                ret[i,:,:] = self._evaluate_deriv(ips[i])
            return ret

    @abstractmethod
    def _evaluate_deriv(self, ip):
        """
        Evaluates the derivative of a finite element at the given integration point.

        Parameters:
        ip (numpy.array): The integration point at which to evaluate the finite element.

        Returns:
        numpy.array: The values of the derivative of the finite element basis fcts. at the given integration point.
                     shape (dim, ndof)
        """
        raise Exception("Not implemented - Base class should not be used")

    def _evaluate_laplace_array(self, ips):
        """
        Evaluates the Laplacian of finite element at multiple integration points at once.
        Base class implementation is a simple loop over the integration points.
        Performance gains can only be obtained by overwriting this method.

        Parameters:
        ips (numpy.array): The integration points at which to evaluate the finite element.

        Returns:
        numpy.ndarray: The values of the Laplacian of the finite element at the given integration points.
                       shape: (len(ips), ndof)
        """
        ret = np.empty((len(ips), self.ndof))
        for i in range(len(ips)):
            ret[i,:] = self._evaluate_laplace(ips[i])
        return ret

    def evaluate(self, ip, deriv=False, direction =None, laplace=False):
        """
        Evaluates the (derivative of) finite element at given integration point(s).

        Parameters:
        ip (numpy.array): The integration point(s) at which to evaluate the finite element.
        deriv (bool): Whether to evaluate the derivative of the finite element (or identity).
        direction (int, optional): Direction index for derivative evaluation.
        laplace (bool): Whether to evaluate the Laplacian (sum of second derivatives) of the finite element.

        Returns:
        numpy.array: The values of the finite element basis fcts. at the given integration point.
           shape:                  (ndof) (for single ip) 
                  or          (dim, ndof) (for single ip and deriv = True)
                  or (len(ip),      ndof) (for multiple ips)
                  or (len(ip), dim, ndof) (for multiple ips and deriv = True)
                  or          (ndof) (for single ip and laplace = True)
                  or (len(ip), ndof) (for multiple ips and laplace = True)
        """
        if isinstance(ip, np.ndarray):
            if ip.ndim == 1:
                if laplace:
                    return self._evaluate_laplace(ip)
                elif deriv:
                    if direction is not None:
                        return self._evaluate_deriv(ip)[direction]
                    return self._evaluate_deriv(ip)
                else:
                    return self._evaluate_id(ip)
            else:
                if laplace:
                    return self._evaluate_laplace_array(ip)
                elif deriv:
                    if direction is not None:
                        return self._evaluate_deriv_array(ip,direction)
                    return self._evaluate_deriv_array(ip)
                else:
                    return self._evaluate_id_array(ip)
        else:
            raise Exception("Invalid input")

     

class Lagrange_FE(FE):
    """
    This class represents a Lagrange finite element.
    A Lagrange finite element associates the dofs with the nodes of the element.
    """
    nodes = None

    def __str__(self):
        return f"Lagrange-FE-obj(order={self.order},nodes={self.ndof})"



class Node_FE(FE):
    """
    FE for a point (node).
    """    

    def __init__(self):
        self.eltype = "point"
        self.dim = 0
        self.ndof = 1

    def _evaluate_id(self, ip):
        return np.ones((1,))

    def _evaluate_id_array(self, ip):
        return np.ones((len(ip),1))

    def _evaluate_deriv(self, ip):
        raise Exception("Derivative of node FE should not be called")
