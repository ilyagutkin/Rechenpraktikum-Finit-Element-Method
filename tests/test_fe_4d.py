import numpy as np
import import_hack
import pytest
from methodsnm.fe_4d import *

# helper: random point in the 4D simplex
def random_point():
    x = np.random.rand(4)
    s = np.sum(x)
    if s >= 1.0:
        x = x / (s + 1e-14) * 0.5
    return x

def quadratic_polynomials():
    polys = []

    #constant
    polys.append(lambda x: 1)

    # linear
    for i in range(4):
        polys.append(lambda x, i=i: x[i])

    # quadratic
    for i in range(4):
        polys.append(lambda x, i=i: x[i] * x[i])

    # crossterms
    for i in range(4):
        for j in range(i+1, 4):
            polys.append(lambda x, i=i, j=j: x[i] * x[j])

    return polys

def quadratic_polynomials_with_grads():
    polys = []

    # constant
    polys.append((lambda x: 1.0,
                  lambda x: np.zeros(4)))

    # linear
    for i in range(4):
        polys.append((lambda x, i=i: x[i],
                      lambda x, i=i: np.eye(4)[i]))

    # quadratic
    for i in range(4):
        polys.append((lambda x, i=i: x[i] * x[i],
                      lambda x, i=i: np.array([2*x[i] if j==i else 0 for j in range(4)])))

    # crossterms
    for i in range(4):
        for j in range(i+1,4):
            polys.append((
                lambda x, i=i, j=j: x[i] * x[j],
                lambda x, i=i, j=j: np.array([
                    x[j] if k==i else
                    x[i] if k==j else
                    0.0
                    for k in range(4)
                ])
            ))

    return polys

def linear_polynomials():
    polys = []

    # constant
    polys.append(lambda x: 1.0)

    # linear Terms
    for i in range(4):
        polys.append(lambda x, i=i: x[i])

    return polys



# -------------------------------------------------------------------------
# 1. Partition of unity
# -------------------------------------------------------------------------
@pytest.mark.parametrize("i", range(10))
def test_partition_of_unity(i):
    fe = P2_Hypertriangle_FE()
    x = random_point()
    phi = fe._evaluate_id(x)
    assert abs(np.sum(phi) - 1.0) < 1e-12

# -------------------------------------------------------------------------
# 2. Gradient shape and dimension
# -------------------------------------------------------------------------
def test_gradient_shape():
    fe = P2_Hypertriangle_FE()
    x = np.zeros(4)
    grads = fe._evaluate_deriv(x)
    assert grads.shape == (4, fe.ndof)


# -------------------------------------------------------------------------
# 3. Sum of gradients must be zero 
# -------------------------------------------------------------------------
@pytest.mark.parametrize("i", range(5))
def test_gradients_sum_zero(i):
    fe = P2_Hypertriangle_FE()
    x = random_point()
    grads = fe._evaluate_deriv(x)  # shape (4,15)
    gsum = np.sum(grads, axis=1)   # sum over DOFs
    assert np.linalg.norm(gsum) < 1e-12

@pytest.mark.parametrize("poly", quadratic_polynomials())
def test_P2_reproduces_polynomials(poly):
    fe = P2_Hypertriangle_FE()

    nodes = fe.nodes             # List of 15 points in R^4
    ndof = fe.ndof

    # values of the Polynom at the vertices
    nodal_values = np.array([poly(p) for p in nodes])
    x = random_point()

    # FE-Interpolation: u_h(x) = sum phi_i(x) * u(node_i)
    phi = fe._evaluate_id(x)
    u_h = np.dot(phi, nodal_values)

    u_exact = poly(x)

    assert abs(u_h - u_exact) < 1e-14


@pytest.mark.parametrize("poly,grad", quadratic_polynomials_with_grads())
def test_P2_derivative_reproduction(poly, grad):
    fe = P2_Hypertriangle_FE()

    nodes = fe.nodes
    ndof = fe.ndof

    nodal_values = np.array([poly(p) for p in nodes])

    x = random_point()

    grads = fe._evaluate_deriv(x)
    grad_h = grads @ nodal_values     
    grad_exact = grad(x)

    assert np.linalg.norm(grad_h - grad_exact) < 1e-12


# Tests for the P1 element
# -------------------------------------------------------------------------
# 1. Partition of Unity (exact for P1)
# -------------------------------------------------------------------------
@pytest.mark.parametrize("i", range(20))
def P1test_partition_of_unity(i):
    fe = P1_Hypertriangle_FE()
    x = random_point()
    phi = fe._evaluate_id(x)
    assert abs(np.sum(phi) - 1.0) < 1e-14


# -------------------------------------------------------------------------
# 2. Non-negative (True for P1!)
# -------------------------------------------------------------------------
@pytest.mark.parametrize("i", range(20))
def P1test_shape_nonnegative(i):
    fe = P1_Hypertriangle_FE()
    x = random_point()
    phi = fe._evaluate_id(x)
    assert np.all(phi >= -1e-14)     # numerische Toleranz


# -------------------------------------------------------------------------
# 3. Gradientshape: (4, ndof) = (4, 5)
# -------------------------------------------------------------------------
def P1test_gradient_dimensions():
    fe = P1_Hypertriangle_FE()
    x = random_point()
    grads = fe._evaluate_deriv(x)
    assert grads.shape == (4, fe.ndof)


# -------------------------------------------------------------------------
# 4. Gradient-sum = 0 
# -------------------------------------------------------------------------
@pytest.mark.parametrize("i", range(10))
def P1test_gradient_sum_zero(i):
    fe = P1_Hypertriangle_FE()
    x = random_point()
    grads = fe._evaluate_deriv(x)
    assert np.linalg.norm(np.sum(grads, axis=1)) < 1e-14


# -------------------------------------------------------------------------
# 5. Linear Reproduction: P1 muss be exact
# -------------------------------------------------------------------------
@pytest.mark.parametrize("i", range(20))
def P1test_reproduce_linear_function(i):
    fe = P1_Hypertriangle_FE()
    x = random_point()
    phi = fe._evaluate_id(x)

    V = np.array([
        [0,0,0,0],  # λ0
        [1,0,0,0],  # λ1
        [0,1,0,0],  # λ2
        [0,0,1,0],  # λ3
        [0,0,0,1],  # λ4
    ])

    x_rec = np.zeros(4)
    for i in range(5):
        x_rec += phi[i] * V[i]

    assert np.linalg.norm(x_rec - x) < 1e-14


@pytest.mark.parametrize("poly", linear_polynomials())
def test_P1_reproduces_polynomials(poly):
    fe = P1_Hypertriangle_FE()

    nodes = fe.nodes      
    ndof = fe.ndof

    nodal_values = np.array([poly(p) for p in nodes])

    x = random_point()

    phi = fe._evaluate_id(x)
    u_h = np.dot(phi, nodal_values)

    u_exact = poly(x)

    assert abs(u_h - u_exact) < 1e-14

# ===================== BEGIN DERIVATIVE TESTS P1_HYPERTRIANGLE_FE =====================

def linear_polynomials_4d_with_grads():
    """
    Lineare Polynome auf dem 4D-Simplex und ihre Gradienten.
    Ein P1-Simplex muss diese exakt reproduzieren.
    """
    polys = []
    polys.append((lambda x: 1.0,
                  lambda x: np.zeros(4)))

    for i in range(4):
        polys.append(
            (lambda x, i=i: x[i],
             lambda x, i=i: np.eye(4)[i])
        )

    return polys


@pytest.mark.parametrize("poly,grad", linear_polynomials_4d_with_grads())
def test_p1_hypertriangle_derivative_reproduction(poly, grad):
    fe = P1_Hypertriangle_FE()

    nodes = fe.nodes
    nodal_vals = np.array([poly(p) for p in nodes])

    lamb = np.random.rand(5)
    lamb = lamb / np.sum(lamb)
    x = np.sum(lamb[:,None] * nodes, axis=0)

    grads = fe._evaluate_deriv(x)      # shape (4,5)

    # FE-Gradient: ∑ u(v_i) ∇φ_i(x)
    grad_h = grads @ nodal_vals

    grad_exact = grad(x)

    assert np.linalg.norm(grad_h - grad_exact) < 1e-12

# ====================== END DERIVATIVE TESTS P1_HYPERTRIANGLE_FE ======================


###
#Tests for the Tesseraktelment
###

@pytest.mark.parametrize("i", range(20))
def test_p1_tesserakt_partition_of_unity(i):
    fe = P1_Tesserakt_FE()
    x = np.random.rand(4)   
    phi = fe._evaluate_id(x)
    assert abs(np.sum(phi) - 1.0) < 1e-14

@pytest.mark.parametrize("i", range(20))
def test_p1_tesserakt_nonnegativity(i):
    fe = P1_Tesserakt_FE()
    x = np.random.rand(4)
    phi = fe._evaluate_id(x)
    assert np.all(phi >= -1e-14)

def test_p1_tesserakt_gradient_shape():
    fe = P1_Tesserakt_FE()
    x = np.random.rand(4)
    grads = fe._evaluate_deriv(x)
    assert grads.shape == (4, fe.ndof)

@pytest.mark.parametrize("i", range(20))
def test_p1_tesserakt_grad_sum_zero(i):
    fe = P1_Tesserakt_FE()
    x = np.random.rand(4)
    grads = fe._evaluate_deriv(x)
    assert np.linalg.norm(np.sum(grads, axis=1)) < 1e-14

def linear_polynomials_hypercube():
    polys = [lambda x: 1.0]
    for i in range(4):
        polys.append(lambda x, i=i: x[i])
    return polys


def separable_linear_polynomials():
    return [
        lambda x: 1.0,
        lambda x: x[0],
        lambda x: x[1],
        lambda x: x[2],
        lambda x: x[3],
    ]


@pytest.mark.parametrize("poly", separable_linear_polynomials())
def test_p1_tesserakt_reproduce_separable_linear(poly):
    fe = P1_Tesserakt_FE()

    nodes = fe.nodes
    nodal_vals = np.array([poly(p) for p in nodes])

    x = np.random.rand(4)

    phi = fe._evaluate_id(x)
    u_h = np.dot(phi, nodal_vals)

    u_exact = poly(x)

    assert abs(u_h - u_exact) < 1e-14
