import numpy as np
import pytest

from methodsnm.mesh_4d import UnstructuredHypertriangleMesh
from methodsnm.trafo import HypertriangleTransformation
from netgen.csg import unit_cube
from ngsolve import Mesh,VOL,specialcf

def manual_simplex_volume(pts):
    """
    Volume of the integral should be |det|/24.
    """
    a = pts[0]
    M = np.column_stack([pts[i] - a for i in range(1,5)])
    return abs(np.linalg.det(M)) / 24.0


@pytest.fixture
def mesh4d():
    T = 2
    ngmesh = Mesh(unit_cube.GenerateMesh(maxh=0.5))
    return UnstructuredHypertriangleMesh(T, ngmesh)


def test_jacobian_volume(mesh4d):
    """
    Test if Trafo returns the simplex element
    """
    el = 0
    trafo = mesh4d.trafo(el)

    pts = mesh4d.points[mesh4d.elements()[el]]
    v_manual = manual_simplex_volume(pts)
    v_trafo = abs(np.linalg.det(trafo.jac))
    assert np.isclose(v_manual*24, v_trafo, atol=1e-12)

def test_affine_mapping(mesh4d):
    el = 0
    trafo = mesh4d.trafo(el)
    verts = mesh4d.points[mesh4d.elements()[el]]
    eps = np.random.rand(4)

    a, b, c, d, e = verts
    expected = (a
                + eps[0] * (b - a)
                + eps[1] * (c - a)
                + eps[2] * (d - a)
                + eps[3] * (e - a))

    got = trafo._map(eps)

    assert np.allclose(expected, got, atol=1e-12)


def test_gradient_physical(mesh4d):
    el = 0
    trafo = mesh4d.trafo(el)

    grads_ref = np.array([
        [-1, -1, -1, -1],
        [ 1,  0,  0,  0],
        [ 0,  1,  0,  0],
        [ 0,  0,  1,  0],
        [ 0,  0,  0,  1],
    ])  # (5,4)

    JinvT = np.linalg.inv(trafo.jac).T
    grads_global_expected = grads_ref @ JinvT

    from methodsnm.fe_4d import P1_Hypertriangle_FE
    fe = P1_Hypertriangle_FE()
    grads_ref_from_fe = fe._evaluate_deriv([0,0,0,0]).T  # (5,4)

    grads_global_from_fe = grads_ref_from_fe @ JinvT

    assert np.allclose(grads_global_from_fe, grads_global_expected, atol=1e-12)




