import pytest
import numpy as np
import import_hack
from methodsnm.fes import *
from methodsnm.vectorspace import Productspace
from methodsnm.fe_vector import BlockFE
from methodsnm.mesh_1d import *
from methodsnm.mesh_2d import *
from methodsnm.mesh_4d import *
from netgen.csg import unit_cube
from ngsolve import Mesh,VOL,specialcf

@pytest.fixture
def mesh1d():
    return Mesh1D((0, 1), 4)

@pytest.fixture
def mesh2d():
    return StructuredRectangleMesh(3, 3)

@pytest.fixture
def mesh4d():
    T = 2
    ngmesh = Mesh(unit_cube.GenerateMesh(maxh=0.5))
    mesh = UnstructuredHypertriangleMesh(T, ngmesh)
    return mesh


# ----------------------------------------------------------
#  1. Offsets correct
# ----------------------------------------------------------

@pytest.mark.parametrize("nblocks", [1, 2, 3])
def test_offsets_hypertriangle(mesh4d, nblocks):
    spaces = [P1_Hypertriangle_Space(mesh4d) for _ in range(nblocks)]
    W = Productspace(spaces)

    assert all(W.offsets[i] <= W.offsets[i+1] for i in range(len(W.offsets)-1))

    assert W.offsets[-1] == sum(V.ndof for V in spaces)


# ----------------------------------------------------------
#  2. Local Element DOFs mapping to global DOFs 
# ----------------------------------------------------------

def test_element_dofs_hypertriangle(mesh4d):
    V1 = P1_Hypertriangle_Space(mesh4d)
    V2 = P1_Hypertriangle_Space(mesh4d)
    W = Productspace([V1, V2])

    el = 0

    # local dofs
    d1 = np.array(V1.element_dofs(el))
    d2 = np.array(V2.element_dofs(el)) + W.offsets[1]

    expected = np.concatenate([d1, d2])
    got = W.element_dofs(el)

    assert np.allclose(got, expected)


# ----------------------------------------------------------
#  3. get_freedofs 
# ----------------------------------------------------------

def test_get_freedofs(mesh4d):
    V1 = P1_Hypertriangle_Space(mesh4d)
    V2 = P1_Hypertriangle_Space(mesh4d)

    W = Productspace([V1, V2])
    blocked = {0: [0, 1], 1: [3]}

    freedofs = W.get_freedofs(blocked)

    mask = np.ones(W.ndof, dtype=bool)

    mask[0] = False
    mask[1] = False
    off1 = W.offsets[1]
    mask[off1 + 3] = False

    expected = np.where(mask)[0]

    assert np.allclose(freedofs, expected)


# ----------------------------------------------------------
#  4. BlockFE 
# ----------------------------------------------------------

def test_blockfe(mesh4d):
    V1 = P1_Hypertriangle_Space(mesh4d)
    V2 = P1_Hypertriangle_Space(mesh4d)

    W = Productspace([V1, V2])

    fe = W.finite_element(0)
    assert isinstance(fe, BlockFE)
    assert fe.nblocks == 2
    assert len(fe.fes) == 2
    assert fe.ndof == V1.fe.ndof + V2.fe.ndof
    assert fe.block_offsets == [0, V1.fe.ndof, V1.fe.ndof + V2.fe.ndof]



# ----------------------------------------------------------
#  5. Boundary-DOFs pro Block 
# ----------------------------------------------------------

def test_boundary_dofs_use_get_freedofs(mesh4d):
    V1 = P1_Hypertriangle_Space(mesh4d)
    V2 = P1_Hypertriangle_Space(mesh4d)
    W = Productspace([V1, V2])

    # Component-wise boundary DOFs
    B1 = sorted(V1.boundary_vertices())
    B2 = sorted(V2.boundary_vertices())
    blocked = {0: B1,1: B2,}

    free = W.get_freedofs(blocked)
    mask = np.ones(W.ndof, dtype=bool)
    for d in B1:
        mask[d] = False
    off = W.offsets[1]
    for d in B2:
        mask[off + d] = False

    expected = np.where(mask)[0]

    assert np.allclose(free, expected)



# ----------------------------------------------------------
#  6. Complete DOF coverage
# ----------------------------------------------------------

def test_dof_cover(mesh4d):
    V1 = P1_Hypertriangle_Space(mesh4d)
    V2 = P1_Hypertriangle_Space(mesh4d)
    W = Productspace([V1, V2])

    dofs = []

    for e in range(len(mesh4d.elements())):
        dofs += list(W.element_dofs(e))

    dofs = sorted(set(dofs))
    assert dofs == list(range(W.ndof))
