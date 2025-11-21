import pytest
import import_hack
from methodsnm.fes import *
from numpy.linalg import norm
from methodsnm.mesh_1d import *
from methodsnm.mesh_2d import *
from methodsnm.mesh_4d import *
from netgen.csg import unit_cube
from ngsolve import Mesh,VOL,specialcf
from methodsnm.meshfct import FEFunction

@pytest.mark.parametrize("ne", [1,2,10])
def test_P1_FES_1d(ne):
    mesh = Mesh1D((0,1),ne)
    p1fes = P1_Segments_Space(mesh)
    dofs = p1fes.element_dofs(0, bndry=True)
    assert dofs == [0]
    nbe = len(mesh.elements(bndry=True))
    dofs = p1fes.element_dofs(nbe-1, bndry=True)
    assert dofs == [ne]

def test_P1_FES_2d():
    mesh = StructuredRectangleMesh(2, 2)
    p1fes = P1_Triangle_Space(mesh)
    dof_marked = np.zeros(9, dtype=int)
    for belnr, verts in enumerate(mesh.elements(bndry=True)):
        dofs = p1fes.element_dofs(belnr, bndry=True)
        assert len(dofs) == 2
        dof_marked[dofs] += 1
    assert all(d == 2 for i,d in enumerate(dof_marked) if i != 4)

def test_P3_FES_2d():
    mesh = StructuredRectangleMesh(2, 2)
    p3fes = P3_Triangle_Space(mesh)
    dof_marked = np.zeros(p3fes.ndof, dtype=int)
    for belnr, verts in enumerate(mesh.elements(bndry=True)):
        dofs = p3fes.element_dofs(belnr, bndry=True)
        assert len(dofs) == 4
        dof_marked[dofs] += 1
    dof_marked_ref = np.array([2,2,2,2,0,2,2,2,2,
                               1,1,1,1,0,0,0,0,1,1,1,1,
                               1,1,1,1,0,0,0,0,1,1,1,1,
                               0,0,0,0,0,0,0,0])
    assert (dof_marked_ref == dof_marked_ref).all()

@pytest.fixture
def mesh():
    T = 2
    ngmesh = Mesh(unit_cube.GenerateMesh(maxh=0.5))
    mesh = UnstructuredHypertriangleMesh(T, ngmesh)
    return mesh


def test_P1_Hypertriangle_dofs_per_element(mesh):
    """
    For each 4D simplex, the P1 space must assign exactly
    the 5 vertex DOFs = global vertex numbers.
    """
    fes = P1_Hypertriangle_Space(mesh)

    for elnr, verts in enumerate(mesh.elements()):
        dofs = fes.element_dofs(elnr)
        assert len(dofs) == 5
        assert np.allclose(dofs, verts)

def test_fespace_boundary_sets_consistent(mesh):
    """
    Testing if the boundary decomposes in three disjoint pieces:
    boundary = initial ∪ top ∪ side
    """
    fes = P1_Hypertriangle_Space(mesh)

    I = fes.initial_vertices()
    T = fes.top_vertices()
    B = fes.boundary_vertices()

    S = B - I - T

    assert I.isdisjoint(T)
    assert I.isdisjoint(S)
    assert T.isdisjoint(S)
    assert I | T | S == B

def test_P2_Hypertriangle_element_dofs_structure(mesh):
    fes = P2_Hypertriangle_Space(mesh)

    for elnr, verts in enumerate(mesh.elements()):
        dofs = fes.element_dofs(elnr)

        assert len(dofs) == 15
        assert np.allclose(dofs[:5], verts)
        edge_dofs = dofs[5:]
        assert np.all(edge_dofs >= fes.nv)
        assert len(set(dofs)) == 15

def test_P2_Hypertriangle_edge_connectivity(mesh):
    fes = P2_Hypertriangle_Space(mesh)
    global_edges = set(tuple(sorted(e)) for e in mesh.edges)

    for elnr, verts in enumerate(mesh.elements()):
        el_edges = mesh.hypercell2edge[elnr]

        for gid in el_edges:
            v1, v2 = gid
            assert tuple(sorted((v1, v2))) in global_edges

def test_P2_Hypertriangle_boundary_dofs(mesh):
    fes = P2_Hypertriangle_Space(mesh)

    Bv = fes.boundary_vertices()
    bdofs = fes.boundary_dofs()

    for v in Bv:
        assert v in bdofs

    # Edge DOFs must be circumcised by Vertices on the boundary 
    for eid, (v1, v2) in enumerate(mesh.edges):
        if v1 in Bv and v2 in Bv:
            assert fes.nv + eid in bdofs
        else:
            assert (fes.nv + eid) not in bdofs

def is_on_boundary(x, tol=1e-12):
    return (
        abs(x[0]) < tol or abs(x[0]-1) < tol or
        abs(x[1]) < tol or abs(x[1]-1) < tol or
        abs(x[2]) < tol or abs(x[2]-1) < tol or
        abs(x[3]) < tol or abs(x[3]-1) < tol
    )

def test_boundary_vertices_geometric(mesh):
    bset = set(mesh.bndry_vertices)

    for vid, x in enumerate(mesh.points):
        geom = is_on_boundary(x)
        marked = vid in bset

        if geom:
            assert marked, f"Vertex {vid} is on geometric boundary, but NOT in mesh.bndry_vertices"
        else:
            assert not marked, f"Vertex {vid} is NOT on geometric boundary, but was marked as boundary"

def test_P2_Hypertriangle_boundary_sets_consistent(mesh):
    fes = P2_Hypertriangle_Space(mesh)

    I = fes.initial_vertices()
    T = fes.top_vertices()
    B = fes.boundary_vertices()

    S = B - I - T

    assert I.isdisjoint(T)
    assert I.isdisjoint(S)
    assert T.isdisjoint(S)

    assert I | T | S == B


def test_P2_Hypertriangle_global_dofs_complete(mesh):
    fes = P2_Hypertriangle_Space(mesh)

    assert fes.ndof == fes.nv + fes.nedges

    assert set(range(fes.ndof)) == \
           set(range(fes.nv)) | set(range(fes.nv, fes.nv + fes.nedges))


def test_P2_setP2_on_nodes(mesh):
    V = P2_Hypertriangle_Space(mesh)
    u = FEFunction(V)
    f = lambda x: 1 + 2*x[0] - 3*x[1] + 5*x[2] - 7*x[3]
    u._set_P2(f)
    nv = V.nv
    for v in range(nv):
        x = mesh.points[v]
        assert abs(u.vector[v] - f(x)) < 1e-12

    for eid, (v1, v2) in enumerate(mesh.edges):
        m = 0.5*(mesh.points[v1] + mesh.points[v2])
        g = nv + eid
        assert abs(u.vector[g] - f(m)) < 1e-12

def test_P2_reproduce_quadratic_on_mesh(mesh):
    V = P2_Hypertriangle_Space(mesh)

    u_ex = lambda x: x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2
    uh = FEFunction(V)
    uh._set_P2(u_ex)

    import numpy as np
    for el in range(5):  
        trafo = mesh.trafo(el)
        verts = mesh.points[mesh.elements()[el]]

        for _ in range(5):
            lamb = np.random.rand(5)
            lamb /= np.sum(lamb)
            x = lamb @ verts   

            val_fe  = uh._evaluate(x, trafo)
            val_ex  = u_ex(x)

            assert abs(val_fe - val_ex) < 1e-10

def test_P2_reference_vs_physical_mapping(mesh):
    V  = P2_Hypertriangle_Space(mesh)
    fe = P2_Hypertriangle_FE()

    el = 0
    trafo = mesh.trafo(el)
    verts = mesh.points[mesh.elements()[el]]
    dofs  = V.element_dofs(el)

    uh = FEFunction(V)

    import numpy as np
    for _ in range(5):
        lamb = np.random.rand(5)
        lamb /= np.sum(lamb)

        ref = np.array([lamb[1], lamb[2], lamb[3], lamb[4]])
        x_phys = lamb @ verts

        phi_ref = fe._evaluate_id(ref)  

        for j, gdof in enumerate(dofs):
            uh.vector[:] = 0.0
            uh.vector[gdof] = 1.0

            val_phys = uh._evaluate(x_phys, trafo)

            assert abs(val_phys - phi_ref[j]) < 1e-10, \
                f"el={el}, local dof={j}, global dof={gdof}"
