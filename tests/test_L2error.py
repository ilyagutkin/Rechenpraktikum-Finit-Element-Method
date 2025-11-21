import numpy as np
import import_hack
import pytest
from methodsnm.fe_4d import *
from methodsnm.mesh_4d import *
from methodsnm.meshfct import FEFunction, GlobalFunction
from methodsnm.forms import compute_difference_L2
from netgen.csg import unit_cube
from ngsolve import Mesh,VOL,specialcf

ngmesh = Mesh(unit_cube.GenerateMesh(maxh=0.5))
def test_L2_zero_for_linear_function():
    mesh = UnstructuredHypertriangleMesh(T=1, ngmesh=ngmesh)
    fe   = P1_Hypertriangle_FE()

    # Dummy FES
    class DummyFES:
        def __init__(self, mesh, fe):
            self.mesh = mesh
            self.fe   = fe
            self.ndof = mesh.points.shape[0]
        def element_dofs(self, el):
            return mesh.hypercells[el]
        def finite_element(self, el):
            return fe

    fes = DummyFES(mesh, fe)

    f_exact = lambda x: 1 + x[0] - 2*x[1] + 0.5*x[2] + 3*x[3]
    uh = FEFunction(fes)
    uh._set(f_exact)
    uex = GlobalFunction(f_exact, mesh)

    err = compute_difference_L2(uh, uex, mesh)

    assert err < 1e-12
