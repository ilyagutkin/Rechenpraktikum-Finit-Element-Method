from import_hack import *
from methodsnm.mesh_4d import *
from methodsnm.visualize import *
import math
import numpy as np
from numpy import exp,pi
from methodsnm.vectorspace import *
from methodsnm.fes import *
from netgen.csg import unit_cube
from ngsolve import Mesh,VOL,specialcf

def list_diff(a, b):
    """Entfernt alle Elemente aus Liste a, die in Liste b enthalten sind."""
    return [x for x in a if x not in b]

m = 2
T = 2
epsi = 0.1
ngmesh = Mesh(unit_cube.GenerateMesh(maxh=0.25))
mesh = UnstructuredHypertriangleMesh(T,ngmesh)
mesh = UnstructuredHypertriangleMesh(T,ngmesh)
V1 = P2_Hypertriangle_Space(mesh)
V2 = P2_Hypertriangle_Space(mesh)
V3 = P2_Hypertriangle_Space(mesh)
Q = P1_Hypertriangle_Space(mesh)
Fes = Productspace([V1,V2,V3,Q])

# Parameter für Konvektionsdominanz
nu_list = [1]
for nu in nu_list:      # Extrem kleine Viskosität
    wx, wy, wz = 1.0, 1.0, 1.0
    K = 2          # Frequenz: Erhöhe auf 4 oder 8 für extreme Wiggles
    PI = np.pi

    # Windvektor in 4D (Raum + Zeit)
    w = ConstantVectorFunction(np.array([wx, wy, wz, 1.0]), mesh=mesh)
    NU = GlobalFunction(lambda x: nu, mesh = mesh)

    # 1. Exakte Lösung u (Divergenzfrei)
    # Diese Form sorgt dafür, dass div(u) = 0
    uex1 = lambda x:  np.exp(-x[3]) * np.sin(K*PI*x[0]) * np.cos(K*PI*x[1]) * np.cos(K*PI*x[2])
    uex2 = lambda x: -0.5 * np.exp(-x[3]) * np.cos(K*PI*x[0]) * np.sin(K*PI*x[1]) * np.cos(K*PI*x[2])
    uex3 = lambda x: -0.5 * np.exp(-x[3]) * np.cos(K*PI*x[0]) * np.cos(K*PI*x[1]) * np.sin(K*PI*x[2])

    # 2. Druck p (Wir setzen ihn auf 0 oder eine einfache Funktion)
    pex = lambda x: 0.0 

    # 3. Rechte Seiten f (Manuell abgeleitet für u_t + w*grad(u) - nu*Laplace(u) + grad(p))
    # Hilfskonstante für die Diffusion: 3 * (K*PI)**2
    diff_const = 3.0 * (K * PI)**2 * nu

    def f1_func(x):
        e = np.exp(-x[3])
        s0, c0 = np.sin(K*PI*x[0]), np.cos(K*PI*x[0])
        s1, c1 = np.sin(K*PI*x[1]), np.cos(K*PI*x[1])
        s2, c2 = np.sin(K*PI*x[2]), np.cos(K*PI*x[2])
        
        val = e * (
            -s0*c1*c2 +                         # u_t
            K*PI * (wx*c0*c1*c2 - wy*s0*s1*c2 - wz*s0*c1*s2) + # w * grad(u)
            diff_const * s0*c1*c2               # -nu * Laplace(u)
        )
        return val

    def f2_func(x):
        e = np.exp(-x[3])
        s0, c0 = np.sin(K*PI*x[0]), np.cos(K*PI*x[0])
        s1, c1 = np.sin(K*PI*x[1]), np.cos(K*PI*x[1])
        s2, c2 = np.sin(K*PI*x[2]), np.cos(K*PI*x[2])
        
        val = e * (
            0.5*c0*s1*c2 +                      # u_t
            -0.5*K*PI * (-wx*s0*s1*c2 + wy*c0*c1*c2 - wz*c0*s1*s2) + # w * grad(u)
            -0.5*diff_const * c0*s1*c2           # -nu * Laplace(u)
        )
        return val

    def f3_func(x):
        e = np.exp(-x[3])
        s0, c0 = np.sin(K*PI*x[0]), np.cos(K*PI*x[0])
        s1, c1 = np.sin(K*PI*x[1]), np.cos(K*PI*x[1])
        s2, c2 = np.sin(K*PI*x[2]), np.cos(K*PI*x[2])
        
        val = e * (
            0.5*c0*c1*s2 +                      # u_t
            -0.5*K*PI * (-wx*s0*c1*s2 - wy*c0*s1*s2 + wz*c0*c1*c2) + # w * grad(u)
            -0.5*diff_const * c0*c1*s2           # -nu * Laplace(u)
        )
        return val

    # GlobalFunctions für deine Bibliothek
    fx = GlobalFunction(f1_func, mesh=mesh)
    fy = GlobalFunction(f2_func, mesh=mesh)
    fz = GlobalFunction(f3_func, mesh=mesh)
    g  = GlobalFunction(lambda x: 0.0, mesh=mesh) # div(u) = 0

    from methodsnm.forms import *
    from methodsnm.formint import *
    BF = BilinearVectorForm(Fes)
    c = ConstantFunction(0.0000001 ,mesh = mesh)

    top = mesh.top_bndry_vertices
    initial = mesh.initial_bndry_vertices
    bndry = V1.boundary_dofs()
    bndry_top = V1.boundary_dofs_from_vertex_set(top)
    bndry_not_free =list_diff(bndry, bndry_top)
    freedofs = [d for d in range(V1.ndof) if d not in bndry_not_free]
    print (len(freedofs), " free dofs, ", len(bndry_not_free), " boundary dofs (not free)",Fes.ndof, " total dofs")
    for b in range(3):
        BF.add_block_integrator(b, b, LaplaceIntegral_without_time(NU))
        BF.add_block_integrator(b, b, ConvectionIntegral(w))
        #BF.add_block_integrator(b, b, SUPGIntegral(w,nu))
        BF.add_block_integrator(b, b, SUPGProjIntegral(w,nu))
    for b in range(3):
        BF.add_block_integrator(3, b, DivUQIntegrator())
        BF.add_block_integrator(b, 3, DivVPIntegrator())
    BF.add_block_integrator(3, 3, MassIntegral(c)) #to avoid singular matrix
    BF.assemble()

    LF = LinearVectorForm(Fes)
    LF.add_block_integrator(0, SourceIntegral(fx))
    #LF.add_block_integrator(0, SUPGSourceIntegral(fx, w, nu))
    LF.add_block_integrator(0,SUPGSourceIntegralProjected(fx,w,nu))
    LF.add_block_integrator(1, SourceIntegral(fy))
    #LF.add_block_integrator(1, SUPGSourceIntegral(fy, w, nu))
    LF.add_block_integrator(1,SUPGSourceIntegralProjected(fy,w,nu))
    LF.add_block_integrator(2, SourceIntegral(fz))
    #LF.add_block_integrator(2, SUPGSourceIntegral(fz, w, nu))
    LF.add_block_integrator(2,SUPGSourceIntegralProjected(fz,w,nu))
    #LF.add_block_integrator(3, SourceIntegral(g))
    LF.assemble()


    u = FEVectorFunction(Fes)
    u._set_P2({0: (uex1,bndry), 1: (uex2,bndry), 2: (uex3,bndry)})


    res = LF.vector - BF.matrix.dot(u.vector)
    freedofs_global = Fes.get_freedofs({0: bndry, 1: bndry, 2: bndry})
    from methodsnm.solver import solve_on_freedofs
    u.vector += solve_on_freedofs(BF.matrix,res,freedofs_global)
    u1 , u2, u3 , p = u.blocks()

    from methodsnm.forms import compute_difference_L2,analyze_stability,compute_difference_SUPG
    u_ex1 = GlobalFunction(uex1, mesh = mesh)
    u_ex2 = GlobalFunction(uex2, mesh = mesh)
    u_ex3 = GlobalFunction(uex3, mesh = mesh)
    p_ex = GlobalFunction(pex, mesh = mesh)
    print(f"\nResults for nu = {nu}:")
    l2diff1 = compute_difference_L2(u1, u_ex1, mesh, intorder = 5)
    print("L2 difference u1:", l2diff1)
    l2diff2 = compute_difference_L2(u2, u_ex2, mesh, intorder = 5)
    print("L2 difference u2:", l2diff2)
    l2diff3 = compute_difference_L2(u3, u_ex3, mesh, intorder = 5)
    print("L2 difference u3:", l2diff3)
    l2diffp = compute_difference_L2(p, p_ex, mesh, intorder = 5)
    print("L2 difference p:", l2diffp)
    # supgdiff1 = compute_difference_SUPG(u1, u_ex1, mesh, w, nu, intorder = 5)
    # print("SUPG difference u1:", supgdiff1)
    # supgdiff2 = compute_difference_SUPG(u2, u_ex2, mesh, w, nu, intorder = 5)
    # print("SUPG difference u2:", supgdiff2)
    # supgdiff3 = compute_difference_SUPG(u3, u_ex3, mesh, w, nu, intorder = 5)
    # print("SUPG difference u3:", supgdiff3)