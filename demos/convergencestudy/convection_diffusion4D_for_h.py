from import_hack import *
from methodsnm.mesh_4d import *
from methodsnm.visualize import *
from methodsnm.solver import solve_on_freedofs
from methodsnm.forms import *
from methodsnm.formint import *
from netgen.csg import unit_cube
from ngsolve import Mesh,VOL,specialcf
from methodsnm.fes import *
import numpy as np
from numpy import pi, sin, cos, exp
import matplotlib.pyplot as plt
import pandas as pd
import os

def list_diff(a, b):
    """Entfernt alle Elemente aus Liste a, die in Liste b enthalten sind."""
    return [x for x in a if x not in b]

results = []  # Store num_elements and L2 error pairs

for h in [2**-i for i in range(0,3)]:
    ngmesh = Mesh(unit_cube.GenerateMesh(maxh=h))
    T =int(1/h)
    mesh = UnstructuredHypertriangleMesh(T,ngmesh)
    fes = P1_Hypertriangle_Space(mesh)
    
    plt.savefig('mesh.png', dpi=150, bbox_inches='tight')
    plt.close()


    epsi = 1.e-5
    eps = GlobalFunction(lambda x: epsi, mesh = mesh)
    w = ConstantVectorFunction(np.array([1,1,1,1]), mesh = mesh)

    blf = BilinearForm(fes)
    blf += LaplaceIntegral_without_time(eps)
    blf += ConvectionIntegral(w)
    blf += SUPGIntegral(w,epsi)
    blf.assemble()

    
    f = lambda x: -(1 - exp((x[0]+x[1]+x[2] - 3)/epsi)) / (1 - exp(-1/epsi))* exp(-x[3])
    f = GlobalFunction(f, mesh = mesh)
    lf = LinearForm(fes)
    lf += SourceIntegral(f)
    lf += SUPGSourceIntegral(f, w,epsi)
    lf.assemble()

    top = mesh.top_bndry_vertices
    initial = mesh.initial_bndry_vertices
    bndry = list_diff(mesh.bndry_vertices, top)
    freedofs = list_diff(mesh.vertices,bndry)

    uex = lambda x: (1 - exp((x[0]+x[1]+x[2] - 3)/epsi)) / (1 - exp(-1/epsi)) * exp(-x[3])
    uh = FEFunction(fes)
    uh._set(uex, bndry)


    res = lf.vector - blf.matrix.dot(uh.vector)
    uh.vector += solve_on_freedofs(blf.matrix, res, freedofs)
    from methodsnm.forms import compute_difference_L2
    u_exact = GlobalFunction(uex, mesh = mesh)
    l2diff = compute_difference_L2(uh, u_exact, mesh, intorder = 5)
    print(" h: ", h, " L2-Difference: ", l2diff)
    results.append({ 'h': h, 'L2 Error': l2diff})

    

# Create and save results table
df = pd.DataFrame(results)
print("\n" + "="*60)
print("Convergence Study Results (varying mesh size h)")
print("="*60)
print(df.to_string(index=False))
print("="*60 + "\n")

# Calculate convergence rates
print("Convergence Rates:")
print("-"*60)
l2_errors = df['L2 Error'].values
hs = df['h'].values

convergence_rates = []
for i in range(1, len(l2_errors)):
    rate = np.log(l2_errors[i] / l2_errors[i-1]) / np.log(hs[i] / hs[i-1])
    convergence_rates.append(rate)
    print(f"Between h={hs[i-1]:.4f} and h={hs[i]:.4f}: rate = {rate:.4f}")

avg_rate = np.mean(convergence_rates)
print("-"*60)
print(f"Average convergence rate: {avg_rate:.4f}")
print("="*60 + "\n")

# Save table to CSV
df.to_csv('convergence_results.csv', index=False)
print("Results saved to convergence_results.csv")
print("Results saved to convergence_results.csv")
