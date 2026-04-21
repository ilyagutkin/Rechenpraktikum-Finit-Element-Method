from import_hack import *
from methodsnm.mesh_2d import *
from methodsnm.visualize import *
from methodsnm.solver import solve_on_freedofs
from methodsnm.forms import *
from methodsnm.formint import *
import numpy as np
from numpy import pi, sin, cos, exp
import matplotlib.pyplot as plt
import pandas as pd
import os

def list_diff(a, b):
    """Entfernt alle Elemente aus Liste a, die in Liste b enthalten sind."""
    return [x for x in a if x not in b]
mesh = StructuredRectangleMesh(20, 20)
DrawMesh2D(mesh)
plt.savefig('mesh.png', dpi=150, bbox_inches='tight')
plt.close()

from methodsnm.fes import *
fes = P1_Triangle_Space(mesh)

epsilon = [0.0001,0.001,0.01,0.1,1]
results = []  # Store epsilon and L2 error pairs

for epsi in epsilon:
    wind =1
    eps = GlobalFunction(lambda x: epsi, mesh = mesh)
    w = ConstantVectorFunction(np.array([wind,1]), mesh = mesh)

    blf = BilinearForm(fes)
    blf += LaplaceIntegral_without_time(eps)
    blf += ConvectionIntegral(w)
    blf += SUPGIntegral(w)
    blf.assemble()

    f = lambda x: -(1 - exp((x[0] - 1)/epsi)) / (1 - exp(-1/epsi))* exp(-x[1])
    f = GlobalFunction(f, mesh = mesh)
    lf = LinearForm(fes)
    lf += SourceIntegral(f)
    lf += SUPGSourceIntegral(f, w)
    lf.assemble()

    top = mesh.filter_bndry_points("max",1)
    bottom = mesh.filter_bndry_points("min",1)
    left = mesh.filter_bndry_points("min",0)
    right = mesh.filter_bndry_points("max",0)
    bndry = left + right + bottom
    freedofs = list_diff(mesh.vertices,bndry)

    uex = lambda x: (1 - exp((x[0] - 1)/epsi)) / (1 - exp(-1/epsi))* exp(-x[1])
    uh = FEFunction(fes)
    uh._set(uex, bndry)

    res = lf.vector - blf.matrix.dot(uh.vector)
    uh.vector += solve_on_freedofs(blf.matrix,res,freedofs)
    from methodsnm.forms import compute_difference_L2
    u_exact = GlobalFunction(uex, mesh = mesh)
    l2diff = compute_difference_L2(uh, u_exact, mesh, intorder = 5)
    print("Epsilon: ", epsi, " L2-Difference: ", l2diff)
    results.append({'Epsilon': epsi, 'L2 Error': l2diff})
    
    # Save the solution plot
    fig, ax = plt.subplots()
    DrawFunction2D(uh)
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/solution_eps_{epsi}.png', dpi=150, bbox_inches='tight')
    plt.close()

# Create and save results table
df = pd.DataFrame(results)
print("\n" + "="*50)
print("Convergence Study Results")
print("="*50)
print(df.to_string(index=False))
print("="*50 + "\n")

# Save table to CSV
df.to_csv('convergence_results.csv', index=False)
print("Results saved to convergence_results.csv")
