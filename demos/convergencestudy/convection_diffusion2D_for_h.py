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

results = []  # Store num_elements and L2 error pairs

for num_elements in [2, 4, 8, 16, 32]:
    mesh = StructuredRectangleMesh(num_elements, num_elements)
    #DrawMesh2D(mesh)
    # Calculate h (mesh size)
    h = 1.0 / num_elements
    
    plt.savefig('mesh.png', dpi=150, bbox_inches='tight')
    plt.close()

    from methodsnm.fes import *
    fes = P1_Triangle_Space(mesh)

    epsi = 1
    wind = 1
    eps = GlobalFunction(lambda x: epsi, mesh = mesh)
    w = ConstantVectorFunction(np.array([wind, 1]), mesh = mesh)

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

    top = mesh.filter_bndry_points("max", 1)
    bottom = mesh.filter_bndry_points("min", 1)
    left = mesh.filter_bndry_points("min", 0)
    right = mesh.filter_bndry_points("max", 0)
    bndry = left + right + bottom
    freedofs = list_diff(mesh.vertices, bndry)
    
    uex = lambda x: (1 - exp((x[0] - 1)/epsi)) / (1 - exp(-1/epsi))* exp(-x[1])
    uh = FEFunction(fes)
    uh._set(uex, bndry)

    res = lf.vector - blf.matrix.dot(uh.vector)
    uh.vector += solve_on_freedofs(blf.matrix, res, freedofs)
    from methodsnm.forms import compute_difference_L2
    u_exact = GlobalFunction(uex, mesh = mesh)
    l2diff = compute_difference_L2(uh, u_exact, mesh, intorder = 5)
    print("Elements: ", num_elements, " h: ", h, " L2-Difference: ", l2diff)
    results.append({'Elements': num_elements, 'h': h, 'L2 Error': l2diff})
    #DrawFunction2D(uh)
    # Save the solution plot
    fig, ax = plt.subplots()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/solution_elements_{num_elements}.png', dpi=150, bbox_inches='tight')
    plt.close()

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
