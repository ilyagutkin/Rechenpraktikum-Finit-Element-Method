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

def filter_bndry_dofs(edge_points_dict, extreme_type, coord_index):
    """
    Filtert die DOF-Indizes basierend auf extremalen Punkten.
    Gibt die Indizes zurück, die übrig bleiben.
    """
    if not edge_points_dict:
        return []
    
    points_list = list(edge_points_dict.values())
    
    if extreme_type == "min":
        target_value = min(p[coord_index] for p in points_list)
    elif extreme_type == "max":
        target_value = max(p[coord_index] for p in points_list)
    else:
        raise ValueError("extreme_type must be either 'min' or 'max'")
    
    # Gib nur die Indizes zurück, deren Punkte das Kriterium erfüllen
    return [dof_idx for dof_idx, point in edge_points_dict.items() 
            if point[coord_index] == target_value]

def mittelpunkt(p1,p2):
    return (p1[0]+p2[0])/2, (p1[1]+p2[1])/2

def list_diff(a, b):
    """Entfernt alle Elemente aus Liste a, die in Liste b enthalten sind."""
    return [x for x in a if x not in b]

results = []  # Store num_elements and L2 error pairs

for num_elements in [64,128,264]:
    mesh = StructuredRectangleMesh(num_elements, num_elements)
    #DrawMesh2D(mesh)
    # Calculate h (mesh size)
    h = 1.0 / num_elements
    
    plt.savefig('mesh.png', dpi=150, bbox_inches='tight')
    plt.close()

    from methodsnm.fes import *
    fes = P2_Triangle_Space(mesh)

    epsi = 0.1
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


    edge_points_dict = {}

    for v in mesh.bndry_edges:
        ed = mesh.edges[v]
        dof_index = v + len(mesh.vertices)
        point = mittelpunkt(mesh.points[ed[0]], mesh.points[ed[1]])
        edge_points_dict[dof_index] = point
        
    top_edge =  filter_bndry_dofs(edge_points_dict, "max", 1)
    bottom_edge = filter_bndry_dofs(edge_points_dict, "min", 1)
    left_edge = filter_bndry_dofs(edge_points_dict, "min", 0)
    right_edge = filter_bndry_dofs(edge_points_dict, "max", 0)
    edge_bndry =  bottom_edge + left_edge + right_edge

    top = mesh.filter_bndry_points("max", 1)
    bottom = mesh.filter_bndry_points("min", 1)
    left = mesh.filter_bndry_points("min", 0)
    right = mesh.filter_bndry_points("max", 0)
    bndry = left + right + bottom + edge_bndry
    
    freedofs = list_diff(fes.dof, bndry)
    uex = lambda x: (1 - exp((x[0] - 1)/epsi)) / (1 - exp(-1/epsi))* exp(-x[1])
    uh = FEFunction(fes)
    uh._set_P2(uex, bndry)
    

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
print(f"Convergence Study Results (varying mesh size h), epsilon = {epsi}, finite element order = {fes.fe.order}")
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