from abc import ABC, abstractmethod
import numpy as np
from numpy import array
from methodsnm.trafo import *
from methodsnm.mesh import *
from methodsnm.fe_vector import *
from scipy import sparse

class Form(ABC):
    integrals = None
    fes = None
    def __init__(self):
        raise NotImplementedError("Not implemented")

    def add_integrator(self, integrator):
        self.integrals.append(integrator)

    def __iadd__(self, integrator):
        self.add_integrator(integrator)
        return self

class LinearForm(Form):

    vector = None
    def __init__(self, fes=None):
        self.fes = fes
        self.integrals = []
    
    def assemble(self):
        self.vector = np.zeros(self.fes.ndof)
        mesh = self.fes.mesh
        for elnr, verts in enumerate(mesh.elements()):
            trafo = mesh.trafo(elnr)
            fe = self.fes.finite_element(elnr)
            dofs = self.fes.element_dofs(elnr)
            for integral in self.integrals:
                self.vector[dofs] += integral.compute_element_vector(fe, trafo)


class BilinearForm(Form):

    matrix = None
    def __init__(self, fes=None, fes_test=None, fes_trial=None):
        if fes is None and (fes_test is None or fes_trial is None) or all([f is not None for f in [fes,fes_test,fes_trial]]):
            raise Exception("Invalid arguments, specify either `fes` or `fes_test` and `fes_trial`")
        if fes_test is None:
            fes_test = fes
            fes_trial = fes

        self.fes_trial = fes_trial
        self.fes_test = fes_test
        self.fes = fes_test
        self.integrals = []
    
    def assemble(self):
        self.matrix = sparse.lil_matrix((self.fes_test.ndof, self.fes_trial.ndof))
        mesh = self.fes.mesh
        for elnr, verts in enumerate(mesh.elements()):
            trafo = mesh.trafo(elnr)
            fe_test = self.fes_test.finite_element(elnr)
            dofs_test = self.fes_test.element_dofs(elnr)
            fe_trial = self.fes_trial.finite_element(elnr)
            dofs_trial = self.fes_trial.element_dofs(elnr)
            elmat = np.zeros((len(dofs_test), len(dofs_trial)))
            for integral in self.integrals:
                elmat += integral.compute_element_matrix(fe_test, fe_trial, trafo)
            for i, dofi in enumerate(dofs_test):
                for j, dofj in enumerate(dofs_trial):
                    self.matrix[dofi,dofj] += elmat[i,j]
        self.matrix = self.matrix.tocsr()

class LinearVectorForm(Form):
    """
    Blockstruktur-Linearform für ProductSpaces.
    Verwaltet Integratoren nach Komponente (nur b_test).
    """

    def __init__(self, fes):
        self.fes = fes
        self.fes_test = fes
        nb = len(fes.spaces)
        self.integrators = [[] for _ in range(nb)]
        self.vector = None

    def add_block_integrator(self, b_test, integrator):
        """
        Registriert einen Integrator für die Blockkomponente b_test.
        """
        self.integrators[b_test].append(integrator)

    def assemble(self):
        """
        Versammelt globale RHS unter Verwendung der Blockstruktur.
        """

        ndof = self.fes.ndof
        self.vector = np.zeros(ndof)

        mesh = self.fes.mesh
        nb = len(self.fes.spaces)

        # Schleife über Elemente
        for elnr in range(len(mesh.elements())):
            trafo = mesh.trafo(elnr)

            fe_block = self.fes.finite_element(elnr)
            dofs_block = self.fes.element_dofs(elnr)

            # Lokaler RHS-Vektor
            elloc = np.zeros(fe_block.ndof)

            # Blockweiser Aufbau
            for b_test in range(nb):

                if not self.integrators[b_test]:
                    continue

                # Offsets
                ot = fe_block.block_offsets[b_test]
                nt = fe_block.block_ndofs[b_test]

                # Das FE der getesteten Komponente
                fe_test = self.fes.spaces[b_test].finite_element(elnr)

                # Summiere die Integratoren
                local_block = np.zeros(nt)
                for integ in self.integrators[b_test]:
                    local_block += integ.compute_element_vector(fe_test, trafo)

                elloc[ot : ot + nt] += local_block

            # Streuung in globalen RHS-Vektor
            for i, di in enumerate(dofs_block):
                self.vector[di] += elloc[i]


class BilinearVectorForm(Form):
    """
    Blockstruktur-Bilinearform für ProductSpaces.
    Verwaltet Integratoren blockweise (b_test, b_trial).
    """

    def __init__(self, fes):
        self.fes = fes
        self.fes_test  = fes
        self.fes_trial = fes

        # Liste blockweiser Integratoren:
        nb = len(fes.spaces)
        self.integrators = [[[] for _ in range(nb)] for _ in range(nb)]
        self.matrix = None
    
    def add_block_integrator(self, b_test, b_trial, integrator):
        """
        Registriert einen Integrator für den Block (b_test, b_trial).
        """
        self.integrators[b_test][b_trial].append(integrator)
    
    def assemble(self):
        """
        Assemble der globalen Matrix entsprechend der Blockstruktur.
        """

        ndof = self.fes.ndof
        self.matrix = sparse.lil_matrix((ndof, ndof))
        mesh = self.fes.mesh
        nb = len(self.fes.spaces)

        for elnr in range(len(mesh.elements())):
            trafo = mesh.trafo(elnr)
            fe_block = self.fes.finite_element(elnr)
            dofs_block = self.fes.element_dofs(elnr)
            elmat = np.zeros((fe_block.ndof, fe_block.ndof))

            for b_test in range(nb):
                for b_trial in range(nb):
                    if not self.integrators[b_test][b_trial]:
                        continue

                    ot = fe_block.block_offsets[b_test]
                    nt = fe_block.block_ndofs[b_test]
                    or_ = fe_block.block_offsets[b_trial]
                    nr = fe_block.block_ndofs[b_trial]

                    fe_test  = self.fes.spaces[b_test].finite_element(elnr)
                    fe_trial = self.fes.spaces[b_trial].finite_element(elnr)
                    
                    block_local = np.zeros((nt, nr))
                    for integ in self.integrators[b_test][b_trial]:
                        block_local += integ.compute_element_matrix(fe_test, fe_trial, trafo)
                    elmat[ot:ot+nt, or_:or_+nr] += block_local

            for i, di in enumerate(dofs_block):
                for j, dj in enumerate(dofs_block):
                    self.matrix[di, dj] += elmat[i, j]

        self.matrix = self.matrix.tocsr()

from methodsnm.intrule import select_integration_rule
from numpy.linalg import det
def compute_difference_L2(uh, uex, mesh, intorder=5):
    sumint = 0
    for elnr in range(len(mesh.elements())):
        trafo = mesh.trafo(elnr)
        intrule = select_integration_rule(intorder, trafo.eltype)
        uhvals = uh.evaluate(intrule.nodes, trafo)
        uexvals = uex.evaluate(intrule.nodes, trafo)
        diff = np.zeros(len(intrule.nodes))
        diff = (uhvals - uexvals)**2
        F = trafo.jacobian(intrule.nodes)
        w = array([abs(det(F[i,:,:])) * intrule.weights[i] for i in range(F.shape[0])])
        sumint += np.dot(w, diff)
    return np.sqrt(sumint)

def analyze_stability(uh, uex, wind_coeff, mesh, intorder=5):
    """
    Berechnet L2-Fehler UND den Fehler im Streamline-Gradienten.
    Letzterer zeigt uns, ob die Methode Oszillationen (Wiggles) hat.
    """
    sum_l2 = 0
    sum_streamline = 0
    
    for elnr in range(len(mesh.elements())):
        trafo = mesh.trafo(elnr)
        intrule = select_integration_rule(intorder, trafo.eltype)
        
        # 1. Werte evaluieren
        uh_vals = uh.evaluate(intrule.nodes, trafo)
        uex_vals = uex.evaluate(intrule.nodes, trafo)
        
        # 2. Gradienten evaluieren (für Wiggle-Check entscheidend)
        # Wir brauchen grad(uh - uex) in Welt-Koordinaten
        duh_phys = uh.evaluate(intrule.nodes, trafo, deriv=True)
        duex_phys = uex.evaluate(intrule.nodes, trafo, deriv=True)
        diff_grad = duh_phys - duex_phys # Shape: (n_points, dim)
        
        # 3. Wind evaluieren
        w_vec = wind_coeff.evaluate(intrule.nodes, trafo) # (n_points, dim)
        
        # 4. Streamline-Fehler: |w · grad(uh - uex)|^2
        # Das misst Oszillationen entlang der Strömung
        streamline_diff = np.einsum("nd,nd->n", w_vec, diff_grad)**2
        
        # 5. Geometrie
        F = trafo.jacobian(intrule.nodes)
        detF = np.array([abs(np.linalg.det(F[i,:,:])) for i in range(F.shape[0])])
        quad_weight = detF * intrule.weights
        
        # Integrale aufsummieren
        sum_l2 += np.dot(quad_weight, (uh_vals - uex_vals)**2)
        sum_streamline += np.dot(quad_weight, streamline_diff)
        
    return {
        "L2_Error": np.sqrt(sum_l2),
        "Streamline_Grad_Error": np.sqrt(sum_streamline)
    }

import numpy as np

def compute_difference_SUPG(uh, uex, mesh, wind_coeff, eps, intorder=5):
    """
    Berechnet den Fehler in der Streamline-Seminorm: 
    sqrt( integral( (w * grad(uh - uex))^2 ) )
    """
    sumint = 0
    h_num = 1e-6  # Schrittweite für numerische Ableitung
    dim = mesh.dimension
    
    for elnr in range(len(mesh.elements())):
        trafo = mesh.trafo(elnr)
        intrule = select_integration_rule(intorder, trafo.eltype)
        ips_ref = intrule.nodes
        nq = len(ips_ref)
        
        # --- 1. Gradient der DISKRETEN Lösung uh (analytisch) ---
        fe = uh.fes.finite_element(trafo.elnr)
        dofs = uh.fes.element_dofs(trafo.elnr)
        
        # dphi_ref shape: (nq, dim, ndofs) oder (nq, ndofs, dim)
        dphi_ref = fe.evaluate(ips_ref, deriv=True) 
        uh_local_coeffs = uh.vector[dofs]
        
        # Kontraktion: Wir brauchen (nq, dim)
        # Falls dphi_ref (nq, dim, ndofs) ist:
        duh_ref = np.einsum("qdj, j -> qd", dphi_ref, uh_local_coeffs)
        
        # Transformation in Welt-Raum
        F = trafo.jacobian(ips_ref)
        invF = np.array([np.linalg.inv(F[q].T) for q in range(nq)])
        duh_phys = np.einsum("qij, qj -> qi", invF, duh_ref)

        # --- 2. Gradient der EXAKTEN Lösung uex (numerisch & sicher) ---
        phys_pts = trafo(ips_ref) 
        duex_phys = np.zeros((nq, dim))
        
        for q in range(nq):
            point = phys_pts[q]
            for d in range(dim):
                shift = np.zeros(dim)
                shift[d] = h_num
                
                # Expliziter Einzelpunkt-Aufruf um Broadcast-Fehler zu vermeiden
                f_plus = uex.f(point + shift)
                f_minus = uex.f(point - shift)
                
                # Sicherstellen, dass wir Skalare bekommen
                val_p = f_plus[0] if isinstance(f_plus, (np.ndarray, list)) else f_plus
                val_m = f_minus[0] if isinstance(f_minus, (np.ndarray, list)) else f_minus
                
                duex_phys[q, d] = (val_p - val_m) / (2 * h_num)

        # --- 3. Integration ---
        w_val = wind_coeff.evaluate(ips_ref, trafo)
        diff_grad = duh_phys - duex_phys
        
        # Richtungsableitung: w · grad(e)
        res = np.einsum("nd, nd -> n", w_val, diff_grad)
        
        # Determinante für das Integral
        adetF = np.array([abs(np.linalg.det(F[q])) for q in range(nq)])
        
        sumint += np.dot(adetF * intrule.weights, res**2)
        
    return np.sqrt(sumint)