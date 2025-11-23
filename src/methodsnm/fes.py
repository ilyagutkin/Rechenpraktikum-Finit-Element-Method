from abc import ABC, abstractmethod
import numpy as np
from numpy import array
from methodsnm.fe_1d import *
from methodsnm.fe_2d import *
from methodsnm.fe_4d import *

class FESpace(ABC):
    """
    Abstract base class for finite element spaces.
    """
    ndof = None
    mesh = None
    def __init__(self, mesh):
        """Initialize base FESpace.

        Note: the base class does not set up DOFs or finite elements.
        Subclasses should override and set attributes like ``ndof``,
        ``mesh`` and element templates (e.g. ``fe``, ``sfe``) as needed.

        Parameters
        ----------
        mesh : object
            Mesh-like object providing geometry, connectivity and
            boundary information used by subclasses.
        """
        # intentionally minimal: subclasses set specific attributes
        self.mesh = mesh

    @abstractmethod
    def _finite_element(self, elnr):
        """Return the finite element object used on the given element.

        This is an abstract method that must be implemented by subclasses.

        Parameters
        ----------
        elnr : int
            Local element index (0-based) within the mesh.

        Returns
        -------
        object
            A finite-element descriptor (object implementing shape
            function evaluation used elsewhere in the codebase).
        """
        raise Exception("Not implemented - Base class should not be used")

    @abstractmethod
    def _element_dofs(self, elnr):
        """Return the global DOF indices associated with an element.

        Subclasses must provide an array-like of integer DOF indices
        that describe the mapping from local element DOFs to global
        system-wide DOF numbering.

        Parameters
        ----------
        elnr : int
            Local element index.

        Returns
        -------
        list or ndarray
            Integer indices of global DOFs for the element.
        """
        raise Exception("Not implemented - Base class should not be used")

    def _bndry_finite_element(self, elnr):
        """Return the 1D/face finite element used on a boundary element.

        Default implementation raises when boundary evaluation is
        unsupported. Subclasses that support boundary evaluation should
        override this method.

        Parameters
        ----------
        elnr : int
            Local boundary element index.
        """
        raise Exception("_bndry_finite_element not implemented - Base class should not be used")

    def _bndry_element_dofs(self, elnr):
        """Return global DOF indices for a boundary element.

        This method should mirror ``_element_dofs`` but for boundary
        elements (edges/faces on the mesh boundary). If boundary
        evaluation is not supported the default implementation raises.

        Parameters
        ----------
        elnr : int
            Local boundary element index.
        """
        raise Exception("_bndry_element_dofs not implemented - Base class should not be used")

    def finite_element(self, elnr, bndry=False):
        """
        Returns the finite element for the given element number.
        """
        if bndry:
            return self._bndry_finite_element(elnr)
        else:
            return self._finite_element(elnr)

    def element_dofs(self, elnr, bndry=False):
        """
        Returns the dofs for the given element number.
        """
        if bndry:
            return self._bndry_element_dofs(elnr)
        else:
            return self._element_dofs(elnr)
        
    def boundary_vertices(self):
        """Return the set of vertex indices that lie on the boundary.

        Returns
        -------
        set
            Set of integer vertex indices belonging to the mesh boundary.
        """
        return set(self.mesh.bndry_vertices)

    def initial_vertices(self):
        """Return the set of vertices on the initial/bottom boundary.

        For time-space meshes this represents the 'initial time'
        boundary. If the mesh does not provide this attribute an
        AttributeError will be raised by the caller.
        """
        return set(self.mesh.initial_bndry_vertices)

    def top_vertices(self):
        """Return the set of vertices on the top boundary (final time).

        Useful for applying time-dependent boundary conditions or for
        excluding top vertices from certain boundary constraint sets.
        """
        return set(self.mesh.top_bndry_vertices)
    
    def boundary_dofs_excluding_top(self):
        """Return boundary DOFs excluding those on the top boundary.

        This is a convenience for problems where the top boundary should
        remain free (e.g. time-marching where final-time conditions are
        handled separately).

        Returns
        -------
        list
            Sorted list of boundary DOF indices not on the top boundary.
        """
        v = self.boundary_vertices() - self.top_vertices()
        return sorted(self.boundary_dofs_from_vertex_set(v))

    def boundary_dofs_from_vertex_set(self, vset):
        """Map a set of vertex indices to boundary DOF indices.

        Default behavior assumes one DOF per vertex and returns a
        sorted list of the provided vertex indices. Subclasses that
        have additional DOFs (edge/face DOFs) may override this method
        to include those DOFs as well.

        Parameters
        ----------
        vset : iterable
            Iterable of integer vertex indices.

        Returns
        -------
        list
            Sorted list of integer DOF indices corresponding to the
            provided vertices.
        """
        return sorted(vset)

class VertexFirstSpace_1D(FESpace):
    def __init__(self, mesh):
        """Finite element space where DOFs are placed on vertices.

        This lightweight 1D helper sets a node finite element template
        used for boundary evaluations and stores the mesh reference.

        Parameters
        ----------
        mesh : object
            Mesh-like object providing points, edges and boundary data.
        """
        self.sfe = Node_FE()
        self.mesh = mesh

    def _bndry_element_dofs(self, bndry_elnr):
        """Return DOFs for a boundary element in a vertex-first space.

        For vertex-first spaces DOFs on a boundary element correspond
        directly to the mesh element vertex indices returned by
        ``mesh.elements(bndry=True)``.
        """
        return self.mesh.elements(bndry=True)[bndry_elnr]

    def _bndry_finite_element(self, bndry_elnr):
        return self.sfe

class P1_Segments_Space(VertexFirstSpace_1D):

    def __init__(self, mesh, periodic=False):
        super().__init__(mesh)
        self.periodic = periodic
        if periodic:
            self.ndof = len(mesh.points) - 1 
        else:
            self.ndof = len(mesh.points)
        self.fe = P1_Segment_FE()

    # note: no behavior change, just documentation added

    def _finite_element(self, elnr):
        """Return the segment finite element used for all elements.

        P1 on segments uses the same element template for every
        element, so ``elnr`` is unused.
        """
        return self.fe

    def _element_dofs(self, elnr):
        """Return global DOF indices for a segment element.

        For non-periodic meshes the DOFs are exactly the two vertex
        indices making up the edge. For periodic meshes the last edge
        shares the first global DOF (index 0).
        """
        dofs = self.mesh.edges[elnr]
        if self.periodic and elnr == len(self.mesh.edges) - 1:
            return [dofs[0],0]
        else:
            return dofs

    def _bndry_element_dofs(self, bndry_elnr):
        """Return DOFs for a boundary element on a 1D segment space.

        For periodic spaces the boundary is identified with the single
        shared DOF (index 0). Otherwise return the boundary element
        vertex indices as provided by the mesh.
        """
        dofs = self.mesh.elements(bndry=True)[bndry_elnr]
        if self.periodic:
            return [0]
        else:
            return dofs

class P1disc_Segments_Space(FESpace):

    def __init__(self, mesh):
        self.ndof = 2*len(mesh.edges)
        self.mesh = mesh
        self.fe = P1_Segment_FE()

    # discontinuous P1: two DOFs per edge, no shared vertex DOFs

    def _finite_element(self, elnr):
        """Return the (discontinuous) segment finite element template."""
        return self.fe

    def _element_dofs(self, elnr):
        """Return global DOFs for a discontinuous segment element.

        Each element has two local DOFs stored at indices 2*elnr and
        2*elnr+1 so the global numbering is contiguous per element.
        """
        return [2*elnr, 2*elnr+1]

    def _bndry_finite_element(self, bndry_elnr):
        """Discontinuous elements do not support boundary evaluation.

        Calling this method raises to make the limitation explicit.
        """
        raise Exception("No boundary evaluation for Discontinuous elements")        

    def _bndry_element_dofs(self, bndry_elnr):
        """Discontinuous elements do not provide boundary DOFs."""
        raise Exception("No boundary evaluation for Discontinuous elements")        

class Lagrange_Segments_Space(VertexFirstSpace_1D):

    def __init__(self, mesh, order=1):
        super().__init__(mesh)
        self.nv = len(mesh.points)
        self.ne = len(mesh.edges)
        self.order = order
        self.ndof = self.nv + self.ne*(order-1)
        self.fe=Lagrange_Segment_FE(order=self.order)

    # Lagrange space of arbitrary order on segments: vertex DOFs +
    # internal edge DOFs when order > 1

    def _finite_element(self, elnr):
        """Return the Lagrange segment finite element template."""
        return self.fe

    def _element_dofs(self, elnr):
        """Return global DOFs for a Lagrange segment element.

        DOF ordering: [left vertex, internal edge DOFs..., right vertex].
        """
        offset = self.nv + elnr*(self.order-1)
        dofs = [self.mesh.edges[elnr][0]] + [offset + i for i in range(0,self.order-1)] + [self.mesh.edges[elnr][1]]
        return dofs



class Pk_IntLeg_Segments_Space(VertexFirstSpace_1D):

    def __init__(self, mesh, order=1):
        super().__init__(mesh)
        self.nv = len(mesh.points)
        self.ne = len(mesh.edges)
        self.order = order
        self.ndof = self.nv + self.ne*(order-1)
        self.fe=IntegratedLegendre_Segment_FE(order=self.order)

    # Integrated-Legendre hierarchical basis on segments

    def _finite_element(self, elnr):
        """Return the integrated-Legendre segment FE template."""
        return self.fe

    def _element_dofs(self, elnr):
        """Return global DOFs for an integrated-Legendre segment.

        Note: ordering differs from Lagrange: here vertex DOFs are
        returned first, followed by internal (hierarchical) DOFs.
        """
        offset = self.nv + elnr*(self.order-1)
        dofs = [self.mesh.edges[elnr][0]] + [self.mesh.edges[elnr][1]] + [offset + i for i in range(0,self.order-1)]
        return dofs



class P1_Triangle_Space(FESpace):

    def __init__(self, mesh):
        self.ndof = len(mesh.points)
        self.mesh = mesh
        self.fe = P1_Triangle_FE()
        self.sfe = P1_Segment_FE()

    def _finite_element(self, elnr):
        """Return the P1 triangle finite element template."""
        return self.fe

    def _bndry_finite_element(self, bndry_elnr):
        """Return the linear segment FE used on triangle boundaries."""
        return self.sfe

    def _element_dofs(self, elnr):
        """Return vertex DOFs for a triangle element (P1).

        The returned array lists the three vertex indices in the mesh
        that form the given triangle element.
        """
        return self.mesh.elements()[elnr]

    def _bndry_element_dofs(self, bndry_elnr):
        """Return DOFs for a boundary segment of a triangle mesh.

        For P1 triangles boundary DOFs are simply the two vertices of
        the boundary edge.
        """
        return self.mesh.elements(bndry=True)[bndry_elnr]


class P2_Triangle_Space(FESpace):

    def __init__(self, mesh):
        self.ndof = len(mesh.points) + len(mesh.edges)
        self.nv = len(mesh.points)
        self.mesh = mesh
        self.fe = P2_Triangle_FE()
        self.sfe = Lagrange_Segment_FE(order=2)

    def _finite_element(self, elnr):
        """Return the quadratic triangle FE template (P2)."""
        return self.fe

    def _bndry_finite_element(self, bndry_elnr):
        """Return the quadratic segment FE used on triangle boundaries."""
        return self.sfe

    def _element_dofs(self, elnr):
        """Return global DOFs for a P2 triangle element.

        DOF ordering: vertex DOFs first (3 entries) followed by the
        mid-edge DOFs whose global indices are offset by ``nv``.
        """
        return np.append(self.mesh.elements()[elnr], [self.nv + i for i in self.mesh.faces2edges[elnr]])

    def _bndry_element_dofs(self, bndry_elnr):
        """Return DOFs for a boundary edge in a P2 triangle mesh.

        The ordering is: left vertex, mid-edge DOF, right vertex.
        """
        verts = self.mesh.elements(bndry=True)[bndry_elnr]
        edge = self.mesh.bndry_edges[bndry_elnr]
        return [verts[0], self.nv + edge, verts[1]]

class P3_Triangle_Space(FESpace):

    def __init__(self, mesh):
        self.ndof = len(mesh.points) + 2 * len(mesh.edges) + len(mesh.faces)
        self.nv = len(mesh.points)
        self.ned = len(mesh.edges)
        self.nf = len(mesh.faces)
        self.mesh = mesh
        self.fe = P3_Triangle_FE()
        self.sfe = Lagrange_Segment_FE(order=3)

    def _finite_element(self, elnr):
        """Return the cubic triangle FE template (P3)."""
        return self.fe

    def _element_dofs(self, elnr):
        """Return global DOFs for a P3 triangle element.

        Local DOF ordering (length 10):
        - indices 0..2 : vertex DOFs
        - indices 3..8 : two DOFs per edge (orientation-sensitive)
        - index 9      : cell interior DOF
        """
        dofs = np.empty(10, dtype=int)
        vnums = self.mesh.elements()[elnr]
        dofs[0:3] = vnums
        j = 3
        enums = self.mesh.faces2edges[elnr]
        ref_verts_list = [[1,2],[0,2],[0,1]]
        for ref_edge, edge in enumerate(enums):
            tverts = vnums[ref_verts_list[ref_edge]]
            if (tverts[0] < tverts[1]):
                dofs[j] = self.nv + 2*edge + 0
                j += 1
                dofs[j] = self.nv + 2*edge + 1
                j += 1
            else:
                dofs[j] = self.nv + 2*edge + 1
                j += 1
                dofs[j] = self.nv + 2*edge + 0
                j += 1
        dofs[9] = self.nv + 2 * self.ned + elnr
        return dofs

    def _bndry_finite_element(self, bndry_elnr):
        """Return the cubic segment FE used on triangle boundaries."""
        return self.sfe

    def _bndry_element_dofs(self, bndry_elnr):
        """Return DOFs for a boundary edge in a P3 triangle mesh.

        The ordering is: left vertex, two ordered edge DOFs, right vertex.
        """
        verts = self.mesh.elements(bndry=True)[bndry_elnr]
        edge = self.mesh.bndry_edges[bndry_elnr]
        return [verts[0], self.nv + 2 * edge, self.nv + 2 * edge + 1, verts[1]]


class P1Edge_Triangle_Space(FESpace):

    def __init__(self, mesh):
        self.ndof = len(mesh.edges)
        self.mesh = mesh
        self.fe = P1Edge_Triangle_FE()

    def _finite_element(self, elnr):
        """Return the finite element defined on edges of triangles."""
        return self.fe

    def _element_dofs(self, elnr):
        """Return global DOFs (edge indices) for a triangle element.

        This space uses one DOF per edge; the returned list contains the
        edge indices for the given triangular face.
        """
        edges = self.mesh.faces2edges[elnr]
        return edges
    
class P1_Tesserakt_Space(FESpace):
    """
    This class represents a P1 Tesserakt finite element space.
    """
    def __init__(self, mesh):
        self.ndof = len(mesh.points)
        self.mesh = mesh
        self.fe = P1_Tesserakt_FE()
        """P1 finite element space on a tesseract (4D hypercube).

        Parameters
        ----------
        mesh : object
            Mesh-like object holding points and element connectivity.
        """
        

    def _finite_element(self,elnr):
        """Return the P1 tesseract finite element template."""
        return self.fe

    def _element_dofs(self, elnr):
        return self.mesh.elements()[elnr]

class P1_Hypertriangle_Space(FESpace):
    """
    This class represents a P1 Hypertriangle finite element space.
    """
    def __init__(self, mesh):
        self.ndof = len(mesh.points)
        self.mesh = mesh
        self.fe = P1_Hypertriangle_FE()

    def _finite_element(self,elnr):
        """Return the P1 hypertriangle (simplex) finite element."""
        return self.fe

    def _element_dofs(self, elnr):
        """Return the vertex DOFs for a hypertriangle element (P1)."""
        return self.mesh.elements()[elnr]
    
    
class P2_Hypertriangle_Space(FESpace):
    """
    P2 space on 4D simplex mesh.
    DOFs: 1 per vertex + 1 per edge of each hyper-tetrahedron.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        self.nv = len(mesh.points)
        self.nedges = len(self.mesh.edges)
        self.ndof = self.nv + self.nedges 
        self.fe = P2_Hypertriangle_FE()

    
    def _finite_element(self, elnr):
        """Return the P2 hypertriangle finite element template."""
        return self.fe

    def _element_dofs(self, elnr):
        """
        Local DOF numbering:
        - first 5: vertex dofs (like P1)
        - last 10: edge dofs (global index shifted by nv)
        """
        verts = self.mesh.elements()[elnr]
        edge = self.mesh.hypercell2edge[elnr]
        edge_ids = self.mesh.index_of_edge(edge)
        edge_ids = [self.nv + i for i in edge_ids]
        return np.concatenate([verts, edge_ids])
    
    def boundary_dofs(self):
        """Return all boundary DOF indices for the P2 hypertriangle space.

        The boundary DOFs consist of vertex DOFs lying on the boundary
        plus any edge DOFs whose both vertices belong to the boundary
        vertex set.

        Returns
        -------
        list
            Sorted list of integer DOF indices on the boundary.
        """
        v = self.boundary_vertices()
        bndry_edge = self.mesh.boundary_edges.values()
        bnd_edge = []
        for i in list(bndry_edge):
            i += self.nv
            bnd_edge.append(i)

        return sorted(list(v) + bnd_edge)
    
    def boundary_dofs_from_vertex_set(self, vset):
        """Return boundary DOFs (vertices + edges) restricted to vset.

        Parameters
        ----------
        vset : iterable
            Iterable/collection of vertex indices considered to be on a
            boundary subset.

        Returns
        -------
        list
            Sorted list of corresponding vertex and edge DOF indices.
        """
        v = sorted(vset)
        e = self._edge_boundary_dofs(vset)
        return sorted(v + e)


