from abc import ABC, abstractmethod
import numpy as np
from numpy import array
from netgen.csg import CSGeometry, OrthoBrick
from ngsolve import VOL, BND, Mesh as NGSMesh
from methodsnm.mesh import Mesh
from methodsnm.trafo import TesseraktTransformation , HypertriangleTransformation

class Mesh4D(Mesh):
    """Base class for 4D meshes.

    Subclasses should populate connectivity arrays such as
    ``points``, ``vertices``, ``hypercells`` and boundary lists.
    """
    def __init__(self):
        self.dimension = 4

class StructuredTesseraktMesh(Mesh4D):
    """Structured tesseract mesh on a regular (M x N x K x L) grid.

    The constructor builds vertex coordinates, hypercells (4D
    hypercubes), and lists of edges/faces/volumes. The mapping
    argument may be used to apply a geometric embedding.
    """
    def __init__(self, M, N , K, L, mapping = None):
        super().__init__()

        def check_bndryedges(bndry_vertices, edge):
            if set(edge).issubset(set(bndry_vertices)):
                return True
            else:
                return False
            
        def check_bndryfaces(bndry_vertices, face):
            if set(face).issubset(set(bndry_vertices)):
                return True
            else:
                return False
        
        def check_bndryvolumes(bndry_vertices, volume):
            if set(volume).issubset(set(bndry_vertices)):
                return True
            else:
                return False

        def rekursiv_bndry(lis):
            if len(lis) == 1:
                return [0,lis[0]]
            else:
                produkt1 = np.math.prod([x + 1 for x in lis])
                first = lis.pop(0)
                produkt = np.math.prod([x + 1 for x in lis])
                bndry_vertices = rekursiv_bndry(lis)
                #print(liste)
                liste = []
                liste += bndry_vertices
                liste += [i + j*produkt for i in bndry_vertices for j in range(first)]
                liste += [produkt1-1-i for i in range(produkt)]
                liste += [i for i in range(produkt)]
                return list(set(liste))
            
        
        if mapping is None:
            mapping = lambda x,y,z,t: [x,y,z,t]

        self.points = np.array([array(mapping(i/M,j/N,k/K,l/L)) for l in range(L+1) for k in range(K+1) for j in range(N+1) for i in range(M+1) ])
        
        self.vertices = np.arange((M+1)*(N+1)*(K+1)*(L+1))
                              
        self.hypercells = np.array([[    l*(M+1)*(K+1)*(N+1)+    k*(M+1)*(N+1)+    j*(M+1)+i,     l*(M+1)*(K+1)*(N+1)+    k*(M+1)*(N+1)+    j*(M+1)+i+1,
                                         l*(M+1)*(K+1)*(N+1)+    k*(M+1)*(N+1)+(j+1)*(M+1)+i,     l*(M+1)*(K+1)*(N+1)+    k*(M+1)*(N+1)+(j+1)*(M+1)+i+1,
                                         l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+    j*(M+1)+i,     l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+    j*(M+1)+i+1,
                                         l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+(j+1)*(M+1)+i,     l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+(j+1)*(M+1)+i+1,
                                     (l+1)*(M+1)*(K+1)*(N+1)+    k*(M+1)*(N+1)+    j*(M+1)+i, (l+1)*(M+1)*(K+1)*(N+1)+    k*(M+1)*(N+1)+    j*(M+1)+i+1,
                                     (l+1)*(M+1)*(K+1)*(N+1)+    k*(M+1)*(N+1)+(j+1)*(M+1)+i, (l+1)*(M+1)*(K+1)*(N+1)+    k*(M+1)*(N+1)+(j+1)*(M+1)+i+1,
                                     (l+1)*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+    j*(M+1)+i, (l+1)*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+    j*(M+1)+i+1,
                                     (l+1)*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+(j+1)*(M+1)+i, (l+1)*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+(j+1)*(M+1)+i+1] for i in range(M) for j in range(N) for k in range(K) for l in range(L)], dtype=int)
        
        self.volumes = np.array([[l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i+1,
                                     l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+(j+1)*(M+1)+i,l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+(j+1)*(M+1)+i+1,
                                     l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i+1,
                                     l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+(j+1)*(M+1)+i,l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+(j+1)*(M+1)+i+1]for i in range(M) for j in range(N) for k in range(K)   for l in range(L+1)]
                                     +[[l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i+1,
                                     l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+(j+1)*(M+1)+i,l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+(j+1)*(M+1)+i+1,
                                     (l+1)*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,(l+1)*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i+1,
                                     (l+1)*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+(j+1)*(M+1)+i,(l+1)*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+(j+1)*(M+1)+i+1]for i in range(M) for j in range(N) for k in range(K+1)   for l in range(L)]
                                     +[[l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i+1,
                                        l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i+1,
                                        (l+1)*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,(l+1)*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i+1,
                                        (l+1)*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i,(l+1)*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i+1]for i in range(M) for j in range(N+1) for k in range(K)   for l in range(L)]
                                        +[[l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+(j+1)*(M+1)+i,
                                           l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+(j+1)*(M+1)+i,
                                           (l+1)*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,(l+1)*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+(j+1)*(M+1)+i,
                                           (l+1)*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i,(l+1)*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+(j+1)*(M+1)+i]for i in range(M+1) for j in range(N) for k in range(K)   for l in range(L)], dtype=int)
        
        self.faces = np.array([[l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i+1,
                                     l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+(j+1)*(M+1)+i,l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+(j+1)*(M+1)+i+1]for i in range(M) for j in range(N) for k in range(K+1)   for l in range(L+1)]
                                +[[l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i+1,
                                        l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i+1]for i in range(M) for j in range(N+1) for k in range(K)   for l in range(L+1)]
                                +[[l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+(j+1)*(M+1)+i,
                                           l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+(j+1)*(M+1)+i]for i in range(M+1) for j in range(N) for k in range(K)   for l in range(L+1)]
                                +[[l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i+1, 
                                   (l+1)*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,(l+1)*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i+1]for i in range(M) for j in range(N+1) for k in range(K+1)   for l in range(L)]
                                +[[l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+(j+1)*(M+1)+i,
                                   (l+1)*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,(l+1)*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i] for i in range(M+1) for j in range(N) for k in range(K+1)   for l in range(L)]    
                                +[[l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i,
                                   (l+1)*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i,(l+1)*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i] for i in range(M+1) for j in range(N+1) for k in range(K)   for l in range(L)], dtype=int)
        
        self.edges = np.array([[l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i+1]for i in range(M) for j in range(N+1) for k in range(K+1)   for l in range(L+1)]
                              +[[l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+(j+1)*(M+1)+i]   for i in range(M+1) for j in range(N) for k in range(K+1)   for l in range(L+1)]
                              +[[l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,l*(M+1)*(K+1)*(N+1)+(k+1)*(M+1)*(N+1)+j*(M+1)+i]   for i in range(M+1) for j in range(N+1) for k in range(K)   for l in range(L+1)]
                              +[[l*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i,(l+1)*(M+1)*(K+1)*(N+1)+k*(M+1)*(N+1)+j*(M+1)+i] for i in range(M+1) for j in range(N+1) for k in range(K+1)   for l in range(L)], dtype=int)

        self.bndry_vertices = np.array(rekursiv_bndry([L,K,N,M]))
        self.bndry_edges = [i for i in self.edges if check_bndryedges(self.bndry_vertices,i) ]
        self.bndry_faces = [i for i in self.faces if check_bndryfaces(self.bndry_vertices,i) ]
        self.bndry_volumes = [i for i in self.volumes if check_bndryvolumes(self.bndry_vertices,i) ]
        self.xlength = M
        self.ylength = N    
        self.zlength = K
        self.tlength = L
        self.special_meshsize = 1/min(M,N,K,L)
        self.ne = len(self.hypercells)
         

    def filter_bndry_points(self ,extreme_type, index):
        """Return boundary vertices whose coordinate ``index`` is min/max.

        Parameters
        ----------
        extreme_type : {'min','max'}
        index : int
            Coordinate index (0..3) to inspect.
        """
        points = self.points[self.bndry_vertices]
    
        if extreme_type == "min":
            target_value = min(p[index] for p in points)
        elif extreme_type == "max":
            target_value = max(p[index] for p in points)
        else:
            raise ValueError("extreme_type must be either 'min' or 'max'")
        
        return [i for i in self.bndry_vertices if self.points[i][index] == target_value]

    def trafo(self, elnr, codim=0, bndry=False):
        """Return the element transformation for a tesseract element.

        Only full-volume (codim=0) transformations are supported here.
        """
        if codim > 0 or bndry:
            raise NotImplementedError("Not implemented yet")
        return TesseraktTransformation(self, elnr)


class UnstructuredHypertriangleMesh(Mesh4D):
    """Unstructured 4D mesh built from extruding a 3D NGSolve mesh in time.

    The mesh is constructed by stacking 3D tetrahedral meshes along a
    temporal axis and connecting corresponding vertices to form 4D
    simplices.
    """
    def generate_3d_mesh(maxh=0.2):
        # Create a simple 3D geometry (unit cube)
        geo = CSGeometry()
        cube = OrthoBrick(*[(0, 0, 0), (1, 1, 1)]).bc("outer")
        geo.Add(cube)

        # Generate 3D mesh
        ngmesh = geo.GenerateMesh(maxh=maxh)
        type(ngmesh)
        ngmesh.Curve(1)  # order = 1 (linear)
        
        return NGSMeshMesh(ngmesh)

    def __init__(self, T, ngmesh=None ):
        """Build an unstructured 4D hypertriangle mesh by temporal extrusion.

        Parameters
        ----------
        T : int
            Number of time intervals (levels) to extrude the 3D mesh.
        ngmesh : optional
            Precomputed 3D NGSolve mesh; if None a unit-cube mesh is created.
        """
        if ngmesh is None:
            ngmesh = self.generate_3d_mesh()
        super().__init__()
        self.dimension = 4
        nv = ngmesh.nv
        self.vertices = np.arange((T+1)*nv)
        self.hypercells = np.array([[t*nv+el.vertices[0].nr, t*nv+el.vertices[1].nr, t*nv+el.vertices[2].nr, t*nv+el.vertices[3].nr, (t+1)*nv+el.vertices[3].nr] for el in ngmesh.Elements(VOL) for t in range(T)]
                                   +[[t*nv+el.vertices[0].nr, t*nv+el.vertices[1].nr, t*nv+el.vertices[2].nr, (t+1)*nv+el.vertices[2].nr, (t+1)*nv+el.vertices[3].nr] for el in ngmesh.Elements(VOL) for t in range(T)]
                                   +[[t*nv+el.vertices[0].nr, t*nv+el.vertices[1].nr, (t+1)*nv+el.vertices[1].nr, (t+1)*nv+el.vertices[2].nr, (t+1)*nv+el.vertices[3].nr] for el in ngmesh.Elements(VOL) for t in range(T)]
                                   +[[t*nv+el.vertices[0].nr, (t+1)*nv+el.vertices[0].nr, (t+1)*nv+el.vertices[1].nr, (t+1)*nv+el.vertices[2].nr, (t+1)*nv+el.vertices[3].nr] for el in ngmesh.Elements(VOL) for t in range(T)]
                                   , dtype=int)
        
        vertex_indices = [None] *((T+1)*nv +1)

        for el in ngmesh.Elements(VOL):
            for v in el.vertices:
                    p = ngmesh[v].point
                    nr = v.nr
                    if vertex_indices[nr] is None:
                        vertex_indices[nr] = (p[0], p[1], p[2])

        self.points = np.array([np.append(p, t/T)for t in range(T+1)for p in vertex_indices if p is not None])

        boundary_vertices = list(dict.fromkeys(v.nr for el in ngmesh.Elements(BND) for v in el.vertices))
        boundary_vertices = list(dict.fromkeys(boundary_vertices))
        main_list = [t*nv+v for t in range(T) for v in boundary_vertices]
        first = list(range(nv))
        last = self.vertices[-nv:].tolist()
        self.bndry_vertices = np.array(list(set(main_list + first + last)))
        self.special_meshsize = 1/T
        self.initial_bndry_vertices = first
        self.top_bndry_vertices = last
        self._build_edges()
        self._build_p2_dofs()
        self.ne = len(self.hypercells)

    def trafo(self, elnr, codim=0, bndry=False):
        """Return the hypertriangle transformation for a simplex element.

        Only full-volume (codim=0) transformations are supported.
        """
        if codim > 0 or bndry:
            raise NotImplementedError("Not implemented yet")
        return HypertriangleTransformation(self, elnr)
    
    def _build_edges(self):
        """
        Build all unique edges across all 4D simplices.
        Each simplex has 5 vertices → 10 edges.
        """
        # concise docstring above; implementation unchanged
        edge_set = set()
        self.hypercell2edge =[]

        for cell in self.hypercells:
            v0, v1, v2, v3, v4 = cell
            edges_local = [
                (v0, v1), (v0, v2), (v0, v3), (v0, v4),
                (v1, v2), (v1, v3), (v1, v4),
                (v2, v3), (v2, v4),
                (v3, v4),
            ]
            self.hypercell2edge.append(edges_local)
            for e in edges_local:
                edge_set.add(tuple(sorted(e)))

        self.edges = np.array(sorted(edge_set), dtype=int)   
        self.edge_to_index = {tuple(edge): i for i, edge in enumerate(self.edges)}
    
    def index_of_edge(self, a):
        """
        Gives the corresponding number to the edge
        """
        if isinstance(a, (list, tuple)) and isinstance(a[0], (list, tuple)):
            lis = []
            for p in a:
                key = tuple(sorted(p))  # LIST → SORT → TUPLE
                lis.append(self.edge_to_index[key])
            return lis

        key = tuple(sorted(a))
        return self.edge_to_index[key]



    def _build_p2_dofs(self):
        """
        For each hypercell, create the list of local DOFs:
        5 vertex DOFs + 10 edge DOFs.
        """
        # concise docstring above; implementation unchanged
        nv = len(self.points)  # number of vertex DOFs
        edge_to_dof = {tuple(e): nv + i for i, e in enumerate(self.edges)}
        
        p2_dofs = []
        for cell in self.hypercells:
            v0, v1, v2, v3, v4 = cell
            edgelist = [
                (v0, v1), (v0, v2), (v0, v3), (v0, v4),
                (v1, v2), (v1, v3), (v1, v4),
                (v2, v3), (v2, v4),
                (v3, v4),
            ]
            
            local = list(cell)
            local += [edge_to_dof[tuple(sorted(e))] for e in edgelist]
            p2_dofs.append(local)

        self.p2_dofs = np.array(p2_dofs, dtype=int)

    
    def find_element(self, ip):
        def simplex4_volume(pts):
            """
            Computes 4D simplex volume = |det([p1-p0, p2-p0, p3-p0, p4-p0])| / 24
            """
            a = pts[0]
            M = np.column_stack([pts[i] - a for i in range(1,5)])
            return abs(np.linalg.det(M)) / 24.0


        def check_point_in_simplex_4d(points, ip, tol=1e-12):
            """Check whether point ip lies inside the 4D simplex given by points.

            Uses a barycentric-volume test (replace each vertex by the
            point and compare sub-simplex volumes to the whole volume).
            """
            # Gesamtvolumen
            V = simplex4_volume(points)
            if V < tol:
                return False

            # Baryzentrische Koordinaten über Ersatz eines Eckpunkts
            lambdas = []
            for i in range(5):
                pts_i = points.copy()
                pts_i[i] = ip   # ersetze Ecke i durch den Punkt
                Vi = simplex4_volume(pts_i)
                lambdas.append(Vi / V)

            lambdas = np.array(lambdas)

            # Prüfe baryzentrische Bedingungen
            return np.all(lambdas >= -tol) and abs(np.sum(lambdas) - 1) < tol
        
        for el in range(len(self.elements())):
            pts = self.points[self.elements()[el]]  # shape (5,4)
            if check_point_in_simplex_4d(pts, ip):
                return el
        raise Exception("Point outside mesh")