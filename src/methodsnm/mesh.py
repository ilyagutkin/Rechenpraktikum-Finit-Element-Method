from abc import ABC, abstractmethod
import numpy as np
from numpy import array

class Mesh(ABC):
    dimension = None
    points = None
    vertices = None
    edges = None
    faces = None
    volumes = None
    hypercells = None

    hypercells2volumes = None
    volumes2faces = None
    face2edges = None

    bndry_vertices = None
    bndry_edges = None
    bndry_faces = None
    bndry_volumes = None
    
    def __init__(self):
        raise NotImplementedError("Not implemented")

    def elements(self, codim=0, bndry=False):
        if self.dimension - codim == 0:
            if bndry:
                return Exception("Invalid dimension")
            else:
                return self.vertices
        elif self.dimension - codim == 1:   
            if bndry:
                return [[self.vertices[i]] for i in self.bndry_vertices]
            else:
                return self.edges
        elif self.dimension - codim == 2:   
            if bndry:
                return self.edges[self.bndry_edges]
            else:
                return self.faces
        elif self.dimension - codim == 3:   
            if bndry:
                return self.faces[self.bndry_faces]
            else:
                return self.volumes
        elif self.dimension - codim == 4:
            if bndry:
                return self.volumes[self.bndry_volumes]
            else:
                return self.hypercells    
        
        else:
            raise Exception("Invalid dimension")

    def trafo(self, elnr, codim=0, bndry=False):
        raise NotImplementedError("Not implemented")

