from OpenGL.GL import *
from OpenGL.GLU import *

"""
Basic GlShape class
"""
class GlShape:

    def __init__(self, vertices=None, edges=None):
        self._vertices: list = vertices
        self._edges: list = edges 

    @property 
    def vertices(self): return self._vertices.copy()

    @property 
    def edges(self): return self._edges.copy()

    def render(self):
        glBegin(GL_LINES)
        for edge in self._edges:
            for vertex in edge:
                glVertex3fv(self._vertices[vertex])
        glEnd()

"""
A simple cube
"""
class GlCube(GlShape):
    def __init__(self):
        super().__init__(
            vertices=[
                    (1, -1, -1),
                    (1, 1, -1),
                    (-1, 1, -1),
                    (-1, -1, -1),
                    (1, -1, 1),
                    (1, 1, 1),
                    (-1, -1, 1),
                    (-1, 1, 1)
                    ],
            edges=[
                    (0,1),
                    (0,3),
                    (0,4),
                    (2,1),
                    (2,3),
                    (2,7),
                    (6,3),
                    (6,4),
                    (6,7),
                    (5,1),
                    (5,4),
                    (5,7)
                ]
        )