# -*- coding: utf-8 -*-

import numpy as np

# helper function
def rotate_cell(cell):
    if np.linalg.norm(cell[0]) != cell[0,0]:
        # system needs to be rotated
        A = cell[0]
        B = cell[1]
        C = cell[2]
        AcB = np.cross(A,B)
        uAcB = AcB/np.linalg.norm(AcB)
        lA = np.linalg.norm(A)
        uA = A/lA
        lx = lA
        xy = np.dot(B,uA)
        ly = np.linalg.norm(np.cross(uA,B))
        xz = np.dot(C,uA)
        yz = np.dot(C,np.cross(uAcB,uA))
        lz = np.dot(C,uAcB)
        cell = np.array([
                [lx,0,0],
                [xy,ly,0.0],
                [xz,yz,lz]])
    return cell
 

