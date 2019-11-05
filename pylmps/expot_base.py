# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 17:26:02 CEST 2019

@author: rochus

                 expot _base

   base calss to derive external potentials for use with pylmps

   derive your class from this base class and overload the calc_energy_force and setup methods

   You can either set all params during the instantiation or within setup where you have access to the 
   mol object and its ff addon etc.
   Note that parameter fitting with ff_gen only works if you define params from the mol.ff object. 

"""

import numpy as np
from molsys import mpiobject
from lammps import lammps
from molsys.util.units import kcalmol, electronvolt


class expot_base(mpiobject):

    def __init__(self, mpi_comm=None, out = None):
        super(expot_base, self).__init__(mpi_comm,out)
        self.name = "base"
        self.is_expot = True
        return

    def setup(self, pl):
        # keep access to pylmps object
        self.pl = pl
        # allocate arrays
        self.natoms = self.pl.get_natoms()
        self.energy = 0.0
        self.force  = np.zeros([self.natoms, 3], dtype="float64")
        return

    def calc_numforce(self, delta=0.001):
        #TBI
        return
    
    def calc_energy_force(self):
        """main class to be rewritten
        """
        # in base we do nothing ... in principle self.xyz exists
        return self.energy, self.force

    def callback(self, lmps, vflag):
        """The callback function called by lammps .. should not be changed
        """
        lmps = lammps(ptr=lmps)
        # get the current atom positions
        self.xyz = np.ctypeslib.as_array(lmps.gather_atoms("x",1,3))
        # get current cell
        self.cell = self.pl.get_cell()
        self.xyz.shape=(self.natoms,3)
        # calculate energy and force
        self.calc_energy_force()
        # distribute the forces back
        lmps.scatter_atoms("f", 2, 3, np.ctypeslib.as_ctypes(self.force))
        # return the energy
        return self.energy



class expot_ase(expot_base):

    def __init__(self, atoms, idx):
        super(expot_ase, self).__init__()
        self.atoms = atoms
        self.idx = idx
        return

    def setup(self,pl):
        super(expot_ase, self).setup(pl)
        assert len(self.idx) <= self.natoms
        for i in self.idx:
            assert i < self.natoms
        self.pprint("An ASE external potential was added!")
        return

    def calc_energy_force(self):
        # we have to set the actual coordinates and cell to ASE
        self.atoms.set_cell(self.cell)
        self.atoms.set_positions(self.xyz[self.idx])
        # by default ase uses eV and A as units
        # consequently units has to be changed here to kcal/mol
        self.energy = self.atoms.get_potential_energy()*electronvolt/kcalmol
        self.force[self.idx] = self.atoms.get_forces()*electronvolt/kcalmol
        return self.energy, self.force


"""
   As an illustration this is a derived external potential that just fixes one interatomic distance by a harmonic potential
  
"""

class expot_test(expot_base):

    def __init__(self, a1, a2, r0, k):
        """a test external potential
        
        Args:
            a1 (int): atom1
            a2 (int): atom2
            r0 (float): refdist
            k (float): force constant
        """
        super(expot_test, self).__init__()
        self.name = "test"
        self.a1 = a1
        self.a2 = a2
        self.r0 = r0
        self.k  = k
        return

    def setup(self, pl):
        super(expot_test, self).setup(pl)
        # check if a1 and a2 are in range
        assert self.a1 < self.natoms
        assert self.a2 < self.natoms
        self.pprint("a test external potential between atoms %d (%s) and %d (%s)" % (self.a1, self.pl.mol.elems[self.a1], self.a2, self.pl.mol.elems[self.a2]))
        return

    def calc_energy_force(self):
        d = self.xyz[self.a1]-self.xyz[self.a2]
        r = np.sqrt(np.sum(d*d))
        rr = r-self.r0
        self.energy = 0.5*self.k*rr*rr
        dE = self.k*rr
        self.force[self.a1] = -d/r*dE
        self.force[self.a2] = +d/r*dE
        return self.energy, self.force


