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
from molsys.util.constants import kcalmol, electronvolt, bohr
import time

from .xtb_calc import xtb_calc
try:
    from ase.calculators.turbomole import execute
except ImportError:
    print("ImportError: Impossible to load ASE")

class expot_base(mpiobject):

    def __init__(self, mpi_comm=None, out = None):
        super(expot_base, self).__init__(mpi_comm,out)
        self.name = "base"
        self.is_expot = True
        self.expot_time = 0.0
        return

    def setup(self, pl):
        # keep access to pylmps object
        self.pl = pl
        # allocate arrays
        self.natoms = self.pl.get_natoms()
        self.energy = 0.0
        self.force  = np.zeros([self.natoms, 3], dtype="float64")
        self.step = 0
        return
    
    def calc_energy_force(self):
        """main class to be rewritten
        """
        # in base we do nothing ... in principle self.xyz exists
        return self.energy, self.force

    def callback(self, lmps, vflag):
        """The callback function called by lammps .. should not be changed
        """
        tstart = time.time()
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
        self.step += 1
        self.expot_time += time.time() - tstart 
        return self.energy

    def test_deriv(self, delta=0.0001, verbose=True):
        """test the analytic forces by numerical differentiation

        make shure to call the energy and force once before you call this function
        to make shure all is set up and self.xyz exisits and is initalized

        Args:
            delta (float, optional): shift of coords. Defaults to 0.0001.
            verbose (boolen, optiona): print while we go. Defaults to True

        """
        xyz_keep = self.xyz.copy()
        force_analytik = self.force.copy()
        force_numeric  = np.zeros(force_analytik.shape)
        for i in range(self.natoms):
            for j in range(3):
                self.xyz[i,j] += delta
                ep, f = self.calc_energy_force()
                self.xyz[i,j] -= 2.0*delta
                em, f = self.calc_energy_force()
                self.xyz[i,j] = xyz_keep[i,j]
                force_numeric[i,j] = -(ep-em)/(2.0*delta)
                if verbose:
                    print ("atom %5d dim %2d : analytik %10.5f numeric %10.5f (delta %8.2f)" % (i, j, force_analytik[i,j], force_numeric[i,j], force_analytik[i,j]-force_numeric[i,j]))
        return force_numeric


class expot_ase(expot_base):

    def __init__(self, atoms, idx):
        super(expot_ase, self).__init__()
        self.atoms = atoms
        self.idx = idx
        self.name = "ase"
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

class expot_ase_turbomole(expot_base):

    def __init__(self, atoms, idx):
        super(expot_ase_turbomole, self).__init__()
        self.atoms = atoms
        self.idx = idx
        self.name = "ase"
        return

    def setup(self,pl):
        super(expot_ase_turbomole, self).setup(pl)
        assert len(self.idx) <= self.natoms
        for i in self.idx:
            assert i < self.natoms
        self.pprint("An ASE external potential was added!")
        # perform turbomole define once at the start
        self.atoms.calc.initialize()
        return

    def calc_energy_force(self):
        # we have to set the actual coordinates and cell to ASE
        self.atoms.set_cell(self.cell)
        self.atoms.set_positions(self.xyz[self.idx])
        self.atoms.calc.set_atoms(self.atoms)
        # get energies and forces using turbomole executables
        execute(self.atoms.calc.calculate_energy)
        self.atoms.calc.read_energy()
        execute(self.atoms.calc.calculate_forces)
        self.atoms.calc.read_forces()
        # by default ase uses eV and A as units
        # consequently units has to be changed here to kcal/mol
        self.energy = self.atoms.calc.e_total*electronvolt/kcalmol
        self.force = self.atoms.calc.forces.copy()*electronvolt/kcalmol
        return self.energy, self.force


class expot_xtb(expot_base):

    def __init__(self, mol, periodic=None, gfn_param=0, etemp=300.0, accuracy=1.0, uhf=0, verbose=0, maxiter=250
                ,write_mfp5_file=False,write_frequency=100,mfp5file=None,restart=None,stage=None):
        """constructor for xTB external potential
        
        This will construct an external potential based on the xtb program to be used inside LAMMPS       
 
        Args:
            mol (molsys object): molecular system 
            periodic (boolean, optional): Sets if we have a periodic system. If this argument is set to None periodicity is extracted from the mol object
            gfn_param (int, optional): GFN parameterization. Possible values (-1,0,1,2) -1 is the GFN-FF force field
            etemp (float, optional): Electronic temperature. Defaults to 300 K
            accuracy (float, optional): accuracy setting. Defaults to 1.0 (default in xtb)
            uhf (int, optional): Number of unpaired electrons
            verbose (int, optional): Print level. Possible values 0,1,2. Defaults to 0 (silent mode)
            maxiter (int, optional): Maximum number of iterations in the self consisting charge cycles. Defaults to 250
            write_mfp5_file (boolean, optional): If a mfp5 should be written? Defaults to False 
            write_frequency (int, optional): If write_mfp5_file = True it determine how often the coordinates should be dumped
            mfp5file (mfp5file object, optional): The mfp5 file handeler used
            restart (boolean, optional): Determines if this is a restart
            stage (string, optional): Determines the current stage we are in 

        """
        super(expot_xtb, self).__init__()
        self.mol = mol
        self.gfn_param = gfn_param
        self.uhf = uhf
        self.etemp = etemp
        self.accuracy = accuracy
        self.verbose = verbose
        self.maxiter = maxiter     
        if periodic = None:
            self.periodic = mol.periodic
        else:
            self.periodic = periodic
        self.name = "xtb"
        self.bond_order = None
        self.write_mfp5_file = write_mfp5_file
        self.write_frequency = write_frequency
        self.mfp5file = mfp5file
        self.restart = restart
        self.stage = stage
        return

    def setup(self,pl):
        super(expot_xtb, self).setup(pl)
        # create calculator and do gfn-xTB calculation
        self.gfn = xtb_calc( self.mol
                      , self.gfn_param
                      , pbc=self.periodic
                      , uhf=self.uhf
                      , accuracy=self.accuracy
                      , etemp=self.etemp
                      , verbose=self.verbose
                      , maxiter=self.maxiter
                      , write_mfp5_file=self.write_mfp5_file
                      , write_frequency=self.write_frequency
                      , mfp5file=self.mfp5file
                      , restart=self.restart
                      , stage=self.stage
                      )
        self.pprint("An xTB external potential was added")
        print("Periodic Boundary Conditions:") 
        print(self.periodic)
        return

    def calc_energy_force(self):
        #import sys
        #sys.stdout = open('xtb.out', 'w')
        results = self.gfn.calculate(self.xyz, self.cell)
        #
        # xTB uses a.u. as units so we need to convert
        #
        self.energy  = results['energy'] / kcalmol
        self.force   = -results['gradient'] / kcalmol / bohr
        self.bond_order = results['bondorder']
        return self.energy, self.force

    def get_bond_order(self):
        results = self.gfn.calculate(self.xyz, self.cell)
        self.bond_order = results['bondorder']
        return self.bond_order


    def set_stage(self,stage):
        self.gfn.set_stage(stage)
        return

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


