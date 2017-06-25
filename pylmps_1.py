# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 17:43:56 2017

@author: rochus

                 pylmps

   this is a first and simple wrapper for PyLammps (which is wrapper for the python module lammps which is a wrapper for lammps)
   in the style of pydlpoly ... 

"""
from __future__ import print_function

import numpy as np
import string
from mpi4py import MPI

import molsys
import molsys.util.ff2lammps as ff2lammps

mpi_rank = MPI.COMM_WORLD.Get_rank()
mpi_size = MPI.COMM_WORLD.Get_size()

# overload print function in parallel case
import __builtin__
def print(*args, **kwargs):
    if mpi_rank == 0:
        return __builtin__.print(*args, **kwargs)
    else:
        return

from lammps import PyLammps


evars = {
         "vdW"     : "evdwl",
         "Coulomb" : "ecoul",
         "bond"    : "ebond",
         "angle"   : "eangle",
         "oop"     : "eimp",
         "torsions": "edihed"}
enames = ["vdW", "Coulomb", "bond", "angle", "oop", "torsions"]

class pylmps:
    
    def __init__(self, name):
        self.name = name
        return
        
    def setup(self,xyz=None, local=True, web=True, kspace=False):
        self.lmps = PyLammps()
        self.mol = molsys.mol()
        if xyz == None:
            xyz = self.name + ".mfpx"
        self.mol.read(xyz)
        self.ff2lmp = ff2lammps.ff2lammps(self.mol)
        if local:
            self.data_file = self.name+".data"
            self.inp_file  = self.name+".in"
        else:
            raise ValueError, "to be implemented"
        self.ff2lmp.write_data(filename=self.data_file)
        self.ff2lmp.write_input(filename=self.inp_file, kspace=kspace)
        self.lmps.file(self.inp_file)
        # compute energy of initial config
        self.calc_energy()
        self.report_energies()
        # some settings
        self.natoms = self.lmps.system.natoms        
        return
        
    def calc_energy(self):
        self.lmps.run(0)
        pe = self.lmps.eval("pe")
        return pe
        
    def calc_energy_force(self):
        energy = self.calc_energy()
        fxyz = self.get_force()
        return energy, fxyz
        
    def get_energy_contribs(self):
        e = {}
        for en in enames:
            e[en] = self.lmps.eval(evars[en])
        return e
        
    def report_energies(self):
        e = self.get_energy_contribs()
        etot = 0.0
        for en in enames:
            etot += e[en]
            print("%15s : %15.8f kcal/mol" % (en, e[en]))
        print("%15s : %15.8f kcal/mol" % ("TOTAL", etot))
        return
        
    def get_force(self):
        """
        get the actual force as a numpy array
        """
        fxyz = np.empty([self.natoms, 3],"d")
        for i in xrange(self.natoms):
            fxyz[i,:] = self.lmps.atoms[i].force
        return fxyz
        
    def get_xyz(self):
        """
        get the xyz position as a numpy array
        """
        xyz = np.empty([self.natoms, 3],"d")
        for i in xrange(self.natoms):
            xyz[i,:] = self.lmps.atoms[i].position
        return xyz
       
    def get_subset_xyz(self, aind):
        """
        get the xyz position of a subset defiend by the list aind
        """
        xyz = np.empty([len(aind), 3],"d")
        for i, ai in enumerate(aind):
            xyz[i,:] = self.lmps.atoms[ai].position
        return xyz

    def calc_numforce(self,delta=0.0001):
        """
        compute numerical force and compare with analytic 
        for debugging
        """
        cxyz = np.zeros([3], "d")
        energy, fxyz = self.calc_energy_force()
        num_fxyz = np.zeros([self.natoms,3],"float64")
        # atypes = self.get_atomtypes()
        for a in xrange(self.natoms):
            cxyz[:] = self.lmps.atoms[a].position[:]
            for i in xrange(3):
                cxyz[i] += delta
                self.lmps.atoms[a].position = cxyz
                ep = self.calc_energy()
                ep_contrib = np.array(self.get_energy_contribs().values())
                cxyz[i] -= 2*delta
                self.lmps.atoms[a].position = cxyz
                em = self.calc_energy()
                em_contrib = np.array(self.get_energy_contribs().values())
                cxyz[i] += delta
                self.lmps.atoms[a].position = cxyz
                num_fxyz[a,i] = -(ep-em)/(2.0*delta)
                # self.pprint("ep em delta_e:  %20.15f %20.15f %20.15f " % (ep, em, ep-em))
                print(np.array2string((em_contrib-ep_contrib)/(2.0*delta),precision=2,suppress_small=True))
                print("atom %d (%s) %d: anal: %12.6f num: %12.6f diff: %12.6f " % (a," ",i,fxyz[a,i],num_fxyz[a,i],( fxyz[a,i]-num_fxyz[a,i])))
        return fxyz, num_fxyz
        
        
    def end(self):
        # clean up TBI
        return


