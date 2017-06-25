# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 17:43:56 2017

@author: rochus

                 pylmps

   this is a second and simple wrapper for lammps (not using Pylammps .. makes problem in MPI parallel)
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

from lammps import lammps


evars = {
         "vdW"     : "evdwl",
         "Coulomb" : "ecoul",
         "CoulPBC" : "elong",
         "bond"    : "ebond",
         "angle"   : "eangle",
         "oop"     : "eimp",
         "torsions": "edihed",
         "epot"    : "pe",
         }
enames = ["vdW", "Coulomb", "CoulPBC", "bond", "angle", "oop", "torsions"]
forces = ["fx", "fy", "fz"]

class pylmps:
    
    def __init__(self, name):
        self.name = name
        # TBI
        self.control = {}
        self.control["kspace"] = False
        return
        
    def setup(self,xyz=None, local=True, web=True):
        self.lmps = lammps()
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
        self.ff2lmp.write_input(filename=self.inp_file, kspace=self.control["kspace"])
        self.lmps.file(self.inp_file)
        # connect variables for extracting
        for e in evars:
            self.lmps.command("variable %s equal %s" % (e, evars[e]))
        self.natoms = self.lmps.get_natoms()      
        # compute energy of initial config
        self.calc_energy()
        self.report_energies()
        return

    def get_eterm(self, name):
        assert name in evars
        return self.lmps.extract_variable(name,None,0)
        
    def calc_energy(self):
        self.lmps.command("run 0")
        energy = self.get_eterm("epot")
        return energy
        
#   def calc_energy_force(self):
#        energy = self.calc_energy()
#        fxyz = self.get_force()
#        return energy, fxyz
        
    def get_energy_contribs(self):
        e = {}
        for en in enames:
            e[en] = self.get_eterm(en)
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
        fxyz = np.ctypeslib.as_array(self.lmps.gather_atoms("f",1,3))
        fxyz.shape=(self.natoms,3)
        return fxyz
        
    def get_xyz(self):
        """
        get the xyz position as a numpy array
        """
        xyz = np.ctypeslib.as_array(self.lmps.gather_atoms("x",1,3))
        fyz.shape=(self.natoms,3)
        return xyz
       

#    def calc_numforce(self,delta=0.0001):
#        """
#        compute numerical force and compare with analytic 
#        for debugging
#        """
#        cxyz = np.zeros([3], "d")
#        energy, fxyz = self.calc_energy_force()
#        num_fxyz = np.zeros([self.natoms,3],"float64")
#        # atypes = self.get_atomtypes()
#        for a in xrange(self.natoms):
#            cxyz[:] = self.lmps.atoms[a].position[:]
#            for i in xrange(3):
#                cxyz[i] += delta
#                self.lmps.atoms[a].position = cxyz
#                ep = self.calc_energy()
#                ep_contrib = np.array(self.get_energy_contribs().values())
#                cxyz[i] -= 2*delta
#                self.lmps.atoms[a].position = cxyz
#                em = self.calc_energy()
#                em_contrib = np.array(self.get_energy_contribs().values())
#                cxyz[i] += delta
#                self.lmps.atoms[a].position = cxyz
#                num_fxyz[a,i] = -(ep-em)/(2.0*delta)
#                # self.pprint("ep em delta_e:  %20.15f %20.15f %20.15f " % (ep, em, ep-em))
#                print(np.array2string((em_contrib-ep_contrib)/(2.0*delta),precision=2,suppress_small=True))
#                print("atom %d (%s) %d: anal: %12.6f num: %12.6f diff: %12.6f " % (a," ",i,fxyz[a,i],num_fxyz[a,i],( fxyz[a,i]-num_fxyz[a,i])))
#        return fxyz, num_fxyz
        
        
    def end(self):
        # clean up TBI
        return


