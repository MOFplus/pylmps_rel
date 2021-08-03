
#
# import ctypes for interface to xTB
#
import ctypes
from ctypes import c_int, c_double, c_bool

#
# import interface to xTB
#
try:
    import xtb
    from xtb.interface import Calculator, Param
    from xtb.libxtb import VERBOSITY_FULL, VERBOSITY_MINIMAL, VERBOSITY_MUTED 

except ImportError:
    print("ImportError: Impossible to load xTB")

import numpy as np
import molsys
import sys

from molsys.util import pdlpio2
import molsys.util.elems as elems

from molsys.util.constants import bohr


#
# Class definition for xtb calcaulator
#
class xtb_calc:


   def __init__( self
               , mol
               , gfn: int = 0
               , charge: float = 0.0
               , pbc: bool = False
               , maxiter: int = 250
               , etemp: float = 300.0 
               , accuracy: float = 0.01
               , uhf: int = 0
               , verbose = 0
               , write_pdlp_file = False
               , write_frequency = 100
               , pdlpfile = None
               , restart=None
               , stage = None
               ):

      #
      # Sanity check(s)
      #
      if gfn not in [-1,0,1,2]:
        raise NotImplementedError("Currently only gfn-FF, gfn0-xTB, gfn1-xTB and gfn2-xTP supported")

      if gfn == 2 and pbc == True:
        raise NotImplementedError("Currently PBC only supported for gfn0-xTB and gfn1-xTB")


      parameter_set = {  0 : Param.GFN0xTB
                      ,  1 : Param.GFN1xTB
                      ,  2 : Param.GFN2xTB
                      , -1 : Param.GFNFF
                      }
      #
      # Assign class attributes
      #
      self.gfn = gfn
      self.mol = mol
      self.pbc = pbc
      self.charge = charge
      self.param = parameter_set[gfn] 
      self.etemp = etemp
      self.maxiter = maxiter
      self.accuracy = accuracy
      self.uhf = uhf
      self.verbose = verbose

      # set up things for integration in MD driver
      self.nbondsmax = None
      self.write_pdlp_file = write_pdlp_file
      self.write_frequency = write_frequency
      self.write_counter = 1
      self.startup = True
      self.stage = stage
      if write_pdlp_file:
          self.pdlp = pdlpio2.pdlpio2(pdlpfile, ffe=self, restart=restart)
          # Get upper bound for bonds 
          if self.nbondsmax == None:
              self.nbondsmax = 0
              for e in self.get_elements():
                  self.nbondsmax += elems.maxbond[e]
              self.nbondsmax /= 2
          self.nbondsmax = int(self.nbondsmax)
      else:
          self.pdlp = None
      # set up things that do not change during the calcultion
      self.numbers   = np.array(self.mol.get_elems_number(), dtype=c_int)
      self.calc = None
      return

   def get_natoms(self):
      return self.mol.get_natoms()

   def get_elements(self):
      return self.mol.get_elems()

   def get_xyz(self):
      return self.mol.get_xyz()

   def get_cell(self):
      return self.mol.get_cell()

   def calculate(self, xyz, cell):
      positions = np.array(xyz/bohr, dtype=c_double)
      if self.pbc == True:  
         cell      = np.array(cell/bohr, dtype=c_double)
         pbc       = np.full(3, True, dtype=c_bool)
      else:
         cell = None
         pbc       = np.full(3, False, dtype=c_bool)

      if self.calc is None:
         # make calc when it does not exist (first interation)
         self.calc = Calculator(self.param, self.numbers, positions, self.charge, uhf=self.uhf, lattice=cell, periodic=pbc)
         self.calc.set_verbosity(self.verbose)
         #
         # Set user options
         #
         self.calc.set_electronic_temperature(self.etemp)
         self.calc.set_max_iterations(self.maxiter)
         self.calc.set_accuracy(self.accuracy)
      else:
         # update calc (molecule obejct under the hood)
         self.calc.update(positions, lattice=cell)
      
      res = self.calc.singlepoint()

      results = { 'energy'    : res.get_energy()
                , 'gradient'  : res.get_gradient()
                , 'bondorder' : res.get_bond_orders()
                }

      if self.write_pdlp_file:
         self.write_frame(results)

      return results


   def write_frame(self,results):

       if self.write_counter == self.write_frequency or self.startup:
           self.write_counter = 1  # Set back to zero
           self.startup = False
           self.pdlp.open()
           st = self.pdlp.h5file[self.stage]
           traj = st["traj"]
           bond_order = results['bondorder']
           # Setup bondtab
           bond_tab = np.zeros((self.nbondsmax,2),dtype=int)
           bond_ord = np.zeros((self.nbondsmax))
           bothres = 0.5
           natoms = self.get_natoms()
           nbnd = 0
           for iat in range(natoms):
              for jat in range(0,iat+1): 
                 if bond_order[iat][jat] > bothres:
                    bond_tab[nbnd][0] = iat
                    bond_tab[nbnd][1] = jat
                    bond_ord[nbnd] = bond_order[iat][jat]
                    nbnd += 1
           # Add bondtab to pdlp file
           if "bondord" in traj:
              traj_bondord = traj["bondord"]
              traj_bondord.resize((traj_bondord.shape[0] + 1),axis=0)
              traj_bondord[-1:] = bond_ord
              traj_bondtab = traj["bondtab"]
              traj_bondtab.resize((traj_bondtab.shape[0] + 1),axis=0)
              traj_bondtab[-1:] = bond_tab
           else:
              # entry does not exist. Create it
              self.pdlp.add_bondtab(self.stage, self.nbondsmax, bond_tab, bond_ord)
           # Add trajectory info
           xyz_data = self.mol.get_xyz()
           #
           # add xyz data
           #
           if "xyz" in traj:
              trj_xyz = traj["xyz"]
              trj_xyz.resize((trj_xyz.shape[0] + 1),axis = 0)
              trj_xyz[-1:] = xyz_data 
           else:
              xyzshape = (1,) + xyz_data.shape
              trj_xyz = traj.require_dataset("xyz",shape=xyzshape, dtype=xyz_data.dtype, maxshape=( (None,) + xyz_data.shape), chunks=xyzshape)
              trj_xyz[...] = self.mol.get_xyz()
           velocities = False # TODO
           if velocities:
               trj_vel = traj.require_dataset("vel", shape=vel.shape, dtype=vel.dtype)
               #trj_vel[...] = vel #TODO
           self.pdlp.close() 
       else:
           self.write_counter += 1
