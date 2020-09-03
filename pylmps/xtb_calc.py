
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

#from molsys.util.units import bohr
bohr = 0.529177249  # remove this when bohr is in molysy.util.units

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
               ):

      #
      # Sanity check(s)
      #
      if gfn not in [0,1,2]:
        raise NotImplementedError("Currently only gfn0-xTB, gfn1-xTB and gfn2-xTP supported")

      if gfn == 2 and pbc == True:
        raise NotImplementedError("Currently PBC only supported for gfn0-xTB and gfn1-xTB")


      parameter_set = {  0 : Param.GFN0xTB
                      ,  1 : Param.GFN1xTB
                      ,  2 : Param.GFN2xTB
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

   def calculate(self):

      numbers   = np.array(self.mol.get_elems_number(), dtype=c_int) 
      positions = np.array(self.mol.get_xyz()/bohr, dtype=c_double)
      if self.pbc == True:  
         cell      = np.array(self.mol.get_cell()/bohr, dtype=c_double)
         pbc       = np.full(3, True, dtype=c_bool)
      else:
         cell = None
         pbc       = np.full(3, False, dtype=c_bool)

      calc = Calculator(self.param, numbers, positions, self.charge, uhf=self.uhf, lattice=cell, periodic=pbc)

      calc.set_verbosity(self.verbose)

      #
      # Set user options
      #
      calc.set_electronic_temperature(self.etemp)
      calc.set_max_iterations(self.maxiter)
      calc.set_accuracy(self.accuracy)

      res = calc.singlepoint()

      results = { 'energy'   : res.get_energy()
                , 'gradient' : res.get_gradient()
                }

      return results

