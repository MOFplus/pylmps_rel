
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
    from xtb.interface import XTBLibrary     # needed to access GFN0Calculation
except ImportError:
    print("ImportError: Impossible to load xTB")

import numpy as np
import molsys

from molsys.util.units import bohr

#
# Class definition for xtb calcaulator
#
class xtb_calc:


   def __init__( self
               , mol
               , gfn: int = 0
               , charge: float = 0.0
               , pbc: bool = False
               , options: dict = None
               ):

      xtb0_default_options = {
       'print_level': 0,
       'parallel': 0,
       'accuracy': 1.0,
       'electronic_temperature': 300.0,
       'gradient': True,
       'ccm': True,
       'solvent': 'none',
      }

      xtb1_default_options = {
        'print_level': 2,
        'parallel': 0,
        'accuracy': 1.0,
        'electronic_temperature': 300.0,
        'gradient': True,
        'restart': True,
        'max_iterations': 250,
        'solvent': 'none',
        'ccm': True,
      }

      xtb2_default_options = {
        'print_level': 2,
        'parallel': 0,
        'accuracy': 1.0,
        'electronic_temperature': 300.0,
        'gradient': True,
        'restart': True,
        'max_iterations': 250,
        'solvent': 'none',
        'ccm': True,
      }


      default_options = { 0 : xtb0_default_options
                        , 1 : xtb1_default_options
                        , 2 : xtb2_default_options
                        }
 
      #
      # Sanity check(s)
      #
      if gfn not in [0,1,2]:
        raise NotImplementedError("Currently only gfn0-xTB, gfn1-xTB and gfn2-xTP supported")

      if gfn != 0 and pbs == True:
        raise NotImplementedError("Currently PBC only supported for gfn0-xTB")

      #
      # Assign class attributes
      #
      self.gfn = gfn
      self.mol = mol
      self.pbc = pbc
      self.charge = charge
      self.lib = XTBLibrary()
      if options is None:
         self.options = default_options[gfn] 
      else:
         self.options = options  

   def calculate(self):

      # create arguments for xtb interface
      kwargs = {
          'natoms': self.mol.get_natoms(),
          'numbers': np.array(self.mol.get_elems_number(), dtype=c_int),
          'charge': self.charge,
          'magnetic_moment': 0,
          'positions': np.array(self.mol.get_xyz()/bohr, dtype=c_double),  
          'cell': np.array(self.mol.get_cell()/bohr, dtype=c_double),
          'pbc': np.full(3, self.pbc, dtype=c_bool),          
          'options': self.options,
          'output': "-",
      }

      # Remove options which do not fit for gfn1 or gfn2
      if self.gfn != 0:
         del kwargs['pbc']
         del kwargs['cell']

      if self.gfn == 0:
         results = self.lib.GFN0Calculation(**kwargs)
      elif self.gfn == 1:
         results = self.lib.GFN1Calculation(**kwargs)
      elif self.gfn == 2:
         results = self.lib.GFN2Calculation(**kwargs)
      else:
         raise NotImplementedError("Parameterset not supported")

      return results

