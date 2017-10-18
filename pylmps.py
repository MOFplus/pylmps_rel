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
pressure = ["pxx", "pyy", "pzz", "pxy", "pxz", "pyz"]

class pylmps:
    
    def __init__(self, name):
        self.name = name
        # TBI
        self.control = {}
        self.control["kspace"] = False
        return
        
    def setup(self, mfpx=None, local=True, mol=None, par=None, ff="MOF-FF"):
        self.lmps = lammps(cmdargs=["-log", "none"])
        # depending on what type of input is given a setup will be done
        # the default is to load an mfpx file and assign from MOF+ (using force field MOF-FF)
        # if par is given or ff="file" we use mfpx/ric/par
        # if mol is given then this is expected to be an already assigned mol object
        #      (in the latter case everything else is ignored!)
        if mol != None:
            self.mol = mol
        else:
            # we need to make a molsys and read it in
            self.mol = molsys.mol()
            if mfpx == None:
                mfpx = self.name + ".mfpx"
            self.mol.read(mfpx)
            self.mol.addon("ff")
            if par or ff=="file":
                if par == None:
                    par = self.name
                self.mol.ff.read(par)
            else:
                self.mol.ff.assign_params(ff)
        # now generate the converter
        self.ff2lmp = ff2lammps.ff2lammps(self.mol)
        if local:
            self.data_file = self.name+".data"
            self.inp_file  = self.name+".in"
        else:
            raise ValueError, "to be implemented"
        # before writing output we can adjust the settings in ff2lmp
        # TBI
        self.ff2lmp.write_data(filename=self.data_file)
        self.ff2lmp.write_input(filename=self.inp_file, kspace=self.control["kspace"])
        self.lmps.file(self.inp_file)
        # connect variables for extracting
        for e in evars:
            self.lmps.command("variable %s equal %s" % (e, evars[e]))
        for p in pressure:
            self.lmps.command("variable %s equal %s" % (p,p))
        # stole this from ASE lammpslib ... needed to recompute the stress ?? should affect only the ouptut ... compute is automatically generated
        self.lmps.command('thermo_style custom pe temp pxx pyy pzz')
        self.natoms = self.lmps.get_natoms()      
        # compute energy of initial config
        self.calc_energy()
        self.report_energies()
        return

    def get_eterm(self, name):
        assert name in evars
        return self.lmps.extract_variable(name,None,0)
        
    def calc_energy(self):
        self.lmps.command("run 0 post no")
        energy = self.get_eterm("epot")
        return energy
        
    def calc_energy_force(self):
        energy = self.calc_energy()
        fxyz = self.get_force()
        return energy, fxyz
        
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
        xyz.shape=(self.natoms,3)
        return xyz
        
    def get_stress_tensor(self):
        """
        
        """
        ptensor_flat = np.zeros([6])
        for i,p in enumerate(pressure):
            ptensor_flat[i] = self.lmps.extract_variable(p, None, 0)
        ptensor = np.zeros([3,3], "d")
        # "pxx", "pyy", "pzz", "pxy", "pxz", "pyz"
        ptensor[0,0] = ptensor_flat[0]
        ptensor[1,1] = ptensor_flat[1]
        ptensor[2,2] = ptensor_flat[2]
        ptensor[0,1] = ptensor_flat[3]
        ptensor[1,0] = ptensor_flat[3]
        ptensor[0,2] = ptensor_flat[4]
        ptensor[2,0] = ptensor_flat[4]
        ptensor[1,2] = ptensor_flat[5]
        ptensor[2,1] = ptensor_flat[5] 
        # CHECK this conversion from Anthmosphere (real units in lammps) to kcal/mol/A^3 ... soemthing is wrong here
        return ptensor*1.458397e-5

    def set_xyz(self, xyz):
        """
        set the xyz positions froma numpy array
        """
        self.lmps.scatter_atoms("x",1,3,np.ctypeslib.as_ctypes(xyz))
        return
       
    def get_cell(self):
        var = ["boxxlo", "boxxhi", "boxylo", "boxyhi", "boxzlo", "boxzhi", "xy", "xz", "yz"]
        cell_raw = {}
        for v in var:
            cell_raw[v] = self.lmps.extract_global(v, 1)
        # currently only orthorombic
        cell = np.zeros([3,3],"d")
        cell[0,0]= cell_raw["boxxhi"]-cell_raw["boxxlo"]
        cell[1,1]= cell_raw["boxyhi"]-cell_raw["boxylo"]
        cell[2,2]= cell_raw["boxzhi"]-cell_raw["boxzlo"]
        return cell

    def set_cell(self, cell):
        # only orthormobic
        cd = cell.diagonal()
        self.lmps.command("change_box all x final 0.0 %f y final 0.0 %f z final 0.0 %f remap" % tuple(cd))
        return

    def calc_numforce(self,delta=0.0001):
        """
        compute numerical force and compare with analytic 
        for debugging
        """
        energy, fxyz   = self.calc_energy_force()
        num_fxyz = np.zeros([self.natoms,3],"float64")
        xyz      = self.get_xyz()
        for a in xrange(self.natoms):
            for i in xrange(3):
                keep = xyz[a,i]
                xyz[a,i] += delta
                self.set_xyz(xyz)
                ep = self.calc_energy()
                ep_contrib = np.array(self.get_energy_contribs().values())
                xyz[a,i] -= 2*delta
                self.set_xyz(xyz)
                em = self.calc_energy()
                em_contrib = np.array(self.get_energy_contribs().values())
                xyz[a,i] = keep
                num_fxyz[a,i] = -(ep-em)/(2.0*delta)
                # self.pprint("ep em delta_e:  %20.15f %20.15f %20.15f " % (ep, em, ep-em))
                print(np.array2string((em_contrib-ep_contrib)/(2.0*delta),precision=2,suppress_small=True))
                print("atom %d (%s) %d: anal: %12.6f num: %12.6f diff: %12.6f " % (a," ",i,fxyz[a,i],num_fxyz[a,i],( fxyz[a,i]-num_fxyz[a,i])))
        return fxyz, num_fxyz
        
    def calc_numlatforce(self, delta=0.001):
        """
        compute the numeric force on the lattice (currently only orthormobic)
        """
        num_latforce = np.zeros([3], "d")
        cell = self.get_cell()
        for i in xrange(3):
            cell[i,i] += delta
            self.set_cell(cell)
            ep = self.calc_energy()
            cell[i,i] -= 2*delta
            self.set_cell(cell)
            em = self.calc_energy()
            cell[i,i] += delta
            self.set_cell(cell)
            print(ep, em)
            num_latforce[i] = -(ep-em)/(2.0*delta)
        return num_latforce
                
        
        
    def end(self):
        # clean up TBI
        return

###################### wrapper to tasks like MIN or MD #######################################

    def MIN_cg(self, thresh, method="hftn", etol=0.0, maxiter=10, maxeval=100):
        assert method in ["cg", "hftn", "sd"]
        self.lmps.command("min_style %s" % method)
        self.lmps.command("minimize %f %f %d %d" % (etol, thresh, maxiter*self.natoms, maxeval*self.natoms))
        return
        
    def LATMIN_boxrel(self, st_thresh, thresh, method="cg", etol=0.0, maxiter=10, maxeval=100, p=0.0):
        assert method in ["cg", "sd"]
        stop = False
        self.lmps.command("min_style %s" % method)
        self.lmps.command("minimize %f %f %d %d" % (etol, thresh, maxiter*self.natoms, maxeval*self.natoms))
        while not stop:
            self.lmps.command("fix latmin all box/relax iso %f vmax 0.01" % p)            
            self.lmps.command("minimize %f %f %d %d" % (etol, thresh, maxiter*self.natoms, maxeval*self.natoms))
            self.lmps.command("unfix latmin")
            self.lmps.command("min_style %s" % method)
            self.lmps.command("minimize %f %f %d %d" % (etol, thresh, maxiter*self.natoms, maxeval*self.natoms))
            print("CELL :")
            print(self.get_cell())
            print("Stress TENSOR :")
            st = self.get_pressure()
            print(st)
            rms_st = np.sqrt((st*st).sum())
            if rms_st<st_thresh: stop=True
        return
            
        return

