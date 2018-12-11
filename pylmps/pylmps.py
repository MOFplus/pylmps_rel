# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 17:43:56 2017

@author: rochus

                 pylmps

   this is a second and simple wrapper for lammps (not using Pylammps .. makes problem in MPI parallel)
   in the style of pydlpoly ... 

"""
from __future__ import print_function
import __builtin__

import numpy as np
import string
import os
from mpi4py import MPI

import molsys
import ff2lammps
from util import rotate_cell
from molsys import mpiobject
wcomm = MPI.COMM_WORLD
# overload print function in parallel case
#import __builtin__
#def print(*args, **kwargs):
#    if mpi_rank == 0:
#        return __builtin__.print(*args, **kwargs)
#    else:
#        return

from molsys.util import pdlpio2

try:
    from lammps import lammps
except ImportError:
    print("ImportError: Impossible to load lammps")


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

class pylmps(mpiobject):
    
    def __init__(self, name, logfile = "none", screen = True, mpi_comm=None, out = None):
        super(pylmps, self).__init__(mpi_comm,out)
        self.name = name
        # start lammps
        cmdargs = ['-log', logfile]
        if screen == False: cmdargs+=['-screen', 'none']
        self.lmps = lammps(cmdargs=cmdargs, comm = self.mpi_comm)
        for e in evars:
            self.lmps.command("variable %s equal %s" % (e, evars[e]))
        # TBI
        self.control = {}
        self.control["kspace"] = False
        self.control["oop_umbrella"] = False
        self.control["kspace_gewald"] = 0.0
        self.control["cutoff"] = 12.0
        # defaults
        self.pdlp = None
        self.md_dumps = []
        # datafuncs
        self.data_funcs = {\
            "xyz"   : self.get_xyz,\
            "vel"   : self.get_vel,\
            "force" : self.get_force,\
            "cell"  : self.get_cell,\
        }
        return

    def setup(self, mfpx=None, local=True, mol=None, par=None, ff="MOF-FF", pdlp=None, restart=None,
            logfile = 'none', bcond=3, kspace = False, uff="UFF4MOF"):
        """ the setup creates the data structure necessary to run LAMMPS
        
        
            mfpx (molsys.mol, optional): Defaults to None. mol instance containing the atomistic system
            local (bool, optional): Defaults to True. If true: run in current folder, if not: create run folder
            mol (molsys.mol, optional): Defaults to None. mol instance containing the atomistic system
            par (str, optional): Defaults to None. filename of the .par file containing the term infos
            ff (str, optional): Defaults to "MOF-FF". Name of the used Forcefield when assigning from the web MOF+
            pdlp (str, optional): defaults to None. Filename of the pdlp file 
            restart (str, optional): stage name of the pdlp fiel to restart from
            logfile (str, optional): Defaults to 'none'. logfile
            bcond (int, optional): Defaults to 2. Boundary Condition
            kspace (bool, optional): Defaults to False. if True: use SPME, else use shift damping for coulomb
            uff (str, optional): Defaults to UFF4MOF. Can only be UFF or UFF4MOF. If ff="UFF" then a UFF setup with lammps_interface is generated using either option
        """
        self.control["kspace"] = kspace
#        cmdargs = ['-log', logfile]
#        if screen == False: cmdargs+=['-screen', 'none']
#        self.lmps = lammps(cmdargs=cmdargs, comm = self.mpi_comm)

        # depending on what type of input is given a setup will be done
        # the default is to load an mfpx file and assign from MOF+ (using force field MOF-FF)
        # if par is given or ff="file" we use mfpx/ric/par
        # if mol is given then this is expected to be an already assigned mol object
        #      (in the latter case everything else is ignored!)
        self.start_dir = os.getcwd()+"/"
        # if ff is set to "UFF" assignement is done via a modified lammps_interface from peter boyds
        self.use_uff = False
        if ff == "UFF":
            self.pprint("USING UFF SETUP!! EXPERIMENTAL!!")
            self.use_uff = True
        # set the pdlp filename
        if pdlp is None:
            self.pdlpname = self.start_dir + self.name + ".pdlp"
        else:
            self.pdlpname = self.start_dir + pdlp
        # get the mol instance either directly or from file or as an argument
        if mol != None:
            self.mol = mol
        else:
            if restart is not None:
                # The mol object should be read from the pdlp file
                self.pdlp = pdlpio2.pdlpio2(self.pdlpname, ffe=self, restart=restart)
                self.mol  = self.pdlp.get_mol_from_system()
            else:
                # we need to make a molsys and read it in
                self.mol = molsys.mol()
                if mfpx == None:
                    mfpx = self.name + ".mfpx"
                self.mol.read(mfpx)
        # get the forcefield if this is not done already (if addon is there assume params are exisiting .. TBI a flag in ff addon to indicate that params are set up)
        self.data_file = self.name+".data"
        self.inp_file  = self.name+".in"
        if not self.use_uff:
            if not "ff" in self.mol.loaded_addons:
                self.mol.addon("ff")
                if par or ff=="file":
                    if par == None:
                        par = self.name
                    self.mol.ff.read(par)
                else:
                    self.mol.ff.assign_params(ff)
            self.mol.bcond = bcond
            # now generate the converter
            self.ff2lmp = ff2lammps.ff2lammps(self.mol)
            # adjust the settings
            if self.control["oop_umbrella"]:
                self.pprint("using umbrella_harmonic for OOP terms")
                self.ff2lmp.setting("use_improper_umbrella_harmonic", True)
            if self.control["kspace_gewald"] != 0.0:
                self.ff2lmp.setting("kspace_gewald", self.control["kspace_gewald"])
            self.ff2lmp.setting("cutoff", self.control["cutoff"])
        if local:
            self.rundir=self.start_dir
        else:
            self.rundir = self.start_dir+'/'+self.name
            if self.mpi_comm.Get_rank() ==0:
                if os.path.isdir(self.rundir):
                    i=1
                    temprundir = self.rundir+('_%d' % i)
                    while os.path.isdir(temprundir):
                        i+=1
                        temprundir = self.rundir+('_%d' % i)
                    self.rundir = temprundir
                os.mkdir(self.rundir)
            self.rundir=self.mpi_comm.bcast(self.rundir)
            self.mpi_comm.Barrier()
            self.pprint(self.rundir)
            self.pprint(self.name)
            os.chdir(self.rundir)
        #further settings in order to be compatible to pydlpoly
        self.QMMM = False
        self.bcond = bcond
        if self.use_uff:
            self.setup_uff(uff)
        else:
            # before writing output we can adjust the settings in ff2lmp
            # TBI
            self.ff2lmp.write_data(filename=self.data_file)
            self.ff2lmp.write_input(filename=self.inp_file, kspace=self.control["kspace"])
        # now in and data exist and we can start up    
        self.lmps.file(self.inp_file)
        os.chdir(self.start_dir)
        # connect variables for extracting
        for e in evars:
            self.lmps.command("variable %s equal %s" % (e, evars[e]))
        for p in pressure:
            self.lmps.command("variable %s equal %s" % (p,p))
        self.lmps.command("variable vol equal vol")
        # stole this from ASE lammpslib ... needed to recompute the stress ?? should affect only the ouptut ... compute is automatically generated
        self.lmps.command('thermo_style custom pe temp pxx pyy pzz')
        if self.mol.ff.settings["coreshell"]: self._create_grouping()
        #self.natoms = self.lmps.get_natoms()
        # compute energy of initial config
        self.calc_energy()
        self.report_energies()
        self.md_fixes = []
        # Now connect pdlpio (using pdlpio2)
        #if self.pdlp is None:
        #    self.pdlp = pdlpio2.pdlpio2(self.pdlpname, ffe=self)
        return

    def _create_grouping(self):
        """
        Create some sensible atom grouping. Most importantly there will be
        groups "all_cores" and "all_shells" that combine all the technical
        DOFs. This string is supposed to be directly run via the lammps binding.
        """
        assert self.mol.ff.settings["coreshell"] == True
        self.mol.ff.cores = []
        self.mol.ff.shells = []
        self.mol.ff.shells2cores = []
        for i,at in enumerate(self.mol.atypes):
            if at[0]=="x":
                self.mol.ff.shells.append(i)
                self.mol.ff.shells2cores.append(self.mol.conn[i][0])
            else:
                self.mol.ff.cores.append(i)
        for i in self.mol.ff.shells:
            self.lmps.command("group all_shells id %i" % (i+1))
        for i in self.mol.ff.cores:
            self.lmps.command("group all_atoms_and_cores id %i" % (i+1))
        # TODO: make these variables?
        self.lmps.command("neighbor 2.0 bin")
        self.lmps.command("comm_modify vel yes")
        return 

    def _relax_shells(self):
        """
        Relax the shells. Returns a string to be used via the bindings.
        """
        s = ""
        s += "fix freeze all_atoms_and_cores setforce 0.0 0.0 0.0\n"
        #s += "thermo 1\n"
        s += "minimize 1e-12 1e-6 100 100\n"
        s += "unfix freeze"
        self.lmps.commands_string(s)
        return

    def setup_uff(self, uff):
        """not to be called ... subfunction of setup to generate input for UFF calculation
        uff is the type of forcefield to be used

        NOTE: uses code taken from lammps_main.py
        usage of options has been heavily modified .. no "command line" options
        note that pair style is a dirty hack

        also note: name of in and data files are hardcoded in the modified lammps_interface code
        """
        from lammps_interface.InputHandler import Options
        from lammps_interface.structure_data import from_molsys
        from lammps_interface.lammps_main import LammpsSimulation
        assert uff in ["UFF", "UFF4MOF"]
        options = Options()
        options.force_field = uff
        options.pair_style  = "lj/cut 12.5"
        sim = LammpsSimulation(self.name, options)
        cell, graph = from_molsys(self.mol)
        sim.set_cell(cell)
        sim.set_graph(graph)
        sim.split_graph()
        sim.assign_force_fields()
        sim.compute_simulation_size()
        sim.merge_graphs()
        sim.write_lammps_files()
        # in the uff mode we need access to the unique atom types in terms of their elements
        self.uff_plmps_elems = [sim.unique_atom_types[i][1]["element"] for i in sim.unique_atom_types.keys()]
    
        return



    def setup_data(self,name,datafile,inputfile,mfpx=None,mol=None,local=True,logfile='none',bcond=2,kspace = True):
        ''' setup method for use with a lammps data file that contains the system information
            can be used for running simulations with data generated from lammps_interface '''
        self.start_dir = os.getcwd()  
        self.name = name
        if mfpx is not None:
            self.mol = molsys.mol.from_file(mfpx)
        else:
            if mol is not None:
                self.mol = mol
            else:
                self.pprint('warning, no mol instance created! some functions of pylmps can not be used!')
                self.pprint('provide either the mfpx file or a mol instance as argument of setup_data')
        self.ff2lmp = ff2lammps.ff2lammps(self.mol,setup_FF=False)
        # update here the ff2lmp data by retrieving the atom types from the input file
        atype_infos = [x.split()[3] for x in open(datafile).read().split('Masses')[-1].split('Bond Coeffs')[0].split('\n') if x != '']
        self.ff2lmp.plmps_elems = [x[0:2].split('_')[0].title() for x in atype_infos]
        self.ff2lmp.plmps_atypes = atype_infos 
        self.data_file = datafile
        self.inp_file  = inputfile
        if local:
            self.rundir = self.start_dir
        self.bcond = bcond
        self.lmps.file(self.inp_file)
        os.chdir(self.start_dir)
        for e in evars:
            self.lmps.command("variable %s equal %s" % (e, evars[e]))
        for p in pressure:
            self.lmps.command("variable %s equal %s" % (p,p))
        self.lmps.command("variable vol equal vol")
        self.lmps.command('thermo_style custom pe temp pxx pyy pzz')
        self.calc_energy()
        self.report_energies()
        self.md_fixes = []
        return

    def command(self, com):
        """
        perform a lammps command
        """
        self.lmps.command(com)
        return

    def file(self,fname):
        self.lmps.file(fname)
        return

    @property
    def natoms(self):
        return self.lmps.get_natoms()

    def get_natoms(self):
        return self.lmps.get_natoms()

    def get_elements(self):
        return self.mol.get_elems()

    def get_eterm(self, name):
        assert name in evars
        return self.lmps.extract_variable(name,None,0)

    def set_atoms_moved(self):
        ''' dummy function that does not actually do amything'''
        return

    def set_logger(self, default = 'none'):
        self.lmps.command("log %s" % default)
        return
        
    def calc_energy(self, init=False):
        if init:
            self.lmps.command("run 0 pre yes post no")
        else:
            self.lmps.command("run 1 pre no post no")
        energy = self.get_eterm("epot")
        return energy
        
    def calc_energy_force(self, init=False):
        energy = self.calc_energy(init)
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
            self.pprint("%15s : %15.8f kcal/mol" % (en, e[en]))
        self.pprint("%15s : %15.8f kcal/mol" % ("TOTAL", etot))
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

    def get_vel(self):
        """
        get the velocity as a numpy array
        """
        vel = np.ctypeslib.as_array(self.lmps.gather_atoms("v",1,3))
        vel.shape=(self.natoms,3)
        return vel

    def get_cell_volume(self):
        """ 
        get the cell volume from lammps variable vol
        """
        vol = self.lmps.extract_variable("vol", None, 0)
        return vol
        
    def get_stress_tensor(self):
        """
        get the stress tensor in kcal/mol/A^3
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
        # conversion from Athmosphere (real units in lammps) to kcal/mol/A^3 
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
        # set tilting
        cell[1,0] = cell_raw['xy']
        cell[2,0] = cell_raw['xz']
        cell[2,1] = cell_raw['yz']
        return cell

    def set_cell(self, cell, cell_only=False):
        # we have to check here if the box is correctly rotated in the triclinic case
        cell = rotate_cell(cell)
        if abs(cell[0,1]) > 10e-14: raise IOError("Cell is not properly rotated")
        if abs(cell[0,2]) > 10e-14: raise IOError("Cell is not properly rotated")
        if abs(cell[1,2]) > 10e-14: raise IOError("Cell is not properly rotated")
#        cd = cell.diagonal()
        cd = tuple(ff2lammps.ff2lammps.cell2tilts(cell))
        if cell_only:
            self.lmps.command("change_box all x final 0.0 %f y final 0.0 %f z final 0.0 %f xy final %f xz final %f yz final %f" % cd)
        else:
            self.lmps.command("change_box all x final 0.0 %f y final 0.0 %f z final 0.0 %f xy final %f xz final %f yz final %f remap" % cd)
#        self.lmps.command("change_box all x final 0.0 %f y final 0.0 %f z final 0.0 %f remap" % tuple(cd))
        return

    def get_cellforce(self):
        cell    = self.get_cell()
        # we need to get the stress tensor times the volume here (in ananlogy to get_stress in pydlpoly)
        stress  = self.get_stress_tensor()*self.get_cell_volume()
        # compute force from stress tensor
        cell_inv = np.linalg.inv(cell)
        cellforce = np.dot(cell_inv, stress)
        # at this point we could constrain the force to maintain the boundary conditions
        if self.bcond ==  2:
            # orthormobic: set off-diagonal force to zero
            cellforce *= np.eye(3)
        elif self.bcond == 1:
            # in the cubic case average diagonal terms
            avrgforce = cellforce.trace()/3.0
            cellforce = np.eye(3)*avrgforce
        elif self.bcond == 112:
            # JK: this one is a special boundary condition, where 
            # - a and b are constrained to be the same
            # - c can be different
            avgforce = (cellforce[0,0] + cellforce[1,1]) / 2.0
            cellforce *=  np.eye(3) 
            cellforce[0,0] = avgforce
            cellforce[1,1] = avgforce
        else:
            pass
        return cellforce

    def update_mol(self):
        self.mol.set_cell(self.get_cell())
        self.mol.set_xyz(self.get_xyz())
        return

    def write(self,fname,**kwargs):
        self.update_mol()
        if self.is_master:
            self.pprint('writing mol to %s' % fname)
            self.mol.write(fname,**kwargs)
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
            ep = self.calc_energy(init=True)
            cell[i,i] -= 2*delta
            self.set_cell(cell)
            em = self.calc_energy(init=True)
            cell[i,i] += delta
            self.set_cell(cell)
            #print(ep, em)
            num_latforce[i] = -(ep-em)/(2.0*delta)
        return num_latforce

    def get_bcond(self):
        return self.bcond
        
    def end(self):
        # clean up TBI
        return

###################### wrapper to tasks like MIN or MD #######################################

    def MIN_cg(self, thresh, method="cg", etol=0.0, maxiter=10, maxeval=100):
        assert method in ["cg", "hftn", "sd"]
        # transform tresh from pydlpoly tresh to lammps tresh
        # pydlpoly uses norm(f)*sqrt(1/3nat) whereas lammps uses normf
        thresh *= np.sqrt(3*self.natoms)
        self.lmps.command("min_style %s" % method)
        self.lmps.command("minimize %f %f %d %d" % (etol, thresh, maxiter*self.natoms, maxeval*self.natoms))
        self.report_energies()
        return

    MIN = MIN_cg
        
    def LATMIN_boxrel(self, threshlat, thresh, method="cg", etol=0.0, maxiter=10, maxeval=100, p=0.0,maxstep=20):
        assert method in ["cg", "sd"]
        thresh *= np.sqrt(3*self.natoms)
        couplings = {
                1: 'iso',
                2: 'aniso',
                3: 'tri'}
        stop = False
        self.lmps.command("min_style %s" % method)
        self.lmps.command("minimize %f %f %d %d" % (etol, thresh, maxiter*self.natoms, maxeval*self.natoms))
        counter = 0
        while not stop:
            self.lmps.command("fix latmin all box/relax %s %f vmax 0.01" % (couplings[self.bcond], p))            
            self.lmps.command("minimize %f %f %d %d" % (etol, thresh, maxiter*self.natoms, maxeval*self.natoms))
            self.lmps.command("unfix latmin")
            self.lmps.command("min_style %s" % method)
            self.lmps.command("minimize %f %f %d %d" % (etol, thresh, maxiter*self.natoms, maxeval*self.natoms))
            self.pprint("CELL :")
            self.pprint(self.get_cell())
#            print("Stress TENSOR :")
#            st = self.get_pressure()
#            print(st)
#            rms_st = np.sqrt((st*st).sum())
#            if rms_st<st_thresh: stop=True
            cellforce = self.get_cellforce()
            self.pprint ("Current cellforce:\n%s" % np.array2string(cellforce,precision=4,suppress_small=True))
            rms_cellforce = np.sqrt(np.sum(cellforce*cellforce)/9.0)
            self.pprint ("Current rms cellforce: %12.6f" % rms_cellforce)
#            latiter += 1
#            if latiter >= lat_maxiter: stop = True
            if rms_cellforce < threshlat: stop = True
            counter += 1
            if counter >= maxstep: stop=True
            if hasattr(self,'trajfile')==True:
                from molsys.fileIO import lammpstrj
                lammpstrj.write_raw(self.trajfile,counter,self.get_natoms(),self.get_cell(),self.mol.get_elems(),
                                    self.get_xyz(),np.zeros(self.get_natoms(),dtype='float'))
        return
            
    def LATMIN_sd(self,threshlat, thresh, lat_maxiter= 100, maxiter=500, fact = 2.0e-3, maxstep = 3.0):
        """
        Lattice and Geometry optimization (uses MIN_cg for geom opt and steepest descent in lattice parameters)

        :Parameters:
            - threshlat (float)  : Threshold in RMS force on the lattice parameters
            - thresh (float)     : Threshold in RMS force on geom opt (passed on to :class:`pydlpoly.MIN_cg`)
            - lat_maxiter (int)  : Number of Lattice optimization steepest descent steps
            - fact (float)       : Steepest descent prefactor (fact x gradient = stepsize)
            - maxstep (float)    : Maximum stepsize (step is reduced if larger then this value)

        """
        self.pprint ("\n\nLattice Minimization: using steepest descent for %d steps (threshlat=%10.5f, thresh=%10.5f)" % (lat_maxiter, threshlat, thresh))
        self.pprint ("                      the geometry is relaxed with MIN_cg at each step for a mximum of %d steps" % maxiter)
        self.pprint ("Initial Optimization ")
        self.MIN_cg(thresh, maxiter=maxiter)
        # to zero the stress tensor
        oldenergy = self.calc_energy()
        cell = self.get_cell()
        self.pprint ("Initial cellvectors:\n%s" % np.array2string(cell,precision=4,suppress_small=True))
        cellforce = self.get_cellforce()
        self.pprint ("Initial cellforce:\n%s" % np.array2string(cellforce,precision=4,suppress_small=True))
        stop = False
        latiter = 1
        while not stop:
            self.pprint ("Lattice optimization step %d" % latiter)
            step = fact*cellforce
            steplength = np.sqrt(np.sum(step*step))
            self.pprint("Unconstrained step length: %10.5f Angstrom" % steplength)
            if steplength > maxstep:
                self.pprint("Constraining to a maximum steplength of %10.5f" % maxstep)
                step *= maxstep/steplength
            new_cell = cell + step
            # cell has to be properly rotated for lammps
            new_cell = rotate_cell(new_cell)
            self.pprint ("New cell:\n%s" % np.array2string(new_cell,precision=4,suppress_small=True))
            self.set_cell(new_cell)
            self.MIN_cg(thresh, maxiter=maxiter)
            energy = self.calc_energy()
            if energy > oldenergy:
                self.pprint("WARNING: ENERGY SEEMS TO RISE!!!!!!")
            oldenergy = energy
            cell = self.get_cell()
            #print ("Current cellvectors:\n%s" % str(cell))
            cellforce = self.get_cellforce()
            self.pprint ("Current cellforce:\n%s" % np.array2string(cellforce,precision=4,suppress_small=True))
            rms_cellforce = np.sqrt(np.sum(cellforce*cellforce)/9.0)
            self.pprint ("Current rms cellforce: %12.6f" % rms_cellforce)
            latiter += 1
            if hasattr(self,'trajfile')==True:
                from molsys.fileIO import lammpstrj
                lammpstrj.write_raw(self.trajfile,latiter,self.get_natoms(),self.get_cell(),self.mol.get_elems(),
                                    self.get_xyz(),np.zeros(self.get_natoms(),dtype='float'))
            if latiter >= lat_maxiter: stop = True
            if rms_cellforce < threshlat: stop = True
        self.pprint ("SD minimizer done")
        return

    LATMIN = LATMIN_sd

    def MD_init(self, stage, T = None, p=None, startup = False, ensemble='nve', thermo=None, 
            relax=(0.1,1.), traj=None, rnstep=100, tnstep=100,timestep = 1.0, bcond = 'iso', 
            colvar = None, mttk_volconstraint='yes', log = True, dump=True, append=False):
        """Defines the MD settings
        
        MD_init has to be called before a MD simulation can be performed, the ensemble along with the
        necesary information (Temperature, Pressure, ...), but also trajectory writing frequencies are defined here
        
        Args:
            stage (str): name of the stage
            T (float, optional): Defaults to None. Temperature of the simulation, if List of len 2 LAMMPS will perform
            a linear ramp of the Temperature
            p (float or list of floats, optional): Defaults to None. Pressure of the simulation, if List of len 2 Lammps will perform
            a linear ramp of the Temperature
            startup (bool, optional): Defaults to False. if True, sample initial velocities from maxwell boltzmann distribution
            ensemble (str, optional): Defaults to 'nve'. ensemble of the simulation, can be one of 'nve', 'nvt' or 'npt'
            thermo (str, optional): Defaults to None. Thermostat to be utilized, can be 'ber' or 'hoover'
            relax (tuple, optional): Defaults to (0.1,1.). relaxation times for the Thermostat and Barostat
            traj (list of strings, optional): Defaults to None. defines what is written to the pdlp file
            rnstep (int, optional): Defaults to 100. restart writing frequency
            tnstep (int, optional): Defaults to 100. trajectory writing frequency
            timestep (float, optional): Defaults to 1.0. timestep in fs
            bcond (str, optional): Defaults to 'iso'. boundary condition (check what you set in the setup!) not sure if that does anythign here
            colvar (string, optional): Defaults to None. if given, the Name of the colvar input file. LAMMPS has to be compiled with colvars in order to use it
            mttk_volconstraint (str, optional): Defaults to 'yes'. if 'mttk' is used as barostat, define here whether to constraint the volume
            log (bool, optional): Defaults to True. defines if log file is written
            dump (bool, optional): Defaults to True: defines if an ASCII dump is written
            append (bool, optional): Defaults to False: if True data is appended to the exisiting stage (TBI)
        
        Returns:
            None: None
        """
        assert bcond in ['iso', 'aniso', 'tri']
        def conversion(r):
            return r * 1000/timestep 
        # pressure in athmospheres
        # if wished open a specific log file
        if log:
            self.lmps.command('log %s/%s.log' % (self.rundir,stage))
        # first specify the timestep in femtoseconds
        # the relax values are multiples of the timestep
        self.lmps.command('timestep %12.6f' % timestep)
        # manage output, this is the setup for the output written to screen and log file
        # define a string variable holding the name of the stage to be printed before each output line, like
        # in pydlpoly
        label = '%-5s' % (stage.upper()[:5])
        self.lmps.command('thermo_style custom step ecoul elong ebond eangle edihed eimp pe\
                ke etotal temp press vol cella cellb cellc cellalpha cellbeta cellgamma\
                pxx pyy pzz pxy pxz pyz')
        # check if pressure or temperature ramp is requrested. in this case len(T/P) == 2
        if hasattr(T,'__iter__'):
            T1,T2=T[0],T[1]
        else:
            T1,T2 = T,T
        if hasattr(p,'__iter__'):
            p1,p2=p[0],p[1]
        else:
            p1,p2 = p,p
        # generate regular dump (ASCII)
        if dump is True:
            self.lmps.command('dump %s all custom %i %s.dump id type element xu yu zu' % (stage+"_dump", tnstep, stage))
            if self.use_uff:
                plmps_elems = self.uff_plmps_elems
            else:
                plmps_elems = self.ff2lmp.plmps_elems
            self.lmps.command('dump_modify %s element %s' % (stage+"_dump", string.join(plmps_elems)))
            self.md_dumps.append(stage+"_dump")
        # self.lmps.command('dump %s all h5md %i %s.h5 position box yes' % (stage+"h5md",tnstep,stage))
        if traj is not None:
            # NOTE this is a hack .. need to make cells orhto for bcond=1&2 automatically. add triclinic cells to pdlp
            assert self.mol.bcond < 3
            self.lmps.command("change_box all ortho")
            # ok, we will also write to the pdlp file
            if append:
                raise IOError, "TBI"
            else:
                # stage is new and we need to define it and set it up
                self.pdlp.add_stage(stage)
                self.pdlp.prepare_stage(stage, traj, tnstep, tstep=timestep/1000.0)
                # now close the hdf5 file becasue it will be written within lammps
                self.pdlp.close()
                # now create the dump
                traj_string = string.join(traj)
                print("dump %s all pdlp %i %s.pdlp stage %s %s" % (stage+"_pdlp", tnstep, self.pdlp.fname, stage, traj_string))
                self.lmps.command("dump %s all pdlp %i %s stage %s %s" % (stage+"_pdlp", tnstep, self.pdlp.fname, stage, traj_string))
                self.md_dumps.append(stage+"_pdlp")
        # do velocity startup
        if startup:
            self.lmps.command('velocity all create %12.6f 42 rot yes dist gaussian' % (T1))
        # apply fix
        if ensemble == 'nve':
            self.md_fixes = [stage]
            self.lmps.command('fix %s all nve' % (stage))
        elif ensemble == 'nvt':
            if thermo == 'ber':
                self.lmps.command('fix %s all temp/berendsen %12.6f %12.6f %i'% (stage,T1,T2,conversion(relax[0])))
                self.lmps.command('fix %s_nve all nve' % stage)
                self.md_fixes = [stage, '%s_nve' % stage]
            elif thermo == 'hoover':
                self.lmps.command('fix %s all nvt temp %12.6f %12.6f %i' % (stage,T1,T2,conversion(relax[0])))
                self.md_fixes = [stage]
            else: 
                raise NotImplementedError
        elif ensemble == "npt":
            if thermo == 'hoover':
                self.lmps.command('fix %s all npt temp %12.6f %12.6f %i %s %12.6f %12.6f %i' 
                        % (stage,T1,T2,conversion(relax[0]),bcond, p1, p2, conversion(relax[1])))
                self.md_fixes = [stage]
            elif thermo == 'ber':
                assert bcond != "tri"
                self.lmps.command('fix %s_temp all temp/berendsen %12.6f %12.6f %i'% (stage,T1,T2,conversion(relax[0])))
                self.lmps.command('fix %s_press all press/berendsen %s %12.6f %12.6f %i'% (stage,bcond,p1,p2,conversion(relax[1])))
                self.lmps.command('fix %s_nve all nve' % stage)
                self.md_fixes = ['%s_temp' % stage,'%s_press' % stage , '%s_nve' % stage]
            elif thermo == 'mttk':
                self.lmps.command('fix %s_mttknhc all mttknhc temp %8.4f %8.4f %8.4f tri %12.6f %12.6f %12.6f volconstraint %s'
                               % (stage,T1,T2,conversion(relax[0]),p1,p2,conversion(relax[1]),mttk_volconstraint))
                self.lmps.command('fix_modify %s_mttknhc energy yes'% (stage,))
                self.lmps.command('thermo_style custom step ecoul elong ebond eangle edihed eimp pe ke etotal temp press vol cella cellb cellc cellalpha cellbeta cellgamma pxx pyy pzz pxy pxz pyz')
                self.md_fixes = ['%s_mttknhc'% (stage,)]
            else:
                raise NotImplementedError
        else:
            self.pprint('WARNING: no ensemble specified (this means no fixes are set!), continuing anyway! ')
            #raise NotImplementedError
        if colvar is not None:
            self.lmps.command("fix col all colvars %s" %  colvar)
            self.md_fixes.append("col")
        return

    def MD_run(self, nsteps, printout=100):
        #assert len(self.md_fixes) > 0
        self.lmps.command('thermo %i' % printout)
        self.lmps.command('run %i' % nsteps)
        for fix in self.md_fixes: self.lmps.command('unfix %s' % fix)
        self.md_fixes = []
        for dump in self.md_dumps: self.lmps.command('undump %s' % dump)
        self.md_dumps = []
        self.lmps.command('reset_timestep 0')
        return

 






