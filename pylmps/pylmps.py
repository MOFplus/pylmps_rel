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
import os
from mpi4py import MPI

try:
    import __builtin__
except ImportError:
    import builtins as __builtin__

from pathlib import Path

import molsys
from . import ff2lammps
from .util import rotate_cell
from molsys import mpiobject

from molsys.util import mfp5io 

from molsys.util.timer import timer, Timer

import molsys.util.elems as elems

try:
    from lammps import lammps
except ImportError:
    print("ImportError: Impossible to load lammps")



pressure = ["pxx", "pyy", "pzz", "pxy", "pxz", "pyz"]
bcond_map = {0:'non', 1:'iso', 2:'aniso', 3:'tri'}
cellpar  = ["cella", "cellb", "cellc", "cellalpha", "cellbeta", "cellgamma"]


class pylmps(mpiobject):
    
    def __init__(self, name, logfile = "none", screen = True, mpi_comm=None, out = None, partitions=None):
        super(pylmps, self).__init__(mpi_comm,out)
        self.name = name
        # get timer
        self.timer = Timer("pylmps")
        # start lammps
        cmdargs = ['-log', logfile]
        if screen == False: cmdargs+=['-screen', 'none']
        self.print_to_screen = screen
        # handle partitions .. if None do nothing, if not an integer must be given
        #                      and the number of MPI ranks must be a multiple of partitions
        self.partitions = partitions
        if partitions is not None:           
            procperpart = int(self.mpi_size/partitions)
            assert procperpart > 0
            assert self.mpi_size%procperpart == 0
            self.part_size = procperpart
            self.part_num  = self.mpi_rank//procperpart
            self.part_rank = self.mpi_rank%procperpart
            print ("Using %3d partitions and %3d cores in total. This is partition %d (rank %d)" % \
                        (partitions, self.mpi_size, self.part_num, self.part_rank))
            # generate an empty file lammps.in (if it does not exist) to be used as input 
            #  in case of partitions we need this becasue we can not read from stdin
            infile = Path("lammps.in")
            infile.touch()
            cmdargs += ["-p", "%dx%d" % (partitions, procperpart), "-in", "lammps.in"]
        self.lmps = lammps(cmdargs=cmdargs, comm = self.mpi_comm)
        # handle names of energy contributions
        self.evars = {
         "vdW"     : "evdwl",
         "Coulomb" : "ecoul",
         "CoulPBC" : "elong",
         "bond"    : "ebond",
         "angle"   : "eangle",
         "oop"     : "eimp",
         "torsion": "edihed",
         "epot"    : "pe",
         }
        self.enames = ["vdW", "Coulomb", "CoulPBC", "bond", "angle", "oop", "torsion"]
        for e in self.evars:
            self.lmps.command("variable %s equal %s" % (e, self.evars[e]))
        # control dictionary .. define all defaults here.
        # change either by setting before setup or use a kwarg in setup
        self.control = {}
        self.control["kspace"] = True
        self.control["kspace_method"] = 'ewald'
        self.control["kspace_prec"] = 1.e-6
        self.control["oop_umbrella"] = False
        self.control["kspace_gewald"] = 0.0
        self.control["cutoff"] = 12.0
        self.control["cutoff_coul"] = None
        self.control["origin"] = "zero"        # location of origin in the box: either "zero" or "center"
        self.control["boundary"] = None
        # reax defaults
        self.control["reaxff_timestep"] = 0.1  # ReaxFF timestep is smaller than usual
        self.control["reaxff_filepath"] = "."
        self.use_reaxff = False
        if "REAXFF_FILES" in os.environ:
            self.control["reaxff_filepath"] = os.environ["REAXFF_FILES"]
        self.control["reaxff_bondfile"] = self.name + ".bonds"
        self.control["reaxff_bondfreq"] = 200
        self.control["reaxff_safezone"] = None
        self.control["reaxff_mincap"]   = None
        # defaults
        self.is_setup = False # will be set to True in setup -> to warn if certain functions are used after setup
        self.mfp5 = None
        self.md_dumps = []
        self.external_pot = []
        self.use_restraints = False
        self.restraints = {}      # dictionary of restraints activated on MIN or MD_init
        # datafuncs
        self.data_funcs = {\
            "xyz"    : self.get_xyz,\
            "vel"    : self.get_vel,\
            "force"  : self.get_force,\
            "cell"   : self.get_cell,\
            "charges": self.get_charge,\
            "stress" : self.get_stress_tensor,\
        }
        return

    def add_ename(self, ename, evar):
        if ename not in self.evars:
            self.evars[ename] = evar
            self.enames.append(ename)
            self.command("variable %s equal %s" % (ename, evar)) 
        return

    def add_external_potential(self, expot, callback=None):
        """Add an external Potential

        This potential will be added as a python/invoke fix and needs to be derived from the expot_base class
        It will be called during optimization and MD.
        This needs to be called BEFORE setup.

        Note that the callback function must be defined in the global namespace. If no name is given then we generate it based on the 
        name. If you run multiple instances with the same expot this could lead to problems
        
        Args:
            expot (instance of expot derived class): object that computes the external potential energy and forces
            callback (string): Defaults to None, name of the callback function in the global namespace (will be generated if None)
        """
        assert expot.is_expot == True
        assert self.is_setup == False, "Expot setup must be called before pylmps setup"
        if callback == None:
            callback_name = "callback_expot_%s" % expot.name
        else:
            assert type(callback) == type("")
            callback_name = callback
        # this is pretty uuuugly .. but we need the name of the callback function in the global namespace to find it in the fix
        # TBI we could check here that it does not exist
        # globals()[callback_name] = expot.callback
        self.external_pot.append((expot, callback_name))
        return

    def add_restraint(self, rtype, atoms, params, growK=False):
        """add a restraint to the system
        
        Args:
            rtype (string): type of restraint (bnd, ang, dih)
            atoms (tuple of int): atom indices (pylmps counting from 0!!!!)
            params (tuple/list of floats): force cont, ref value, optional mult for dih
            growK (bool, optional): growing K from zero in next MDrun. Defaults to False.
        """
        resttypes = ["bnd", "ang", "dih"]
        assert rtype in resttypes
        assert len(atoms) == resttypes.index(rtype)+2
        if rtype == "dih":
            assert len(params) == 2 or len(params) == 3
        else:
            assert len(params) == 2 
        params = list(params) + [growK]
        atoms = tuple(atoms)
        self.restraints[atoms] = params
        self.use_restraints = True
        return
    
    def clear_restraints(self):
        self.use_restraints = False
        self.restraints = {}
        return

    def set_restraints(self):
        if not self.use_restraints:
            return
        rest_string = ""
        for r in self.restraints:
            rp = self.restraints[r]
            if len(r) == 2:
                r0 = rp[1]
                kfin = rp[0]
                if rp[2]:
                    kint = 0.0
                else:
                    kint = kfin
                rest_string += " bond %d %d %10.3f %10.3f %10.3f" % (r[0]+1, r[1]+1, kint, kfin, r0)
            elif len(r) == 3:
                a0 = rp[1]
                kfin = rp[0]
                if rp[2]:
                    kint = 0.0
                else:
                    kint = kfin
                rest_string += " angle %d %d %d %10.3f %10.3f %10.3f" % (r[0], r[1], r[2], kint, kfin, a0)
            elif len(r) == 4:
                d0 = rp[1]
                kfin = rp[0]
                if len(rp) == 4:
                    mult = rp[2]
                    grow = rp[3]
                else:
                    grow = rp[2]
                    mult = False
                if grow:
                    kint = 0.0
                else:
                    kint = kfin
                rest_string += " dihedral %d %d %d %d %10.3f %10.3f %10.3f" % (r[0], r[1], r[2], r[3], kint, kfin, d0)
                if mult:
                    rest_string += " mult %d" % mult
            else:
                raise "THis should not happen"
        print ("Restraints are active")
        # DEBUG DEBUG
        print (rest_string)
        self.lmps.command("fix restr all restrain %s" % rest_string)
        self.lmps.command("fix_modify restr energy yes")
        return

    def unset_restraint(self):
        if self.use_restraints:
            self.lmps.command("unfix restr")
        return

    def setup(self, mfpx=None, local=True, mol=None, par=None, ff="MOF-FF", mfp5=None, restart=None, restart_vel=False, restart_ff=True, pressure_bath_atype=None,
            logfile = 'none', bcond=3, uff="UFF4MOF", use_mfp5=False, reaxff="cho", kspace_style='ewald',
            kspace=True, silent=False, noheader=False,  **kwargs):
        """ the setup creates the data structure necessary to run LAMMPS
        
            any keyword arguments known to control will be set to control

            Args:    
                mfpx (molsys.mol, optional): Defaults to None. mol instance containing the atomistic system
                local (bool, optional): Defaults to True. If true: run in current folder, if not: create run folder
                mol (molsys.mol, optional): Defaults to None. mol instance containing the atomistic system
                par (str, optional): Defaults to None. filename of the .par file containing the term infos
                ff (str, optional): Defaults to "MOF-FF". Name of the used Forcefield when assigning from the web MOF+
                mfp5 (str, optional): defaults to None. Filename of the mfp5 file 
                restart (str, optional): stage name of the mfp5 fiel to restart from
                restart_vel (bool, optional): Defaults to False. If True: read velocities from restart stage (must be given) 
                logfile (str, optional): Defaults to 'none'. logfile
                bcond (int, optional): Defaults to 3. Boundary Condition - 1: cubic, 2: orthorombic, 3: triclinic
                uff (str, optional): Defaults to UFF4MOF. Can only be UFF or UFF4MOF. If ff="UFF" then a UFF setup with lammps_interface is generated using either option
                use_mfp5 (bool, optionl): defaults to False, if True use dump_mfp5 (must be compiled)
                reaxff (str, optional): defaults to "cho". name of the reaxff force field file (ffiled.reax.<name>) to be used if FF=="ReaxFF" 
                kspace (bool): defaults to True. If True, use kspace methods to compute the electrostatic interactions
                kspace_style (str), defaults to "ewald": The method to be used if kspace == True. For now, try with "ewald" or "pppm" (https://lammps.sandia.gov/doc/kspace_style.html)
                kspace_prec (float), defaults to 1e-6: accuracy setting for the kspace method (https://lammps.sandia.gov/doc/kspace_style.html)
                silent (bool, optional), if True do not report energies, defaults to False
                noheader (bool, optional), if True, do not output the header in the lammps input files, defaults to False
        """
        self.timer.start()
        # put all known kwargs into self.control
        for kw in kwargs:
            if kw in self.control:
                self.control[kw] = kwargs[kw]
#        cmdargs = ['-log', logfile]
#        if screen == False: cmdargs+=['-screen', 'none']
#        self.lmps = lammps(cmdargs=cmdargs, comm = self.mpi_comm)
        # assert settings
        assert self.control["origin"] in ["zero", "center"]
        self.control['kspace_style'] = kspace_style
        self.control['kspace'] = kspace
        # depending on what type of input is given a setup will be done
        # the default is to load an mfpx file and assign from MOF+ (using force field MOF-FF)
        # if par is given or ff="file" we use mfpx/ric/par
        # if mol is given then this is expected to be an already assigned mol object
        #      (in the latter case everything else is ignored!)
        self.start_dir = os.getcwd()+"/"
        # if ff is set to "UFF" assignement is done via a modified lammps_interface from peter boyds
        self.use_uff = False
        self.use_reaxff = False
        self.use_xtb = False
        self.use_ase = False
        if ff == "UFF":
            self.pprint("USING UFF SETUP!! EXPERIMENTAL!!")
            self.use_uff = True
        if ff == "ReaxFF":
            self.pprint("USING ReaxFF SETUP!!")
            self.use_reaxff = True
            self.reaxff = reaxff
            # fix energy printout
            for en in ("vdW", "CoulPBC", "bond", "angle", "oop", "torsion"):
                self.enames.remove(en)
            self.enames = ["reax_bond"]+self.enames
            self.evars["reax_bond"] = "evdwl"
        if ff == "xTB" or ff == "ase":
            if ff == "xTB":
                self.use_xtb = True
            elif ff == "ase":
                self.use_ase = True
            for en in ("Coulomb", "vdW", "CoulPBC", "bond", "angle", "oop", "torsion"):
                self.enames.remove(en)
        # set the mfp5 filename
        if mfp5 is None:
            self.mfp5name = self.start_dir + self.name + ".mfp5"
        else:
            self.mfp5name = self.start_dir + mfp5
        # get the mol instance either directly or from file or as an argument
        if mol != None:
            self.mol = mol
        else:
            if restart is not None:
                # The mol object should be read from the mfp5 file
                self.mfp5 = mfp5io.mfp5io(self.mfp5name, ffe=self, restart=restart)   # TODO rename when changed in molsys
                if restart_vel is True:
                    self.mol, restart_vel  = self.mfp5.get_mol_from_system(vel=True, restart_ff=restart_ff)
                else:
                    self.mol = self.mfp5.get_mol_from_system(restart_ff=restart_ff) 
            else:
                # we need to make a molsys and read it in
                self.mol = molsys.mol()
                if mfpx == None:
                    mfpx = self.name + ".mfpx"
                self.mol.read(mfpx)

        # set pressure bath atype to avoid attractive interactions between pressure bath and solute
        if pressure_bath_atype != None:
            self.mol.ff.settings["pressure_bath_atype"] = pressure_bath_atype

        # get the forcefield if this is not done already (if addon is there assume params are exisiting .. TBI a flag in ff addon to indicate that params are set up)
        self.data_file = self.name+".data"
        self.inp_file  = self.name+".in"
        if not self.use_uff and not self.use_reaxff and not self.use_xtb and not self.use_ase:
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
            with self.timer("init ff2lammps"):
                self.ff2lmp = ff2lammps.ff2lammps(self.mol,print_timer=self.print_to_screen)
            # adjust the settings
            if self.control["oop_umbrella"]:
                self.pprint("using umbrella_harmonic for OOP terms")
                self.ff2lmp.setting("use_improper_umbrella_harmonic", True)
            if self.control["kspace_gewald"] != 0.0:
                self.ff2lmp.setting("kspace_gewald", self.control["kspace_gewald"])
            if 'kspace_method' in self.control:
                self.ff2lmp.setting("kspace_method", self.control["kspace_method"])
            if 'kspace_prec' in self.control:
                self.ff2lmp.setting("kspace_prec", self.control["kspace_prec"])
            if 'cutoff' in self.control:
                self.ff2lmp.setting("cutoff", self.control["cutoff"])
            if self.control['cutoff_coul'] is not None:
                self.ff2lmp.setting('cutoff_coul', self.control['cutoff_coul'])
        elif self.use_reaxff or self.use_xtb or self.use_ase:
            # incase of reaxff we need to converter only for the data file
            self.ff2lmp = ff2lammps.ff2lammps(self.mol, reax=True)
        if self.use_uff:
            self.setup_uff(uff)
        # now converter is in place .. transfer settings
        try:
            self.ff2lmp.setting("origin", self.control["origin"])
        except:
            pass
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
        if self.use_reaxff:
            with self.timer("write data"):
                self.ff2lmp.write_data(filename=self.data_file)
            with self.timer("setup reaxff"):
                # in this subroutine lamps commands are issued to read the data file and strat up (instead of reading a lammps input file)
                self.setup_reaxff()
        elif self.use_xtb:
            with self.timer("write data"):
                self.ff2lmp.write_data(filename=self.data_file)
            with self.timer("setup xtb"):
                # in this subroutine lamps commands are issued to read the data file and start up (instead of reading a lammps input file)
                self.setup_xtb()
        elif self.use_ase:
            with self.timer("write data"):
                self.ff2lmp.write_data(filename=self.data_file)
            with self.timer("setup ase"):
                # in this subroutine lamps commands are issued to read the data file and start up (instead of reading a lammps input file)
                self.setup_xtb()
        elif self.use_uff==False:
            # before writing output we can adjust the settings in ff2lmp
            # TBI
            with self.timer("write data"):
                self.ff2lmp.write_data(filename=self.data_file)
            with self.timer("write input"):
                self.ff2lmp.write_input(filename=self.inp_file, kspace=self.control["kspace"], noheader=noheader, boundary=self.control["boundary"])
            with self.timer("lammps read input"):
                self.lmps.file(self.inp_file)
        else:
            self.lmps.file(self.inp_file) # for UFF setup
        os.chdir(self.start_dir)
        # connect variables for extracting
        for e in self.evars:
            self.lmps.command("variable %s equal %s" % (e, self.evars[e]))
        for p in pressure:
            self.lmps.command("variable %s equal %s" % (p,p))
        self.lmps.command("variable vol equal vol")
        # stole this from ASE lammpslib ... needed to recompute the stress ?? should affect only the ouptut ... compute is automatically generated
        self.lmps.command('thermo_style custom pe temp pxx pyy pzz')
        try:
            if self.mol.ff.settings["coreshell"]: self._create_grouping()
        except:
            pass
        #self.natoms = self.lmps.get_natoms()
        # setup any registered external potential
        for expot, callback_name in self.external_pot:
            # run the expot's setup with self as an argument --> you have access to all info within the mol object
            expot.setup(self)
            fix_id = "expot_"+expot.name
            self.lmps.command("fix %s all python/invoke 1 post_force %s" % (fix_id, callback_name))
            self.lmps.command("fix_modify %s energy yes" % fix_id)
            self.add_ename(expot.name, "f_"+fix_id)
            self.pprint("External Potential %s is set up as fix %s" % (expot.name, fix_id))
        # compute energy of initial config
        if restart_vel is not False:
            # set velocities that have been read from mfp5 file (do not use startup=True in MDinit becasue that will overwrite the velocities again)
            self.set_vel(restart_vel)
        self.calc_energy(init=True)
        if not silent:
          self.report_energies()
        self.md_fixes = []
        # Now connect mfp5io (using mfp5io)
        if use_mfp5 and (self.mfp5 is None):
            self.mfp5 = mfp5io.mfp5io(self.mfp5name, ffe=self) # TODO rename when changed in molsys
        # set the flag
        self.is_setup = True
        # report timing
        if self.is_master and self.print_to_screen:
            self.timer.report()
        if not self.use_uff and not self.use_reaxff:
            self.ff2lmp.report_timer()
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
        s += "minimize 1e-12 1e-6 100 200\n"
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
        options.origin = self.control["origin"]
        sim = LammpsSimulation(self.name, options)
        cell, graph = from_molsys(self.mol)
        if cell == None:
            print ("bcond is ", bcond)
        sim.set_cell(cell)
        sim.set_graph(graph)
        sim.split_graph()
        sim.assign_force_fields()
        sim.compute_simulation_size()
        sim.merge_graphs()
        sim.write_lammps_files()
        # in the uff mode we need access to the unique atom types in terms of their elements
        self.uff_plmps_elems = [sim.unique_atom_types[i][1]["element"] for i in list(sim.unique_atom_types.keys())]
    
        return

    def setup_reaxff(self):
        """set up the reax calculation (data file exists)        
        """
        self.lmps.command('units real')
        self.lmps.command('atom_style charge')
        self.lmps.command('atom_modify map hash')
        if self.mol.bcond > 0:
            self.lmps.command('boundary p p p')
        else:
            self.lmps.command('boundary f f f')
        self.lmps.command('read_data ' + self.data_file)  
        # . init force field
        ff = 'pair_style reax/c NULL'
        # possibility to modify mincap and safezone keywords
        if self.control["reaxff_mincap"] is not None:
            ff += ' mincap %d' % int(self.control["reaxff_mincap"])
            self.pprint("Using a non-default mincap of %d" % int(self.control["reaxff_mincap"]))         
        if self.control["reaxff_safezone"] is not None:
            ff += ' safezone %f' % float(self.control["reaxff_safezone"])
            self.pprint("Using a non-default safezone of %f" % float(self.control["reaxff_safezone"]))         
        self.lmps.command(ff)
        # use reaxff content to define the force field and atomtypes from the converter
        reaxff_file = self.control["reaxff_filepath"] + "/ffield.reax." + self.reaxff
        atypes = " ".join(self.ff2lmp.plmps_atypes)
        self.lmps.command('pair_coeff * * %s %s' % (reaxff_file, atypes))
        # now define QEq and other things  with default settings
        self.lmps.command('fix qeq all qeq/reax 1 0.0 10.0 1e-6 reax/c')   
        self.lmps.command('neighbor 2 bin')       
        self.lmps.command('neigh_modify every 10 delay 0 check no')              
        return
        
    def setup_xtb(self):
        """set up the xTB calculation (data file exists)
        """
        self.lmps.command('units real')
        self.lmps.command('atom_style charge')
        self.lmps.command('atom_modify map hash')
        if self.mol.bcond > 0:
            self.lmps.command('boundary p p p')
        else:
            self.lmps.command('boundary f f f')
        self.lmps.command('read_data ' + self.data_file)  

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
        for e in self.evars:
            self.lmps.command("variable %s equal %s" % (e, self.evars[e]))
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
        assert name in self.evars
        return self.lmps.extract_variable(name,None,0)

    def set_atoms_moved(self):
        ''' dummy function that does not actually do amything'''
        return

    def set_logger(self, default = 'none'):
        self.lmps.command("log %s" % default)
        return

    def pprint(self, *args, **kwargs):
        """Parallel print function"""
        if self.is_master and self.print_to_screen:
            __builtin__.print(*args, file=self.out, **kwargs)
            self.out.flush()
        
    def calc_energy(self, init=False):
        if init:
            self.lmps.command("run 0 pre yes post no")
        else:
            self.lmps.command("run 1 pre no post no")
        self.energy = self.get_eterm("epot")
        return self.energy
        
    def calc_energy_force(self, init=False):
        energy = self.calc_energy(init)
        fxyz = self.get_force()
        return energy, fxyz

    def get_energy_contribs(self):
        e = {}
        for en in self.enames:
            e[en] = self.get_eterm(en)
        return e
        
    def report_energies(self):
        e = self.get_energy_contribs()
        etot = 0.0
        for en in self.enames:
            etot += e[en]
            self.pprint("%15s : %15.8f kcal/mol" % (en, e[en]))
        self.pprint("%15s : %15.8f kcal/mol" % ("TOTAL", etot))
        return etot
      
      
    def get_total_energy(self):
        e = self.get_energy_contribs()
        etot = 0.0
        for en in self.enames:
            etot += e[en]
        return etot
        
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

    def get_charge(self):
        """
        get the charges as a numpy array
        """
        chg = np.ctypeslib.as_array(self.lmps.gather_atoms("q",1,1))
        return chg

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

    def set_vel(self, vel):
        """
        set the velocities from a numpy array
        """
        self.lmps.scatter_atoms("v",1,3,np.ctypeslib.as_ctypes(vel))
        return
       
    def get_cell(self):
        var = ["boxxlo", "boxxhi", "boxylo", "boxyhi", "boxzlo", "boxzhi", "xy", "xz", "yz"]
        cell_raw = {}
        for v in var:
            cell_raw[v] = self.lmps.extract_global(v)
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
        """set the cell vectors

        Args:
            cell (numpy array): cell vectors
            cell_only (bool, optional): if True change only the cell params but do not change the systems coords. Defaults to False
        """
        # we have to check here if the box is correctly rotated in the triclinic case
        cell = rotate_cell(cell) 
        #if abs(cell[0,1]) > 10e-14: raise IOError("Cell is not properly rotated")
        #if abs(cell[0,2]) > 10e-14: raise IOError("Cell is not properly rotated")
        #if abs(cell[1,2]) > 10e-14: raise IOError("Cell is not properly rotated")
        cd = cell.diagonal()
        if self.control["origin"] == "zero":
            cd_l = np.zeros([3])
            cd_h = cd
        else:
            cd_l = -cd*0.5
            cd_h = cd*0.5
        comm = "change_box all x final %f %f y final %f %f z final %f %f" % (cd_l[0], cd_h[0], cd_l[1], cd_h[1], cd_l[2], cd_h[2])
        if self.bcond <= 2:
            if ((self.bcond == 1) and (np.var(cd) > 1e-6)): # check if that is a cubic cell, raise error if not!
                raise ValueError('the cell to be set is not a cubic cell,diagonals: '+str(cd))
        elif self.bcond == 3:
            cd_tilts = (cell[1,0],cell[2,0],cell[2,1])
            # cd = tuple(ff2lammps.ff2lammps.cell2tilts(cell)) -> old code refers to ff2lmapps .. unlcear!
            comm += " xy final %f xz final %f yz final %f" % cd_tilts
        else:
            raise ValueError("Unknown bcond %d" % self.bcond)
        if not cell_only:
            comm += " remap"
        self.lmps.command(comm)
        return

    def get_cellforce(self):
        """get the forces on the cell vectors

        TBI: improve handling for triclinic systems .. keep lower trianlge structure from the start

        Returns:
            numpy array: cell force
        """
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

    def write(self,fname, partitions=False, **kwargs):
        self.update_mol()
        if partitions:
            # write each partition
            fname = Path(fname)
            fname = fname.parent / (fname.stem + (".part%d" % self.part_num) + fname.suffix)
            # HACK : molsysmo.write expects a string ... convert moslys to pathlib
            fname = str(fname)
            self.pprint('writing mol to %s' % fname)
            self.mol.write(fname, rank = self.part_rank, **kwargs)
            return
        if self.is_master:
            self.pprint('writing mol to %s' % fname)
            self.mol.write(fname, **kwargs)
        return 

    def write_restart(self,fname='restart.pylmps'):
        cell = self.get_cell()
        xyz = self.get_xyz()
        vel = self.get_vel()
        if self.is_master:
            import pickle
            d = {'cell': cell,
                 'xyz': xyz,
                 'vel': vel}
            pickle.dump(d,open(fname,'wb'))

    def read_restart(self,fname):
        """
            reads a pickled restart written via the read_restart function
        """
        import pickle
        d = pickle.load(open(fname,'rb'))
        xyz = d['xyz']
        vel = d['vel']
        cell = d['cell']
        self.set_cell(cell,cell_only=True)
        self.set_xyz(xyz)
        self.set_vel(vel)
        return

    def calc_numforce(self,delta=0.0001):
        """
        compute numerical force and compare with analytic 
        for debugging
        """
        energy, fxyz   = self.calc_energy_force()
        num_fxyz = np.zeros([self.natoms,3],"float64")
        xyz      = self.get_xyz()
        for a in range(self.natoms):
            for i in range(3):
                keep = xyz[a,i]
                xyz[a,i] += delta
                self.set_xyz(xyz)
                ep = self.calc_energy()
                ep_contrib = np.array(list(self.get_energy_contribs().values()))
                xyz[a,i] -= 2*delta
                self.set_xyz(xyz)
                em = self.calc_energy()
                em_contrib = np.array(list(self.get_energy_contribs().values()))
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
        for i in range(3):
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

###################### stuff for using partitions ############################################

    def collect_part_xyz(self, all=True):
        """collect xyz positions form all partitions in one array on the master node

        can be used as a source for the molsys.traj addon to write all structures to a trajctory file
        """
        if self.partitions is None: return
        # generate emtpy array on master node
        if self.is_master:
            xyz_part = np.empty([self.partitions, self.natoms, 3], dtype="float64")
        else:
            xyz_part = None
        # now loop over partitions
        for p in range(self.partitions):
            # if i am rank 0 of that partition get my xyz and send it to the master node
            if self.part_num == p and self.part_rank == 0:
                pxyz = self.get_xyz()
                self.mpi_comm.Send([pxyz, MPI.DOUBLE], dest=0)
            if self.is_master:
                self.mpi_comm.Recv([xyz_part[p], MPI.DOUBLE], source=p*self.part_size)
        if all:
            xyz_part = self.mpi_comm.bcast(xyz_part)
        return xyz_part

    def write_part(self, fname):
        xyz_part = self.collect_part_xyz()
        temp_mol = self.mol.clone()
        temp_mol.addon("traj", source="array", array=xyz_part)
        temp_mol.traj.write(fname)
        return xyz_part # returned for analysis ...

###################### wrapper to tasks like MIN or MD #######################################

    def MIN_cg(self, thresh, method="cg", etol=0.0, maxiter=10, maxeval=100):
        assert method in ["cg", "hftn", "sd"]
        self.set_restraints()
        # transform tresh from pydlpoly tresh to lammps tresh
        # pydlpoly uses norm(f)*sqrt(1/3nat) whereas lammps uses normf
        thresh *= np.sqrt(3*self.natoms)
        self.lmps.command("min_style %s" % method)
        self.lmps.command("minimize %f %f %d %d" % (etol, thresh, maxiter*self.natoms, maxeval*self.natoms))
        etot = self.report_energies()
        # this command used here wihtout reaxff setup results in crashing LATMIN
        # the error is: 
        # ERROR: Energy was not tallied on needed timestep (../compute_pe.cpp:76)
        # Last command: run 1 pre no post no
        # if self.use_reaxff is True: self.lmps.command("reset_timestep 0")
        self.unset_restraint() 
        return etot

    MIN = MIN_cg
    '''
       custom minimization function (sw)
    '''
    def minimize_cg(self, thresh, method="cg", etol=0.0, maxiter=10000, maxeval=10000, silent=False):
        assert method in ["cg", "hftn", "sd"]
        self.lmps.command("min_style %s" % method)
        self.lmps.command("minimize %f %f %d %d" % (etol, thresh, maxiter, maxeval))
        if not silent:
          self.report_energies()
        return
      
      
    def LATMIN_boxrel(self, threshlat, thresh, method="cg", etol=0.0, maxiter=10, maxeval=100, p=0.0,maxstep=20,iso=None):
        assert method in ["cg", "sd"]
        thresh *= np.sqrt(3*self.natoms)
        if iso is None:
          iso = bcond_map[self.bcond]
        stop = False
        self.lmps.command("min_style %s" % method)
        self.lmps.command("minimize %f %f %d %d" % (etol, thresh, maxiter*self.natoms, maxeval*self.natoms))
        counter = 0
        while not stop:
            self.lmps.command("fix latmin all box/relax %s %f vmax 0.01" % (iso, p))            
            self.lmps.command("minimize %f %f %d %d" % (etol, thresh, maxiter*self.natoms, maxeval*self.natoms))
            self.lmps.command("unfix latmin")
            self.lmps.command("min_style %s" % method)
            self.lmps.command("minimize %f %f %d %d" % (etol, thresh, maxiter*self.natoms, maxeval*self.natoms))
            self.pprint("CELL :")
            self.pprint(self.get_cell())
            cellforce = self.get_cellforce()
            self.pprint ("Current cellforce:\n%s" % np.array2string(cellforce,precision=4,suppress_small=True))
            rms_cellforce = np.sqrt(np.sum(cellforce*cellforce)/9.0)
            self.pprint ("Current rms cellforce: %12.6f" % rms_cellforce)
            if rms_cellforce < threshlat: stop = True
            counter += 1
            if counter >= maxstep: stop=True
            if hasattr(self,'trajfile')==True:
                from molsys.fileIO import lammpstrj
                lammpstrj.write_raw(self.trajfile,counter,self.get_natoms(),self.get_cell(),self.mol.get_elems(),
                                    self.get_xyz(),np.zeros(self.get_natoms(),dtype='float'))
        self.unset_restraint()
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
    
    def LATMIN_sd2(self,threshlat, thresh, lat_maxiter= 100, maxiter=500, fact = 2.0e-3, maxstep = 3.0):
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
            #if energy > oldenergy:
            #    self.pprint("WARNING: ENERGY SEEMS TO RISE!!!!!!")
            #    # revert the step!
            #    old_cell = cell - step
            #    self.set_cell(old_cell)
            #    self.MIN_cg(thresh,maxiter=maxiter)
            #    fact /= 1.2
            #    print(fact)
            oldenergy = energy
            cell = self.get_cell()
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

    def MD_init(self, stage, T = None, p=None, startup = False, startup_seed = 42, ensemble='nve', thermo=None, 
            relax=(0.1,1.), traj=[], rnstep=100, tnstep=100,timestep = 1.0, bcond = None,mttkbcond='tri', 
            colvar = None, mttk_volconstraint="no", log = True, dump=True, append=False, dump_thermo=True, 
            wrap = True, additional_thermo_output=[]):
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
            startup_seed (int, optional): Defaults to 42. Random seed used for the initial velocities 
            ensemble (str, optional): Defaults to 'nve'. ensemble of the simulation, can be one of 'nve', 'nvt' or 'npt'
            thermo (str, optional): Defaults to None. Thermostat to be utilized, can be 'ber' or 'hoover'
            relax (tuple, optional): Defaults to (0.1,1.). relaxation times for the Thermostat and Barostat
            traj (list of strings, optional): Defaults to None. defines what is written to the mfp5 file
            rnstep (int, optional): Defaults to 100. restart writing frequency
            tnstep (int, optional): Defaults to 100. trajectory writing frequency
            timestep (float, optional): Defaults to 1.0. timestep in fs
            bcond (str, optional): Defaults to None. by default, the bcond defined in setup is used. only if overwritten here as 'iso' (1), 'aniso' (2) or 'tri' (3), this bcond is used.
            colvar (string, optional): Defaults to None. if given, the Name of the colvar input file. LAMMPS has to be compiled with colvars in order to use it
            mttk_volconstraint (str, optional): Defaults to 'yes'. if 'mttk' is used as barostat, define here whether to constraint the volume
            log (bool, optional): Defaults to True. defines if log file is written
            dump (bool, optional): Defaults to True: defines if an ASCII dump is written
            append (bool, optional): Defaults to False: if True data is appended to the exisiting stage (TBI)
            dump_thermo (bool, optional): defaults to True: if True dump the thermo data written to the log file also to the mfp5 dump
            additional_thermo_output (list, optional): defaults to []: if non-empty, add the thermo columns to the output
        
        Returns:
            None: None
        """
        if bcond == None: bcond = bcond_map[self.bcond]
        assert bcond in ['non', 'iso', 'aniso', 'tri']
        conv_relax = 1000/timestep 
        # pressure in atmospheres
        # if wished open a specific log file
        if log:
            self.lmps.command('log %s/%s.log' % (self.rundir,stage))
        # first specify the timestep in femtoseconds
        # the relax values are multiples of the timestep
        if self.use_reaxff:
            self.lmps.command('timestep %.02f' % self.control["reaxff_timestep"]) 
        else:
            self.lmps.command('timestep %12.6f' % timestep)
        # manage output, this is the setup for the output written to screen and log file
        # define a string variable holding the name of the stage to be printed before each output line, like
        # in pydlpoly
        label = '%-5s' % (stage.upper()[:5])
        # build the thermo_style list (sent to to lammps as thermos tyle commend at the end)
        thermo_style = [self.evars[n] for n in self.enames]
        thermo_style += [self.evars["epot"]]
        # this is md .. add some crucial stuff
        thermo_style += ["ke", "etotal", "temp", "press", "vol"]
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
            if wrap:
                self.lmps.command('dump %s all custom %i %s.dump id type element x y z' % (stage+"_dump", tnstep, stage))
            else:
                self.lmps.command('dump %s all custom %i %s.dump id type element xu yu zu' % (stage+"_dump", tnstep, stage))
            if self.use_uff:
                plmps_elems = self.uff_plmps_elems
            elif self.use_reaxff or self.use_xtb:
                plmps_elems = self.ff2lmp.plmps_atypes
            else:
                plmps_elems = self.ff2lmp.plmps_elems
            self.lmps.command('dump_modify %s element %s' % (stage+"_dump", " ".join(plmps_elems)))
            self.md_dumps.append(stage+"_dump")
        # do velocity startup
        if startup:
            self.lmps.command('velocity all create %12.6f %d rot yes dist gaussian' % (T1,startup_seed))
        # apply fix
        if ensemble == 'nve':
            self.md_fixes = [stage]
            self.lmps.command('fix %s all nve' % (stage))
        elif ensemble == 'nvt':
            if thermo == 'ber':
                self.lmps.command('fix %s all temp/berendsen %12.6f %12.6f %i'% (stage,T1,T2,conv_relax*relax[0]))
                self.lmps.command('fix %s_nve all nve' % stage)
                self.md_fixes = [stage, '%s_nve' % stage]
            elif thermo == 'hoover':
                self.lmps.command('fix %s all nvt temp %12.6f %12.6f %i' % (stage,T1,T2,conv_relax*relax[0]))
                self.md_fixes = [stage]
            else: 
                raise NotImplementedError
        elif ensemble == "npt":
            # this is NPT so add output of pressure and cell
            thermo_style += cellpar
            thermo_style += pressure
            if thermo == 'hoover':
                self.lmps.command('fix %s all npt temp %12.6f %12.6f %i %s %12.6f %12.6f %i' 
                        % (stage,T1,T2,conv_relax*relax[0],bcond, p1, p2, conv_relax*relax[1]))
                self.md_fixes = [stage]
            elif thermo == 'ber':
                assert bcond != "tri"
                self.lmps.command('fix %s_temp all temp/berendsen %12.6f %12.6f %i'% (stage,T1,T2,conv_relax*relax[0]))
                self.lmps.command('fix %s_press all press/berendsen %s %12.6f %12.6f %i'% (stage,bcond,p1,p2,conv_relax*relax[1]))
                self.lmps.command('fix %s_nve all nve' % stage)
                self.md_fixes = ['%s_temp' % stage,'%s_press' % stage , '%s_nve' % stage]
            elif thermo == 'mttk':
                if mttkbcond=='iso':
                    self.lmps.command('fix %s_mttknhc all mttknhc temp %8.4f %8.4f %8.4f iso %12.6f %12.6f %12.6f volconstraint %s'
                               % (stage,T1,T2,conv_relax*relax[0],p1,p2,conv_relax*relax[1],mttk_volconstraint))
                else:
                    self.lmps.command('fix %s_mttknhc all mttknhc temp %8.4f %8.4f %8.4f tri %12.6f %12.6f %12.6f volconstraint %s'
                               % (stage,T1,T2,conv_relax*relax[0],p1,p2,conv_relax*relax[1],mttk_volconstraint))
                
                self.lmps.command('fix_modify %s_mttknhc energy yes'% (stage,))
                thermo_style += ["enthalpy"]
                self.md_fixes = ['%s_mttknhc'% (stage,)]
            else:
                raise NotImplementedError
        else:
            self.pprint('WARNING: no ensemble specified (this means no fixes are set!), continuing anyway! ')
            #raise NotImplementedError
        if colvar is not None:
            self.lmps.command("fix col all colvars %s" %  colvar)
            self.md_fixes.append("col")
        # add the fix to produce the bond file in cae this is a reax calcualtion
        if self.use_reaxff or self.use_xtb:
            create_bondfile = (self.control["reaxff_bondfile"] is not None)
            if create_bondfile:
                # compute an upper estimate of the maximum number of bonds in the system
                self.nbondsmax = 0
                for e in self.get_elements():
                    self.nbondsmax += elems.maxbond[e]
                self.nbondsmax /= 2
                if self.use_reaxff:
                    self.pprint("Writing ReaxFF bondtab and bondorder to mfp5 file (nbondsmax = %d)" % self.nbondsmax)
                    self.lmps.command("fix reaxc_bnd all reax/c/bonds %d %s mfp5 %d" % \
                                          (self.control["reaxff_bondfreq"], self.control["reaxff_bondfile"], self.nbondsmax))
                    self.md_fixes.append("reaxc_bnd")
                if self.use_xtb:
                    self.pprint("Writing xTB bondtaba and bondorder to mfp5 file (nbondsmax = %d)" % self.nbondsmax)
                    #TODO

        # now define what scalar values should be written to the log file
        thermo_style += additional_thermo_output
        thermo_style += ["spcpu"]
        thermo_style_string = "thermo_style custom step " + " ".join(thermo_style)
        self.lmps.command(thermo_style_string)
        # now the thermo_style is defined and the length is known so we can setup the mfp5 dump  
        if self.mfp5 is not None:
            # ok, we will also write to the mfp5 file
            if append:
                raise IOError("TBI")
            else:
                # stage is new and we need to define it and set it up
                self.mfp5.add_stage(stage)
                if dump_thermo:
                    thermo_values = ["step"] + thermo_style
                else:
                    thermo_values = []
                self.mfp5.prepare_stage(stage, traj, tnstep, tstep=timestep/1000.0, thermo_values=thermo_values)
                # now create the dump
                traj_string = " ".join(traj + ["restart"])
                if dump_thermo:
                    traj_string += " thermo"
                if self.use_reaxff:
                    if self.control["reaxff_bondfile"] is not None:
                        # add the datasets for bondtab and bondorder to the current stage
                        self.mfp5.add_bondtab(stage, self.nbondsmax)
                        traj_string += " bond"
                # now close the hdf5 file becasue it will be written within lammps
                self.mfp5.close()
                # print("dump %s all pdlp %i %s stage %s %s" % (stage+"_pdlp", tnstep, self.pdlp.fname, stage, traj_string))
                self.lmps.command("dump %s all mfp5 %i %s stage %s %s " % (stage+"_mfp5", tnstep, self.mfp5.fname, stage, traj_string))
                self.md_dumps.append(stage+"_mfp5")
        return

    def MD_run(self, nsteps, printout=100, clear_dumps_fixes=True):
        # set restraints if there are any
        self.set_restraints()
        #assert len(self.md_fixes) > 0
        self.lmps.command('thermo %i' % printout)
        # lammps can not do runs with larger than 32 bit integer steps -> do consecutive calls
        int32max = 2147483648
        if nsteps > int32max:
            nbig = nsteps//int32max
            nrem = nsteps%int32max
            for i in range(nbig):
                self.lmps.run('run %i' % int32max)
            self.lmps.run('run %i' % nrem)
        else: 
            self.lmps.command('run %i' % nsteps)
        if clear_dumps_fixes:
            for fix in self.md_fixes:
                self.lmps.command('unfix %s' % fix)
            self.md_fixes = []
            for dump in self.md_dumps:
                self.lmps.command('undump %s' % dump)
            self.md_dumps = []
            self.lmps.command('reset_timestep 0')
        self.unset_restraint()
        return

### Johannes constD stuff

    def get_reciprocal_cell(self):
        return np.linalg.inv(self.get_cell()).T

    def set_dfield(self,field_vect, field_mask = [True, True, True], 
                    ref = None, reduced = False):
        self._use_dfield = True
        r_cell = self.get_reciprocal_cell()
        V = self.get_cell_volume()
        if ref is not None:
            pol = ref/V
            d_ref = V*np.dot(r_cell,pol)
        else:
            d_ref = np.array([0.,0.,0.])
        if reduced == False:
            d = V*np.dot(r_cell,field_vect)
        else:
            d = field_vect
        # set fix
        d += d_ref
        # format field vector
        s = ""
        for i in range(3):
            if field_mask[i] == False:
                s += "NULL "
            else:
                s += "%12.6f " % d[i]
        print(s)
        # setup polarization lammps variables
        self.lmps.command("fix initconfig all store/state 0 xu yu zu")
        self.lmps.command("compute displ all displace/atom")
        self.lmps.command("variable OmegaPxi atom (c_displ[1]+f_initconfig[1])*q")
        self.lmps.command("variable OmegaPyi atom (c_displ[2]+f_initconfig[2])*q")
        self.lmps.command("variable OmegaPzi atom (c_displ[3]+f_initconfig[3])*q")
        self.lmps.command("compute OmegaPx all reduce sum v_OmegaPxi")
        self.lmps.command("compute OmegaPy all reduce sum v_OmegaPyi")
        self.lmps.command("compute OmegaPz all reduce sum v_OmegaPzi")
        self.add_ename("dfield", "f_dfield")
        self.lmps.command("fix dfield all dfield %s c_OmegaPx c_OmegaPy c_OmegaPz energy f_dfield" % s)
        self.lmps.command("fix_modify dfield energy yes")
        self.lmps.command("fix_modify dfield virial yes")
        return

 #       for expot, callback_name in self.external_pot:
 #           # run the expot's setup with self as an argument --> you have access to all info within the mol object
 #           expot.setup(self)
 #           fix_id = "expot_"+expot.name
 #           self.add_ename(expot.name, "f_"+fix_id)
 #           self.lmps.command("fix %s all python/invoke 1 post_force %s" % (fix_id, callback_name))
 #           self.lmps.command("fix_modify %s energy yes" % fix_id)
 #           self.pprint("External Potential %s is set up as fix %s" % (expot.name, fix_id))

    def unset_dfield(self):
        assert self._use_dfield == True
        self.lmps.command("unfix Densemble")
        return



#### NEB implementation (beta)

    def NEB(self, final, ftol, nsteps=5000, nsteps_climb=None, Nevery= 100, K=1.0, \
                    min_style="quickmin", fix_ends=True, group=None, Kperp=2.0):
        assert min_style in ("quickmin", "fire")
        assert self.partitions is not None, "To use NEB you need to request partitions when starting up"
        self.pprint("NEB calculation with %d beads" % self.partitions)
        # make sure that the final mol object is matching with the existing mol object
        assert self.mol.natoms == final.natoms
        assert self.mol.elems == final.elems
        # settings:
        if nsteps_climb is None:
            nsteps_climb = nsteps
        # interpolate structures between inital an final state in a linear way
        initl_xyz = self.mol.get_xyz()
        final_xyz = final.get_xyz()
        if self.part_num > 0:
            my_xyz = initl_xyz + (self.part_num/(self.partitions-1)) * (final_xyz - initl_xyz)
            self.set_xyz(my_xyz)
        self.write_part(self.name + "_neb_init.xyz")
        # now set min_style and fix
        if group is None:
            neb_group = "all"
        else:
            atom_frm = len(group)* " %d"
            self.lmps.command(("group neb id " + atom_frm) % tuple([i+1 for i in group]))
            print (("group neb id " + atom_frm) % tuple([i+1 for i in group]))
            neb_group = "neb"
        self.lmps.command("fix neb %s neb %10.3f perp %10.3f" % (neb_group, K, Kperp))
        self.lmps.command("min_style %s" % min_style)
        # if fix_ends is true set forces of endpoints to zero (note: lammps counts partitions from 1)
        if fix_ends:
            self.lmps.command("partition yes 1  fix freeze_init all setforce 0.0 0.0 0.0")
            self.lmps.command("partition yes %d fix freeze_init all setforce 0.0 0.0 0.0" % self.partitions)
        # run it
        self.lmps.command("neb 0.0 %6.3f %d %d %d none" % (ftol, nsteps, nsteps_climb, Nevery))
        # unfix
        self.lmps.command("unfix neb")
        # self.write("neb_final.xyz", partitions=True)
        self.write_part(self.name + "_neb_final.xyz")
        return

 






