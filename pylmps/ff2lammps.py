# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 09:37:25 2017

@author: rochus


              ff2lammps
              
class to be instantiated with an exisiting mol object and paramters already assinged
it will write a data and a lamps input file              

"""

import numpy as np
import string
import copy

import molsys
import molsys.util.elems as elements
from molsys.addon import base
from molsys.util.timing import timer, Timer

import logging
logger = logging.getLogger('molsys.ff2lammps')


mdyn2kcal = 143.88
angleunit = 0.02191418
rad2deg = 180.0/np.pi 

from .util import rotate_cell


class ff2lammps(base):
    
    def __init__(self, mol,setup_FF=True, reax=False, print_timer=True):
        """
        setup system and get parameter 
        
        :Parameters:
        
            - mol: mol object with ff addon and params assigned
            - setup_FF [bool]: defaults to True, skip FF setup when False
            - reax [bool]: defaults to False: if True then ReaxFF is used

        In case of ReaxFF no ff addon is present and no bonds/angles/dihedrals/oops are written
        The atom_style is charge (no mol-ID)

        """
        super(ff2lammps,self).__init__(mol)
        # generate a timer
        self.timer = Timer(name = "ff2lammps")
        self.timer.start("init")
        self.print_timer = print_timer
        # generate the force field
        if setup_FF != True:
            return
        # general settings                
        self._settings = {}
        # set defaults
        self._settings["cutoff"] = 12.0
        self._settings["cutoff_coul"] = None
        self._settings["parformat"] = "%15.8g"
        self._settings["vdw_a"] = 1.84e5
        self._settings["vdw_b"] = 12.0
        self._settings["vdw_c"] = 2.25
        self._settings["vdw_dampfact"] = 0.25
        self._settings["vdw_smooth"] = 0.9
        self._settings["coul_smooth"] = 0.9
        self._settings["use_angle_cosine_buck6d"] = True
        self._settings["kspace_method"] = "ewald"
        self._settings["kspace_prec"] = 1.0e-6
        self._settings["use_improper_umbrella_harmonic"] = False # default is to use improper_inversion_harmonic
        self._settings["origin"] = "zero"
        # init some basic stuff always needed
        self.ricnames = ["bnd", "ang", "dih", "oop", "cha", "vdw"]
        self.nric = {}
        self.par_types = {}
        self.rics = {}
        for r in self.ricnames:
            self.nric[r] = 0
            self.par_types[r] = {}
            self.rics[r] = {}
        self.reax=False
        if reax:
            self.reax = True
            self.plmps_atypes = list(set(self._mol.get_elems()))
            self.plmps_atypes.sort()
            self.plmps_mass = {}
            for at in self.plmps_atypes:
                self.plmps_mass[at] = elements.mass[at]
            self.timer.stop()
            return
        self.timer.start("setup pair pots")
        self._mol.ff.setup_pair_potentials()
        self.timer.stop()
        # set up the molecules
        self.timer.start("molecules addon")
        self._mol.addon("molecules")
        self._mol.molecules()
        self.timer.stop()
        # make lists of paramtypes and conenct to mol.ff obejcts as shortcuts
        self.par = {}
        self.parind = {}
        self.npar = {}
        for r in self.ricnames:
            self.par[r]       = self._mol.ff.par[r]
            self.parind[r]    = self._mol.ff.parind[r]
            self.rics[r]      = self._mol.ff.ric_type[r]
            # sort identical parameters (sorted) using a tuple to hash it into the dict par_types : value is a number starting from 1
            par_types = {}
            i = 1
            iric = 0
            self.timer.start("par loop %s" % r)
            for pil in self.parind[r]:
                if pil:
                    pil.sort()
                    tpil = tuple(pil)
                    # we have to check if we have none potentials in the par structure, then we have to remove them
                    if len(tpil) == 1 and self.par[r][tpil[0]][0] == 'none': 
                        continue
                    else:
                        iric += 1
                    if not tpil in par_types:
                        par_types[tpil] = i
                        i += 1
            self.timer.stop()
            self.par_types[r] = par_types
            self.npar[r] = i-1
            self.nric[r] = iric
        # map additional nonbonded types
        self.par["vdwpr"] = self._mol.ff.par["vdwpr"]
        self.par["chapr"] = self._mol.ff.par["chapr"]
        # we need to verify that the vdw types and the charge types match because the sigma needs to be in the pair_coeff for lammps
        # thus we build our own atomtypes list combining vdw and cha and use the mol.ff.vdwdata as a source for the combined vdw params
        # but add the combined 1.0/sigma_ij here
        self.plmps_atypes = []
        self.plmps_elems = []
        self.plmps_pair_data = {}
        self.plmps_mass = {} # mass from the element .. even if the vdw and cha type differ it is still the same atom
        for i in range(self._mol.get_natoms()):
            vdwt = self.parind["vdw"][i][0]
            chrt = self.parind["cha"][i][0]
            at = vdwt+"/"+chrt
            if not at in self.plmps_atypes:
                #print("new atomtype %s" % at)
                self.plmps_atypes.append(at)
                #self.plmps_elems.append(self._mol.elems[i].title()+str(len(self.plmps_atypes)))
                self.plmps_elems.append(self._mol.elems[i].title())
                # extract the mass ...
                etup = vdwt.split("->")[1].split("|")[0]
                etup = etup[1:-2]
                e = etup.split("_")[0]
                e = [x for x in e if x.isalpha()]
                #self.plmps_mass[at] = elements.mass[e]
                try:
                    self.plmps_mass[at] = elements.mass[self.plmps_elems[-1].lower()]
                except:
                    self.plmps_mass[at] = 1.0
                #print("with mass %12.6f" % elements.mass[e])
#        for i, ati in enumerate(self.plmps_atypes):
#            for j, atj in enumerate(self.plmps_atypes[i:],i):
#                vdwi, chai = ati.split("/")
#                vdwj, chaj = atj.split("/")
#                vdwpairdata = self._mol.ff.vdwdata[vdwi+":"+vdwj]
#                sigma_i = self.par["cha"][chai][1][1]
#                sigma_j = self.par["cha"][chaj][1][1]
#                # compute sigma_ij
#                sigma_ij = np.sqrt(sigma_i*sigma_i+sigma_j*sigma_j)
#                # vdwpairdata is (pot, [rad, eps])
#                pair_data = []
#                pair_data.append(vdwpairdata)
#                #pair_data = copy.copy(vdwpairdata[1])
#                pair_data.append(1.0/sigma_ij)
#                self.plmps_pair_data[(i+1,j+1)] = pair_data
        # add settings from ff addon
        for k,v in list(self._mol.ff.settings.items()):
            self._settings[k]=v
        if self._settings["chargetype"]=="gaussian":
            assert self._settings["vdwtype"]=="exp6_damped" or self._settings["vdwtype"]=="wangbuck"
        self.timer.stop()
        return 

    def adjust_cell(self):
        if self._mol.bcond > 0:
            fracs = self._mol.get_frac_xyz()
            cell  = self._mol.get_cell()
            self.tilt = 'small'
            # now check if cell is oriented along the (1,0,0) unit vector
            if np.linalg.norm(cell[0]) != cell[0,0]:
                rcell = rotate_cell(cell)
                self._mol.set_cell(rcell, cell_only=False)
            else:
                rcell = cell
            lx,ly,lz,xy,xz,yz = rcell[0,0],rcell[1,1],rcell[2,2],rcell[1,0],rcell[2,0],rcell[2,1]
                # system needs to be rotated
#                rcell=np.zeros([3,3])
#                A = cell[0]
#                B = cell[1]
#                C = cell[2]
#                AcB = np.cross(A,B)
#                uAcB = AcB/np.linalg.norm(AcB)
#                lA = np.linalg.norm(A)
#                uA = A/lA
#                lx = lA
#                xy = np.dot(B,uA)
#                ly = np.linalg.norm(np.cross(uA,B))
#                xz = np.dot(C,uA)
#                yz = np.dot(C,np.cross(uAcB,uA))
#                lz = np.dot(C,uAcB)
                # check for tiltings
            if abs(xy)>lx/2: 
                #logger.warning('xy tilting is too large in respect to lx')
                self.tilt='large'
            if abs(xz)>lx/2: 
                #logger.warning('xz tilting is too large in respect to lx')
                self.tilt='large'
            if abs(yz)>lx/2: 
                #logger.warning('yz tilting is too large in respect to lx')
                self.tilt='large'
            if abs(xz)>ly/2: 
                #logger.warning('xz tilting is too large in respect to ly')
                self.tilt='large'
            if abs(yz)>ly/2:
                #logger.warning('yz tilting is too large in respect to ly')
                self.tilt='large'
            # check if celldiag is positve, else a left hand side basis is formed
            if rcell.diagonal()[0]<0.0: raise IOError('Left hand side coordinate system detected')
            if rcell.diagonal()[1]<0.0: raise IOError('Left hand side coordinate system detected')
            if rcell.diagonal()[2]<0.0: raise IOError('Left hand side coordinate system detected')
#            self._mol.set_cell(rcell, cell_only=False)
#                import pdb; pdb.set_trace()
        return

    @staticmethod
    def cell2tilts(cell):
        return [cell[0,0],cell[1,1],cell[2,2],cell[1,0],cell[2,0],cell[2,1]]


    def setting(self, s, val):
        if not s in self._settings:
            self.pprint("This settings %s is not allowed" % s)
            return
        else:
            self._settings[s] = val
            return
        
    @timer("write data")
    def write_data(self, filename="tmp.data"):
        if self.mpi_rank > 0: 
            # wait on master to finish
            self.mpi_comm.barrier()
            return
        self.data_filename = filename
        f = open(filename, "w")
        # write header 
        if self.reax:
            header = "LAMMPS data file for mol object using ReaxFF\n\n"
        else: 
            header = "LAMMPS data file for mol object with MOF-FF params from www.mofplus.org\n\n"
        header += "%10d atoms\n"      % self._mol.get_natoms()
        if self.nric['bnd'] != 0: header += "%10d bonds\n"      % self.nric['bnd']
        if self.nric['ang'] != 0: header += "%10d angles\n"     % self.nric['ang']
        if self.nric['dih'] != 0: header += "%10d dihedrals\n"  % self.nric['dih']
        if self.nric['oop'] != 0:
            if self._settings["use_improper_umbrella_harmonic"] == True:
                header += "%10d impropers\n"  % (self.nric['oop']*3) # need all three permutations
            else:
                header += "%10d impropers\n"  % self.nric['oop']            
        # types are different paramtere types 
        header += "%10d atom types\n"       % len(self.plmps_atypes)
        if len(self.par_types["bnd"]) != 0: header += "%10d bond types\n"       % len(self.par_types["bnd"]) 
        if len(self.par_types["ang"]) != 0: header += "%10d angle types\n"      % len(self.par_types["ang"])
        if len(self.par_types["dih"]) != 0: header += "%10d dihedral types\n"   % len(self.par_types["dih"])
        if len(self.par_types["oop"]) != 0: header += "%10d improper types\n\n" % len(self.par_types["oop"])
        self.adjust_cell()
        xyz = self._mol.get_xyz()
        if self._mol.bcond == 0:
            # in the nonperiodic case center the molecule in the origin
            # JK: Lammps wants a box also in the non-periodic (free) case.
            self._mol.periodic=False
            self._mol.translate(-self._mol.get_com())
            cmax = xyz.max(axis=0)+10.0
            cmin = xyz.min(axis=0)-10.0
            tilts = (0.0,0.0,0.0)
        elif self._mol.bcond<2:
            # orthorombic/cubic bcondq
            cell = self._mol.get_cell()
            if self._settings["origin"] == "zero":
                cmax = cell.diagonal()
                cmin = np.zeros([3])
            else:
                cmax = cell.diagonal()*0.5
                cmin = -cmax
            tilts = (0.0,0.0,0.0)
        else:
            # triclinic bcond
            cell = self._mol.get_cell()
            if self._settings["origin"] == "zero":
                cmin = np.zeros([3])
                cmax = cell.diagonal()
            else:
                cmax = cell.diagonal()*0.5
                cmin = -cmax
            tilts = (cell[1,0], cell[2,0], cell[2,1])
        if self._mol.bcond >= 0:
            header += '%12.6f %12.6f  xlo xhi\n' % (cmin[0], cmax[0])
            header += '%12.6f %12.6f  ylo yhi\n' % (cmin[1], cmax[1])
            header += '%12.6f %12.6f  zlo zhi\n' % (cmin[2], cmax[2])
        if self._mol.bcond > 2:
            header += '%12.6f %12.6f %12.6f  xy xz yz\n' % tilts
        # NOTE in lammps masses are mapped on atomtypes which indicate vdw interactions (pair potentials)
        #   => we do NOT use the masses set up in the mol object because of this mapping
        #   so we need to extract the element from the vdw paramter name which is a bit clumsy (DONE IN INIT NOW)
        header += "\nMasses\n\n"        
        for i, at in enumerate(self.plmps_atypes):
            header += "%5d %10.4f # %s\n" % (i+1, self.plmps_mass[at], at)
        f.write(header)
        # write Atoms
        # NOTE ... this is MOF-FF and we silently assume that all charge params are Gaussians!!
        f.write("\nAtoms\n\n")
        if self.reax:
            elems = self._mol.get_elems()
            for i in range(self._mol.get_natoms()):
                at = elems[i]
                atype = self.plmps_atypes.index(at)+1
                x,y,z = xyz[i]
                # for reaxff chrg = 0.0 becasue it is set by Qeq
                #   ind  atype chrg x y z # comment
                chrg = 0.0
                f.write("%10d %5d %12.8f %12.6f %12.6f %12.6f\n" % (i+1, atype, chrg, x,y,z))
        else:
            charges = self.get_charges()
            # write atoms with charges
            for i in range(self._mol.get_natoms()):
                vdwt  = self.parind["vdw"][i][0]
                chat  = self.parind["cha"][i][0]
                at = vdwt+"/"+chat
                atype = self.plmps_atypes.index(at)+1
                molnumb = self._mol.molecules.mgroups["molecules"].whichmol[i]+1
                x,y,z = xyz[i]
                chrg  = charges[i]
                #   ind  atype molnumb chrg x y z # comment
                f.write("%10d %5d %5d %12.8f %12.6f %12.6f %12.6f # %s\n" % (i+1, molnumb, atype, chrg, x,y,z, vdwt))
            chargesum = charges.sum()
            if abs(chargesum) > 1e-8:
              self.pprint("The total charge of the system is: %12.8f" % chargesum)
        # write bonds
        if len(self.rics["bnd"]) != 0: f.write("\nBonds\n\n")
        for i in range(len(self.rics["bnd"])):
            bndt = tuple(self.parind["bnd"][i])
            a,b  = self.rics["bnd"][i]
            if bndt in list(self.par_types['bnd'].keys()):
                f.write("%10d %5d %8d %8d  # %s\n" % (i+1, self.par_types["bnd"][bndt], a+1, b+1, bndt))
        # write angles
        if len(self.rics["ang"]) != 0: f.write("\nAngles\n\n")
        for i in range(len(self.rics["ang"])):
            angt = tuple(self.parind["ang"][i])
            a,b,c  = self.rics["ang"][i]
            if angt in list(self.par_types['ang'].keys()):
                f.write("%10d %5d %8d %8d %8d  # %s\n" % (i+1, self.par_types["ang"][angt], a+1, b+1, c+1, angt))
        # write dihedrals
        if len(self.rics["dih"]) != 0: f.write("\nDihedrals\n\n")
        for i in range(len(self.rics["dih"])):
            diht = tuple(self.parind["dih"][i])
            a,b,c,d  = self.rics["dih"][i]
            if diht in list(self.par_types['dih'].keys()):
                f.write("%10d %5d %8d %8d %8d %8d # %s\n" % (i+1, self.par_types["dih"][diht], a+1, b+1, c+1, d+1, diht))
        # write impropers/oops
        if len(self.rics["oop"]) != 0: f.write("\nImpropers\n\n")
        for i in range(len(self.rics["oop"])):            
            oopt = tuple(self.parind["oop"][i])
            if oopt:
                a,b,c,d  = self.rics["oop"][i]
                if oopt in list(self.par_types['oop'].keys()):
                    f.write("%10d %5d %8d %8d %8d %8d # %s\n" % (i+1, self.par_types["oop"][tuple(oopt)], a+1, b+1, c+1, d+1, oopt))
                    if self._settings["use_improper_umbrella_harmonic"] == True:
                        # add the other two permutations of the bended atom (abcd : a is central, d is bent)
                        f.write("%10d %5d %8d %8d %8d %8d # %s\n" % (i+1, self.par_types["oop"][tuple(oopt)], a+1, d+1, b+1, c+1, oopt))
                        f.write("%10d %5d %8d %8d %8d %8d # %s\n" % (i+1, self.par_types["oop"][tuple(oopt)], a+1, c+1, d+1, b+1, oopt))
        f.write("\n")
        f.close()
        # sync with other nodes
        self.mpi_comm.barrier()
        return

    def parf(self, n):
        pf = self._settings["parformat"]+" "
        return n*pf

    @timer("write 2 internal")
    def write2internal(self,lmps,pair = False, charge=False):
        if pair:
            pstrings = self.pairterm_formatter()
            for p in pstrings: lmps.lmps.command(p)
        if charge:
            pstrings = self.charge_formatter()
            for p in pstrings: lmps.lmps.command(p)
        formatter = {"bnd": self.bondterm_formatter,
                "ang": self.angleterm_formatter,
                "dih": self.dihedralterm_formatter,
                "oop": self.oopterm_formatter}
        for ict in ['bnd','ang','dih','oop']:
            for bt in list(self.par_types[ict].keys()):
                bt_number = self.par_types[ict][bt]
                for ibt in bt:
                    pot_type, params = self.par[ict][ibt]
                    pstrings = formatter[ict](bt_number, pot_type, params)
                    for p in pstrings: lmps.lmps.command(p)
        return

    # RS (Nov 2020) revision of charges -> add delta charges
    #      works for write_data where not just chrges are printed
    #      the follwoing routines are for fitting and only set charges 
    #      some obsolete code is reomved
    def charge_formatter(self):
        pstrings = []
        charges = self.get_charges()
        for i in range(self._mol.get_natoms()):
            pstrings.append("set atom %5d charge %12.8f" % (i+1, charges[i]))
        return pstrings

    def get_charges(self):
        charges = np.zeros(self._mol.get_natoms()) # we need to compute the array first and only then we can write it out
        # set up the delta charges dictionary
        delta_chrg = {}
        for k in self.par["chapr"]:
            if self.par["chapr"][k][0] == "delta":
                delta = self.par["chapr"][k][1][0]
                at1, at2 = k.split("(")[1].split(")")[0].split(",")
                delta_chrg[at1] = (at2, delta)
        # compute charges
        conn = self._mol.get_conn()
        for i in range(self._mol.get_natoms()):
            chat  = self.parind["cha"][i][0]
            chrgpar    = self.par["cha"][chat]
            assert chrgpar[0] == "gaussian", "Only Gaussian type charges supported"   # also "point" should work -> To be checked
            charges[i] += chrgpar[1][0]
            # check if chat in delta_chrg
            chat_red = chat.split("(")[1].split(")")[0]
            if chat_red in delta_chrg:
                at2, delta = delta_chrg[chat_red]
                # chek if any of the bonded atoms is of type at2
                for j in conn[i]:
                    if repr(self._mol.ff.ric.aftypes[j]) == at2:   # Note: aftypes are aftype objects and not strings .. we call repr() to get the string
                        # print ("atom %s connected to atom %s ..apply delta %f" % (chat_red, at2, delta))
                        charges[i] += delta
                        charges[j] -= delta
        return charges




    def pairterm_formatter(self,comment = False):
        # this method behaves different thant the other formatters because it
        # performs a loop over all pairs
        # recompute pairdata by the combination rules
        self._mol.ff.setup_pair_potentials()
        pstrings = []
        #TODO recompute pair data before
        for i, ati in enumerate(self.plmps_atypes):
            for j, atj in enumerate(self.plmps_atypes[i:],i):
                # compute the pair data relevant stuff directly here
                vdwi, chai = ati.split("/")
                vdwj, chaj = atj.split("/")
                vdw = self._mol.ff.vdwdata[vdwi+":"+vdwj]
                if self._settings["chargetype"] == "gaussian":
                    sigma_i = self.par["cha"][chai][1][1]
                    sigma_j = self.par["cha"][chaj][1][1]
                    # compute sigma_ij
                    alpha_ij = 1.0/np.sqrt(sigma_i*sigma_i+sigma_j*sigma_j)
                if self._settings["vdwtype"]=="exp6_damped":
                    if vdw[0] == "buck6d":
                        r0, eps = vdw[1]
                        A = self._settings["vdw_a"]*eps
                        B = self._settings["vdw_b"]/r0
                        C = eps*self._settings["vdw_c"]*r0**6
                        D = 6.0*(self._settings["vdw_dampfact"]*r0)**14
                        if "pressure_bath_atype" in self._settings.keys():
                            a1 = ati.count(self._settings["pressure_bath_atype"]) > 0
                            a2 = atj.count(self._settings["pressure_bath_atype"]) > 0
                            if (a1 and not a2) or (not a1 and a2):
                                print ("DEBUG : one is a pressure bath atom %s %s" % (ati, atj))
                                C = 0.0
                                D = 0.0

                        #pstrings.append(("pair_coeff %5d %5d " + self.parf(5) + "   # %s <--> %s) % (i+1,j+1, A, B, C, D, alpha_ij, ati, atj))
                        #f.write(("pair_coeff %5d %5d " + self.parf(5) + "   # %s <--> %s\n") % (i+1,j+1, A, B, C, D, alpha_ij, ati, atj))
                    elif vdw[0] == "buck":
                        A,B,C = vdw[1]
                        D = 0.
                    elif vdw[0] == "buck6de":
                        A,B,C,D = vdw[1]
                    elif vdw[0] == "lbuck":
                        sigma, epsilon, gamma = vdw[1]
                        A = 6*epsilon*np.exp(gamma)/(gamma-6)
                        B = gamma/sigma
                        C = gamma*epsilon*sigma**6/(gamma-6)
                        D = 0.
                    else:
                        raise ValueError("unknown pair potential")
                    if comment:
                        pstrings.append(("pair_coeff %5d %5d " + self.parf(5) + "   # %s <--> %s") % (i+1,j+1, A, B, C, D, alpha_ij, ati, atj))
                    else:
                        pstrings.append(("pair_coeff %5d %5d " + self.parf(5)) % (i+1,j+1, A, B, C, D, alpha_ij))
                elif self._settings["vdwtype"] == "wangbuck":
                    if vdw[0]=="wbuck":
                        A,B,C = vdw[1]
                    else:
                        raise ValueError("unknown pair potential")
                    if comment:
                        pstrings.append(("pair_coeff %5d %5d " + self.parf(4) + "   # %s <--> %s") % (i+1,j+1, A, B, C, alpha_ij, ati, atj))
                    else:
                        pstrings.append(("pair_coeff %5d %5d " + self.parf(4)) % (i+1,j+1, A, B, C, alpha_ij))
                elif self._settings["vdwtype"]=="buck":
                    if vdw[0] == "buck":
                        A,B,C = vdw[1]
                        B=1./B
                    elif vdw[0] == "lbuck":
                        sigma, epsilon, gamma = vdw[1]
                        A = 6*epsilon*np.exp(gamma)/(gamma-6)
                        B = gamma/sigma
                        C = gamma*epsilon*sigma**6/(gamma-6)
                        D = 0.
                        B = 1./B
                    elif vdw[0] =="mm3":
                        r0, eps = vdw[1]
                        A = self._settings["vdw_a"]*eps
                        B = self._settings["vdw_b"]/r0
                        B = 1./B
                        C = eps*self._settings["vdw_c"]*r0**6
                    else:
                        raise ValueError("unknown pair potential")
                    if comment:
                        pstrings.append(("pair_coeff %5d %5d " + self.parf(3) + "   # %s <--> %s") % (i+1,j+1, A, B, C, ati, atj))
                    else:
                        pstrings.append(("pair_coeff %5d %5d " + self.parf(3)) % (i+1,j+1, A, B, C))
                elif self._settings["vdwtype"]=="lj_12_6":
                  if vdw[0] == "lj_12_6":
                    #if len(vdw) <= 3:
                    r0 , eps = vdw[1]
                    sig = r0/(2.0**(1.0/6.0))
                    if comment:
                      pstrings.append(("pair_coeff %5d %5d " + self.parf(3) + "   # %s <--> %s") % (i+1,j+1, eps, sig, alpha_ij, ati, atj))
                    else:
                      pstrings.append(("pair_coeff %5d %5d " + self.parf(3)) % (i+1,j+1, eps, sig, alpha_ij))
                  else:
                    pstrings.append(("pair_coeff %5d %5d " + self.parf(3)) % (i+1,j+1, A, B, C))
                else:
                    raise ValueError("unknown pair setting")
        return pstrings



    def bondterm_formatter(self, number, pot_type, params):
        assert type(params) == list
        if np.count_nonzero(params) == 0:
            #TODO implement used feature here, quick hack would be to make one dry run
            # startup with ff2pydlpoly and get the info form there :D
            pass 
        if pot_type == "mm3":
            r0 = params[1]
            K2 = params[0]*mdyn2kcal/2.0 
            K3 = K2*(-2.55)
            K4 = K2*(2.55**2.)*(7.0/12.0)
            pstring = "bond_coeff %5d class2 %12.6f %12.6f %12.6f %12.6f" % (number,r0, K2, K3, K4)
        elif pot_type == "quartic":
            r0 = params[1]
            K2 = params[0]*mdyn2kcal/2.0 
            K3 = -1*K2*params[2]
            K4 = K2*(2.55**2.)*params[3]
            pstring = "bond_coeff %5d class2 %12.6f %12.6f %12.6f %12.6f" % (number,r0, K2, K3, K4)
        elif pot_type == "harm":
            r0 = params[1]
            K2 = params[0]*mdyn2kcal/2.0 
            pstring = "bond_coeff %5d harmonic %12.6f %12.6f" % (number,K2, r0)
        elif pot_type == "morse":
            r0 = params[1]
            E0 = params[2]
            k  = params[0]*mdyn2kcal/2.0
            alpha = np.sqrt(k/E0)
            pstring = "bond_coeff %5d morse %12.6f%12.6f %12.6f" % (number, E0, alpha, r0)
        else:
            raise ValueError("unknown bond potential")
        return [pstring]

    def angleterm_formatter(self, number, pot_type, params):
        assert type(params) == list
        pstrings = []
        if np.count_nonzero(params) == 0:
            #TODO implement used feature here, quick hack would be to make one dry run
            # startup with ff2pydlpoly and get the info form there :D
            pass
        if pot_type == "mm3":
            th0 = params[1]
            K2  = params[0]*mdyn2kcal/2.0 
            K3 = K2*(-0.014)*rad2deg
            K4 = K2*5.6e-5*rad2deg**2
            K5 = K2*-7.0e-7*rad2deg**3
            K6 = K2*2.2e-8*rad2deg**4
            pstring = "angle_coeff %5d class2/p6 %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f" % (number,th0, K2, K3, K4, K5, K6)
            pstrings.append(pstring)
            # HACk to catch angles witout strbnd
#            if len(at) == 1:
#                pstrings.append("angle_coeff %5d class2/p6 bb 0.0 1.0 1.0" % (number))
#                pstrings.append("angle_coeff %5d class2/p6 ba 0.0 0.0 1.0 1.0" % (number))
        elif pot_type == "strbnd":
            ksb1, ksb2, kss = params[:3]
            r01, r02        = params[3:5]
            th0             = params[5]
            pstrings.append("angle_coeff %5d class2/p6 bb %12.6f %12.6f %12.6f" % (number, kss*mdyn2kcal, r01, r02))
            pstrings.append("angle_coeff %5d class2/p6 ba %12.6f %12.6f %12.6f %12.6f" % (number, ksb1*mdyn2kcal, ksb2*mdyn2kcal, r01, r02))
            # f.write("angle_coeff %5d bb %12.6f %12.6f %12.6f\n" % (at_number, kss*mdyn2kcal, r01, r02))
            # f.write("angle_coeff %5d ba %12.6f %12.6f %12.6f %12.6f\n" % (at_number, ksb1*mdyn2kcal, ksb2*mdyn2kcal, r01, r02))
        elif pot_type == "fourier":
            a0 = params[1]
            fold = params[2]
            k = 0.5*params[0]*angleunit*rad2deg*rad2deg/fold
            pstring = "%12.6f %5d %12.6f" % (k, fold, a0)
            if self._settings["use_angle_cosine_buck6d"]:
                pstrings.append("angle_coeff %5d cosine/buck6d   %s" % (number, pstring))                   
            else:
                pstrings.append("angle_coeff %5d cosine/vdwl13   %s 1.0" % (number, pstring))
        else:
            raise ValueError("unknown angle potential")
        return pstrings

#                    pstring = "%12.6f %12.6f %12.6f %12.6f %12.6f %12.6f" % (th0, K2, K3, K4, K5, K6)
#                    # pstring = "%12.6f %12.6f" % (th0, K2)
#                    f.write("angle_coeff %5d class2/p6    %s    # %s\n" % (at_number, pstring, iat))



    def dihedralterm_formatter(self, number, pot_type, params):
        if np.count_nonzero(params) == 0:
            #TODO implement used feature here, quick hack would be to make one dry run
            # startup with ff2pydlpoly and get the info form there :D
            pass
        if pot_type == "cos3":
            v1, v2, v3 = params[:3]
            pstring = "opls %12.6f %12.6f %12.6f %12.6f" % (v1, v2, v3, 0.0)
        elif pot_type == "cos4":
            v1, v2, v3, v4 = params[:4]
            pstring = "opls %12.6f %12.6f %12.6f %12.6f" % (v1, v2, v3, v4)
        elif pot_type == "class2":
            v1, v2, v3, v4, v5, v6 = params[:6]
            pstring = "class2 %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f" % (v1/2., v2, v3/2., v4, v5/2., v6)
        elif pot_type == "bb13":
            kss, r1, r3 = params[:3]
            pstring = "class2 bb13 %12.6f %12.6f %12.6f" % (kss*mdyn2kcal, r1, r3)
        elif pot_type == "aat":
            kbb, th1, th2 = params[:3]
            pstring = "class2 aat  %12.6f %12.6f %12.6f" % (kbb*mdyn2kcal, th1, th2)
        elif pot_type == "at":
            kb1_1, kb1_2, kb1_3, kb2_1, kb2_2, kb2_3, th1, th2 = params[:8]
            pstring = "class2 at   %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f" % (kb1_1*mdyn2kcal,
                      kb1_2*mdyn2kcal, kb1_3*mdyn2kcal, kb2_1*mdyn2kcal, kb2_2*mdyn2kcal, kb2_3*mdyn2kcal, th1, th2)
        elif pot_type == "ebt":
            kb1_1, kb1_2, kb1_3, kb2_1, kb2_2, kb2_3, r1, r3 = params[:8]
            pstring = "class2 ebt  %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f" % (kb1_1*mdyn2kcal,
                      kb1_2*mdyn2kcal, kb1_3*mdyn2kcal, kb2_1*mdyn2kcal, kb2_2*mdyn2kcal, kb2_3*mdyn2kcal, r1, r3)
        elif pot_type == "mbt":
            kb1, kb2, kb3, r2 = params[:4]
            pstring = "class2 mbt  %12.6f %12.6f %12.6f %12.6f" % (kb1*mdyn2kcal, kb2*mdyn2kcal, kb3*mdyn2kcal, r2)
        else:
            raise ValueError("unknown dihedral potential")
        return ["dihedral_coeff %5d %s" % (number, pstring)]


    def oopterm_formatter(self, number, pot_type, params):
        if np.count_nonzero(params) == 0:
            #TODO implement used feature here, quick hack would be to make one dry run
            # startup with ff2pydlpoly and get the info form there :D
            pass
        if pot_type == "harm":
            pstring = "%12.6f %12.6f" % (params[0]*mdyn2kcal*1.5, params[1])
        else:
            raise ValueError("unknown improper/oop potential")
        return ["improper_coeff %5d %s" % (number, pstring)]


    @timer("write input")
    def write_input(self, filename = "lmp.input", header=None, footer=None, kspace=False, noheader=False, boundary=None):
        """
        NOTE: add read data ... fix header with periodic info
        """
        assert self.reax != True, "Not to be used with ReaxFF"
        if self.mpi_rank > 0: 
            # wait for master to finish
            self.mpi_comm.barrier()
            return
        self.input_filename = filename
        f = open(filename, "w")
        # write standard header
        if not noheader:      
            f.write("clear\n")
            f.write("units real\n")
            if boundary:
                assert len(boundary)==3
            else:
                if self._mol.bcond == 0:
                    boundary = ("f", "f", "f")
                else:
                    boundary = ("p", "p", "p")
            f.write("boundary %s %s %s\n" % tuple(boundary))
            f.write("atom_style full\n")
            if self._mol.bcond > 2:
                f.write('box tilt large\n')
            f.write("read_data %s\n\n" % self.data_filename)
            f.write("neighbor 2.0 bin\n\n")
            # extra header
            if header:
                hf = open(header, "r")
                f.write(hf.readlines())
                hf.close()
        f.write("\n# ------------------------ MOF-FF FORCE FIELD ------------------------------\n")
        # pair style
        if kspace:
            if not noheader:
                # use kspace for the long range electrostatics and the corresponding long for the real space pair
                f.write("\nkspace_style %s %10.4g\n" % (self._settings["kspace_method"], self._settings["kspace_prec"]))
            # for DEBUG f.write("kspace_modify gewald 0.265058\n")
            if self._mol.ff.settings["coreshell"] == True:
                if self._settings["vdwtype"] == "buck" and self._settings["chargetype"] == "point":
                    f.write("pair_style buck/coul/long/cs %10.4f\n\n" % (self._settings["cutoff"]))
                elif self._settings["vdwtype"] == "wangbuck" and self._settings["chargetype"] == "gaussian":
                    f.write("pair_style wangbuck/coul/gauss/long/cs %10.4f %10.4f %10.4f\n\n" %
                        (self._settings["vdw_smooth"], self._settings["coul_smooth"], self._settings["cutoff"]))
            elif self._settings["vdwtype"] == "wangbuck":
                 f.write("pair_style wangbuck/coul/gauss/long %10.4f %10.4f %10.4f\n\n" % 
                    (self._settings["vdw_smooth"], self._settings["coul_smooth"], self._settings["cutoff"]))
            elif (self._settings["chargetype"] == "gaussian") & ((self._settings["vdwtype"] == "buck6d")|(self._settings["vdwtype"] == "exp6_damped")):
                f.write("pair_style buck6d/coul/gauss/long %10.4f %10.4f %10.4f\n\n" % 
                    (self._settings["vdw_smooth"], self._settings["coul_smooth"], self._settings["cutoff"]))
            elif (self._settings["chargetype"] == "gaussian") & (self._settings["vdwtype"] == "lj_12_6"):
                f.write("pair_style lj/charmm/coul/gauss/long %10.4f %10.4f %10.4f %10.4f\n\n\n\n" %
                    (self._settings["cut_lj_inner"], self._settings["cut_lj"],self._settings["coul_smooth"], self._settings["cut_coul"]))
            elif self._settings["chargetype"] == "point":
                f.write("pair_style buck/coul/long %10.4f\n\n" % (self._settings["cutoff"]))
                #f.write("pair_style buck/coul/long %10.4f\n\n" % (14.0))
            else:
                raise NotImplementedError
        else:
            # use shift damping (dsf)

            if self._mol.ff.settings["coreshell"] == True:
                f.write("\npair_style buck6d/coul/gauss/dsf/cs %10.4f %10.4f\n\n" % (self._settings["vdw_smooth"], self._settings["cutoff"]))
            elif (self._settings["vdwtype"] == "buck6d")|(self._settings["vdwtype"] == "exp6_damped"):
                if self._settings['cutoff_coul'] is not None:
                    f.write("\npair_style buck6d/coul/gauss/dsf %10.4f %10.4f %10.4f\n\n" % (self._settings["vdw_smooth"], self._settings["cutoff"],self._settings['cutoff_coul']))
                else:
                    f.write("\npair_style buck6d/coul/gauss/dsf %10.4f %10.4f\n\n" % (self._settings["vdw_smooth"], self._settings["cutoff"]))
            elif self._settings["vdwtype"] == "lj_12_6":
                f.write("\npair_style lj/charmm/coul/gauss/dsf %10.4f %10.4f %10.4f\n\n" %
                     (self._settings["cut_lj_inner"], self._settings["cut_lj"], self._settings["cut_coul"]))
        pairstrings = self.pairterm_formatter(comment = True)
        for s in pairstrings: f.write((s+"\n"))
#        for i, ati in enumerate(self.plmps_atypes):
#            for j, atj in enumerate(self.plmps_atypes[i:],i):
#                r0, eps, alpha_ij = self.plmps_pair_data[(i+1,j+1)]
#                A = self._settings["vdw_a"]*eps
#                B = self._settings["vdw_b"]/r0
#                C = eps*self._settings["vdw_c"]*r0**6
#                D = 6.0*(self._settings["vdw_dampfact"]*r0)**14
#                f.write(("pair_coeff %5d %5d " + self.parf(5) + "   # %s <--> %s\n") % (i+1,j+1, A, B, C, D, alpha_ij, ati, atj))            
        # bond style
        if len(list(self.par_types["bnd"].keys())) > 0: f.write("\nbond_style hybrid class2 morse harmonic\n\n")
        for bt in list(self.par_types["bnd"].keys()):
            bt_number = self.par_types["bnd"][bt]
            for ibt in bt:
                pot_type, params = self.par["bnd"][ibt]
                if pot_type == "mm3":
                    r0 = params[1]
                    K2 = params[0]*mdyn2kcal/2.0 
                    K3 = K2*(-2.55)
                    K4 = K2*(2.55**2.)*(7.0/12.0)
                    pstring = "class2 %12.6f %12.6f %12.6f %12.6f" % (r0, K2, K3, K4)
                elif pot_type == "quartic":
                    r0 = params[1]
                    K2 = params[0]*mdyn2kcal/2.0 
                    K3 = -1*K2*params[2]
                    K4 = K2*(2.55**2.)*params[3]
                    pstring = "class2 %12.6f %12.6f %12.6f %12.6f" % (r0, K2, K3, K4)
                elif pot_type == "harm":
                    r0 = params[1]
                    K2 = params[0]*mdyn2kcal/2.0 
                    pstring = "harmonic %12.6f %12.6f" % (K2, r0)
                elif pot_type == "morse":
                    r0 = params[1]
                    E0 = params[2]
                    k  = params[0]*mdyn2kcal/2.0
                    alpha = np.sqrt(k/E0)
                    pstring = "morse %12.6f %12.6f %12.6f" % (E0, alpha, r0)
                else:
                    raise ValueError("unknown bond potential")
                f.write("bond_coeff %5d %s    # %s\n" % (bt_number, pstring, ibt))
        # angle style
        if len(list(self.par_types["ang"].keys())) > 0:
            if self._settings["vdwtype"]=="buck" or self._settings["vdwtype"]=="wangbuck" or self._settings['vdwtype']=="lj_12_6":
                f.write("\nangle_style hybrid class2/p6\n\n")
            else:
                if self._settings["use_angle_cosine_buck6d"]:
                    f.write("\nangle_style hybrid class2/p6 cosine/buck6d\n\n")
                else:
                    f.write("\nangle_style hybrid class2/p6 cosine/vdwl13\n\n")
        # f.write("\nangle_style class2/mofff\n\n")
        for at in list(self.par_types["ang"].keys()):
            at_number = self.par_types["ang"][at]
            for iat in at:
                pot_type, params = self.par["ang"][iat]
                if pot_type == "mm3":
                    th0 = params[1]
                    K2  = params[0]*mdyn2kcal/2.0 
                    K3 = K2*(-0.014)*rad2deg
                    K4 = K2*5.6e-5*rad2deg**2
                    K5 = K2*-7.0e-7*rad2deg**3
                    K6 = K2*2.2e-8*rad2deg**4
                    pstring = "%12.6f %12.6f %12.6f %12.6f %12.6f %12.6f" % (th0, K2, K3, K4, K5, K6)
                    # pstring = "%12.6f %12.6f" % (th0, K2)
                    f.write("angle_coeff %5d class2/p6    %s    # %s\n" % (at_number, pstring, iat))
                    # f.write("angle_coeff %5d    %s    # %s\n" % (at_number, pstring, iat))
                    # HACk to catch angles witout strbnd
                    if len(at) == 1:
                        f.write("angle_coeff %5d class2/p6 bb 0.0 1.0 1.0\n" % (at_number))
                        f.write("angle_coeff %5d class2/p6 ba 0.0 0.0 1.0 1.0\n" % (at_number))
                elif pot_type == "strbnd":
                    ksb1, ksb2, kss = params[:3]
                    r01, r02        = params[3:5]
                    th0             = params[5]
                    f.write("angle_coeff %5d class2/p6 bb %12.6f %12.6f %12.6f\n" % (at_number, kss*mdyn2kcal, r01, r02))
                    f.write("angle_coeff %5d class2/p6 ba %12.6f %12.6f %12.6f %12.6f\n" % (at_number, ksb1*mdyn2kcal, ksb2*mdyn2kcal, r01, r02))
                    # f.write("angle_coeff %5d bb %12.6f %12.6f %12.6f\n" % (at_number, kss*mdyn2kcal, r01, r02))
                    # f.write("angle_coeff %5d ba %12.6f %12.6f %12.6f %12.6f\n" % (at_number, ksb1*mdyn2kcal, ksb2*mdyn2kcal, r01, r02))
                elif pot_type == "fourier":
                    a0 = params[1]
                    fold = params[2]
                    k = 0.5*params[0]*angleunit*rad2deg*rad2deg/fold
                    pstring = "%12.6f %5d %12.6f" % (k, fold, a0)
                    if self._settings["use_angle_cosine_buck6d"]:
                        f.write("angle_coeff %5d cosine/buck6d   %s    # %s\n" % (at_number, pstring, iat))                        
                    else:
                        f.write("angle_coeff %5d cosine/vdwl13   %s 1.0   # %s\n" % (at_number, pstring, iat))
                else:
                    raise ValueError("unknown angle potential")
        # dihedral style
        if len(self.par_types["dih"].keys()) > 0: f.write("\ndihedral_style hybrid opls class2\n\n")
 #           f.write("\ndihedral_style opls\n\n")
        for dt in self.par_types["dih"].keys():
            dt_number = self.par_types["dih"][dt]
            # manage class2 terms if they are not set
            # the following order applies
            # [class2, mbt, ebt, at, aat, bb13]
            class2 = [False, False, False, False, False, False] 
            for idt in dt:
                pot_type, params = self.par["dih"][idt]
                pstrings = []
                if (pot_type == "cos3") & (len(dt) == 1):
                    assert len(dt) == 1, "Only class2 allows combination of potentials!"
                    v1, v2, v3 = params[:3]
                    pstring = "opls %12.6f %12.6f %12.6f %12.6f" % (v1, v2, v3, 0.0)
                elif (pot_type == "cos3") & (len(dt) > 1):
                    # use class2 if there are torsional cross terms
                    # also, change the ptype to class2
                    v1, v2, v3 = params[:3]
                    self.par["dih"][idt] = ['class2',[v1,0.0,v2,0.0,v3,0.0]]
                    class2[0] = True
                    pstring = "class2 %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f" % (v1/2.,0.0,v2/2.,0.0,v3/2.,0.0)
                elif pot_type == "cos4":
                    assert len(dt) == 1, "Only class2 allows combination of potentials!"
                    v1, v2, v3, v4 = params[:4]
                    pstring = "opls %12.6f %12.6f %12.6f %12.6f" % (v1, v2, v3, v4)
                elif pot_type == "class2":
                    class2[0] = True
                    #v for energy barrier
                    #r for rest_value
                    v1, r1, v2, r2, v3, r3 = params[:6]
                    pstring = "class2 %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f" % (v1/2.,r1,v2/2.,r2,v3/2.,r3)
                elif pot_type == "mbt":
                    kb1, kb2, kb3, r2 = params[:4]
                    pstring = "class2 mbt  %12.6f %12.6f %12.6f %12.6f" % (kb1*mdyn2kcal, kb2*mdyn2kcal, kb3*mdyn2kcal, r2)
                    class2[1] = True
                elif pot_type == "ebt":
                    kb1_1, kb1_2, kb1_3, kb2_1, kb2_2, kb2_3, r1, r3 = params[:8]
                    pstring = "class2 ebt  %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f" % (kb1_1*mdyn2kcal,
                              kb1_2*mdyn2kcal, kb1_3*mdyn2kcal, kb2_1*mdyn2kcal, kb2_2*mdyn2kcal, kb2_3*mdyn2kcal, r1, r3)
                    class2[2] = True
                elif pot_type == "at":
                    kb1_1, kb1_2, kb1_3, kb2_1, kb2_2, kb2_3, th1, th2 = params[:8]
                    pstring = "class2 at   %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f" % (kb1_1*mdyn2kcal,
                              kb1_2*mdyn2kcal, kb1_3*mdyn2kcal, kb2_1*mdyn2kcal, kb2_2*mdyn2kcal, kb2_3*mdyn2kcal, th1, th2)
                    class2[3] = True
                elif pot_type == "aat":
                    kbb, th1, th2 = params[:3]
                    pstring = "class2 aat  %12.6f %12.6f %12.6f" % (kbb*mdyn2kcal, th1, th2)
                    class2[4] = True
                elif pot_type == "bb13":
                    class2[5] = True
                    kss, r1, r3 = params[:3]
                    pstring = "class2 bb13 %12.6f %12.6f %12.6f" % (kss*mdyn2kcal, r1, r3) 
                else:
                    raise ValueError("unknown dihedral potential")
                f.write("dihedral_coeff %5d %s    # %s\n" % (dt_number, pstring, idt))
            # check if we have a class2 potential
            if True in class2:
                # we have a class2 potential, so we have to fill up the missing pots with dummies
                if class2[0] == False:
                    f.write("dihedral_coeff %5d class2 0.0 0.0 0.0 0.0 0.0 0.0 # dummy\n" % (dt_number))
                if class2[1] == False:
                    f.write("dihedral_coeff %5d class2 mbt 0.0 0.0 0.0 0.0 # dummy\n" % (dt_number))
                if class2[2] == False:
                    f.write("dihedral_coeff %5d class2 ebt 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 # dummy\n" % (dt_number))
                if class2[3] == False:
                    f.write("dihedral_coeff %5d class2 at  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 # dummy\n" % (dt_number))
                if class2[4] == False:
                    f.write("dihedral_coeff %5d class2 aat 0.0 0.0 0.0 # dummy\n" % (dt_number))
                if class2[5] == False:
                    f.write("dihedral_coeff %5d class2 bb13 0.0 0.0 0.0\n" % (dt_number))
        # improper/oop style
        if len(list(self.par_types["oop"].keys())) > 0:
            if self._settings["use_improper_umbrella_harmonic"] == True:
                f.write("\nimproper_style umbrella/harmonic\n\n")
            else:
                f.write("\nimproper_style inversion/harmonic\n\n")
        for it in list(self.par_types["oop"].keys()):
            it_number = self.par_types["oop"][it]
            for iit in it:
                pot_type, params = self.par["oop"][iit]
                if pot_type == "harm":
                    if self._settings["use_improper_umbrella_harmonic"] == True:
                        pstring = "%12.6f %12.6f" % (params[0]*mdyn2kcal, params[1])
                    else:
                        pstring = "%12.6f %12.6f" % (params[0]*mdyn2kcal*1.5, params[1])                        
                else:
                    raise ValueError("unknown improper/oop potential")
                f.write("improper_coeff %5d %s    # %s\n" % (it_number, pstring, iit))
        #f.write("\nspecial_bonds lj 0.0 0.0 1.0 coul 1.0 1.0 1.0\n\n")
        f.write("\nspecial_bonds lj %4.2f %4.2f %4.2f coul %4.2f %4.2f %4.2f\n\n" %
            (self._settings["vdw12"],self._settings["vdw13"],self._settings["vdw14"],
            self._settings["coul12"],self._settings["coul13"],self._settings["coul14"]))
        f.write("# ------------------------ MOF-FF FORCE FIELD END --------------------------\n")
        # write footer
        if footer:
            ff = open(footer, "r")
            f.write(ff.readlines())
            ff.close()
        f.close()
        # sync with other nodes
        self.mpi_comm.barrier()
        return

    def report_timer(self):
        if self.mpi_rank == 0:
            if self.print_timer is True:
                self.timer.write()
