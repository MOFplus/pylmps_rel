#!/usr/bin/python3 -i
# -*- coding: utf-8 -*-

###########################################################################
#
#  Script to batch convert multiple lammps .dump trajectory files to hdf5
#  and store all of them in the same pdlp file (must exist beforehand)
#
############################################################################

import numpy 
import os
import shutil
import subprocess
import sys
import molsys


backup_pdlp=True

dumps = [x for x in os.listdir('.') if x.rsplit('.',1)[-1] == 'dump']
mfpxs = [x for x in os.listdir('.') if x.rsplit('.',1)[-1] == 'mfpx']
pdlps = [x for x in os.listdir('.') if x.rsplit('.',1)[-1] == 'pdlp']

if len(pdlps) != 1:
    print('multiple or no pdlp(s) found, exiting... ')
    os._exit(0)

pdlp = pdlps[0]
if len(mfpxs) == 0:
    print('no mfpx found! exiting')
    os._exit(0)

if len(mfpxs) != 1:
    print('multiple mfpxs detected, select the one to be used')
    for i,mfpx in enumerate(mfpxs):
        print('%3d %s' % (i,mfpx))
    inp = input('select mfpx [int]')
    try: 
        i = int(inp)
    except:
        print('invalid input. give an integer!')
        os._exit(0)
    mfpx = mfpxs[i]
else:
    mfpx = mfpxs[0]

if backup_pdlp is True:
    # do not backup if there is one already!
    if len([x for x in os.listdir('.') if x.rsplit('.',1)[-1] == 'pdlp_keep'])   == 0:
        os.system('cp %s %s_keep' % (pdlp,pdlp))
    
for dump in dumps:  
    name = mfpx.rsplit('.',1)[0]
    stagename = dump.rsplit('.',1)[0]
    s = 'add_dump_to_pdlptraj.py %s %s %s %s' % (name,dump,pdlp,stagename)
    os.system(s)
    print(s)
