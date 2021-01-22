# pylmps

Pylmps is a Python wrapper for lammps ( i.e. for the python interface to lammps). It is made to have a similar syntax as [pydlpoly](https://github.com/MOFplus/pydlpoly).


## Installing

In order to install pylmps clone this repository into destination of your choosing (we always use /home/%USER/sandbox/, also the installation instructions below use this path)

```
https://github.com/MOFplus/pylmps.git
```
or if you have put an ssh key to github
```
git@github.com:MOFplus/pylmps.git
```

Afterwards the PATH and PYTHONOATH have to be updated. Add to your .bashrc :
```
export PYTHONPATH=/home/$USER/sandbox/pylmps:$PYTHONPATH
export PATH=/home/$USER/sandbox/pylmps/scripts:$PATH
```

Mandatory dependencies are:

* [LAMMPS](https://github.com/MOFplus/lammps)
* [molsys](https://github.com/MOFplus/molsys) 

## Installing LAMMPS

Lammps needs to be built as a shared library. The following list is our default way of compiling it 
(version from 2017 or newer necessary, since the USER-MOFFF package is from 2017)
(Note: we include ReaxFF by default ... omit if you are sure that you do not need it)

### Clone from github

First clone the repo from github.com:MOFplus/lammps with
```
git clone https://github.com/MOFplus/lammps
```

In case you need to use the mttk barostat as implemented by the Ghent group you need now to pull from
Steven Vandenbrande's git repository. First cd into the lammps directory.
```
git pull https://github.com/stevenvdb/lammps newbarostat
```
(or use git with ssh if you have a key on github)
You will have to commit the merge .. Currently this merges without problem, however, if it does not than changes in lammps
prevent this and you need to carefully resolve!! Let us know!!!

IMPORTANT: We have observed some issues in regular simulations when this new MTTK barostat is included. Therefore, DO NOT PUSH this
repo back to MOFplus once this new barostat was pulled. Use a second repo without it to develop.

### Compile the code

Follow these steps one by one:

```
cd lammps

# the following is necessary to successfully install this lammps version even if you do not want to use h5md or hdf5
cd lib/h5md
make -f Makefile.h5cc

# In order to access the hdf5 files you have to edit the file Makefile.lammps
# Mine looks like this (UBUNTU based MINT):
#  h5md_SYSINC = -I/usr/include/hdf5/serial
#  h5md_SYSLIB = -L/usr/lib/x86_64-linux-gnu/hdf5/serial/ -lhdf5
# For a typical conda environment in your local directory (repalce <USER> by your username)
#  h5md_SYSINC = -I/home/<USER>/conda/envs/molsys/include
#  h5md_SYSLIB = -L/home/<USER>/conda/envs/molsys/lib -lhdf5
#  the first entry points to the leader file (try "locate hdf5.h" to find it)
#  the second entry points to the lib (try "locate libhdf5")

cd ../../src

make yes-USER-COLVARS
make yes-CLASS2
make yes-KSPACE
make yes-MANYBODY
make yes-MC
make yes-USER-MOFFF
make yes-USER-MISC
make yes-MOLECULE
make yes-USER-H5MD
make yes-USER-REAXC
make yes-PYTHON
make yes-REPLICA

#  --- by default we install the colvars package. very useful!

cd ../lib/colvars/
make -f Makefile.g++
cd ../../src/

# For using with python3 you have to copy the Makefile.lammps.python3 to Makefile.lammps in /lib/python
# In case of using python3.8 or higher you need to add the option --embed for python3-config in addition to --ldflags
# in the line where python_SYSLIB is defined (This is important especially for the conda environment, where python3.8 is used)

# compile as shared lib 
make mode=shlib mpi
cd ../python
ln -s ../src/liblammps.so .
ln -s ../src/liblammps_mpi.so .
cd ..
```

### Setup PYTHONPATH

afterwards the ./python directory has to be added to the pythonpath
```
export PYTHONPATH=/home/$USER/sandbox/lammps/python:$PYTHONPATH
```

## Running the tests

There will soon be a testing framework framework available.

## Building the Documentation
Mandatory dependencies to built the documentationare
```
pip install Sphinx
pip install sphinx-rtd-theme
```

As soon as there is a documentation it can be compiled by running
```
make html
```
in the doc folder.
A Built directory containing
```
/built/html/index.html 
```
was created. It can be opened with the browser of your choice



## Contributing

TBA

## License

TBA

## Acknowledgments

TBA
