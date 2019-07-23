# pylmps

Pylmps is a pyhton wrapper for lammps ( i.e. for the python interface to lammps). It is made to have a similar syntax as [pydlpoly](https://github.com/MOFplus/pydlpoly).

### Installing

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

* [LAMMPS](https://lammps.sandia.gov/download.html)
* [molsys](https://github.com/MOFplus/molsys) 

### Installing LAMMPS

Lammps needs to be built as a shared library. The following list is our default way of compiling it 
(version from 2017 or newer necessary, since the USER-MOFFF package is from 2017)

Note on installing Lammps with the mttk barostat implementation of the ghent group
* clone (this repository)[https://github.com/stevenvdb/lammps].
* switch to the newbarostat branch before you continue with the rest of the installation via
```
git checkout newbarostat
```
cd to the lammps directory and install via

```

#Follow these steps to add the h5md extension:      # this is necessary to successfully install this lammps version even if you do not want to use h5md or hdf5

cd lammps
# in the lammps dir 
cd lib/h5md
# compile the ch5md library with 
make -f Makefile.h5cc
# In order to access the hdf5 files you might have to edit the file Makefile.lammps
# ###!!!###!!!###!!!### CHANGE IN FILE  
# Mine looks like this (UBUNTU based MINT):

#  h5md_SYSINC = -I/usr/include/hdf5/serial
#  h5md_SYSLIB = -L/usr/lib/x86_64-linux-gnu/hdf5/serial/ -lhdf5

#  the first entry points to the leader file (try "locate hdf5.h" to find it)
#  the second entry points to the lib (try "locate libhdf5")
cd ../../src

#cd src

make yes-USER-COLVARS
make yes-CLASS2
make yes-KSPACE
make yes-MANYBODY
make yes-MC
make yes-USER-MOFFF
make yes-USER-MISC
make yes-MOLECULE
make yes-USER-REAXC

#  --- by default we install the colvars package. very useful!

cd ../lib/colvars/
make -f Makefile.g++
cd ../../src/
make mode=shlib mpi
cd ../python
ln -s ../src/liblammps.so .
ln -s ../src/liblammps_mpi.so .
cd ..
```
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
