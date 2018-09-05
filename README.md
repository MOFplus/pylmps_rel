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

cd to the lammps directory and install via

```
cd src

make yes-USER-COLVARS
make yes-CLASS2
make yes-KSPACE
make yes-MANYBODY
make yes-MC
make yes-USER-MOFFF
make yes-USER-MISC
make yes-MOLECULE

  --- by default we install the colvars package. very useful!

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


## Contributing

TBA

## License

TBA

## Acknowledgments

TBA
