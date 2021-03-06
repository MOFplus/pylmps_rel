# pylmps

Pylmps is a Python wrapper for lammps ( i.e. for the python interface to lammps). It is made to have a similar API as our legacy code [pydlpoly](https://github.com/MOFplus/pydlpoly). It is named to differentiate from the [PyLammps](https://docs.lammps.org/Howto_pylammps.html). 
Note that pylmps is working fully parallel (with mpi4py) and contains numerous features to simplify simulations especially with our MOF-FF force field. The actual lammps input is generated by pylmps. We also use a special trajectory format called mfp5 (for MOFplus-HDF5) to store basically all info in compact form. Therefore, you need to use our own variant of lammps to work with pylmps. 


## Installing

In order to install pylmps and in particular to compile the proper lammps please download the following components:

- [pylmps](https://github.com/MOFplus/pylmps_rel)
- [lammps](https://github.com/MOFplus/lammps) -> checkout new_Sept2021
- [molsys](https://github.com/MOFplus/molsys_rel)
- [mofplus](https://github.com/MOFplus/mofplus_rel)

Or simply use the global repo [cmc-tools](https://github.com/MOFplus/cmc-tools) and follow the README there

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

- Rochus Schmid
- Johannes P. Dürholt
- Julian Keupp
- Roberto Amabile
- Sandro Wieser
- Gunnar Schmitz
- Vanessa Angenent
- Babak Farhadi-Jahromi
- Larissa Schaper

## License

MIT License

## Acknowledgments

many people helping us, testing code, giving ideas ...
