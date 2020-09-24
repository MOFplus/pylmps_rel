import molsys
import pylmps


# Note: This will test an loose optimization using ReaxFF in pylmps

ref_energies = { 'reax_bond' : -9446.648273662398
               , 'Coulomb'   : -35.65053237466263
               }

m = molsys.mol.from_file("phenantrene.xyz")

pl = pylmps.pylmps("reax")
pl.setup(local=True, mol=m, ff="ReaxFF", use_pdlp=True, origin="center")

pl.MIN(0.1)

energies = pl.get_energy_contribs()


for e in ref_energies.keys():
    assert abs(ref_energies[e]-energies[e])<1.0e-6
