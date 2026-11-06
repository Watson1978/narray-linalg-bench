"""Write the STO-3G integrals that the Hartree-Fock fixture is built from.

PySCF is not a runtime dependency. This runs once, in a throwaway venv, and
its output (script/hartree_fock_sto3g.json) is tracked so that
script/make_fixtures.rb can rebuild data/ without it.

  python3 -m venv /tmp/pyscf && /tmp/pyscf/bin/pip install pyscf
  /tmp/pyscf/bin/python script/hartree_fock_integrals.py
"""

import json
import os

import pyscf
from pyscf import gto, scf

MOLECULES = {
    "h2": {"atom": "H 0 0 0; H 0 0 1.4", "charge": 0},
    "heh": {"atom": "He 0 0 0; H 0 0 1.4632", "charge": 1},
}


def integrals(spec):
    mol = gto.M(atom=spec["atom"], unit="Bohr", basis="sto-3g", charge=spec["charge"], verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    energy = mf.kernel()
    return {
        "S": mol.intor("int1e_ovlp").tolist(),
        "H": (mol.intor("int1e_kin") + mol.intor("int1e_nuc")).tolist(),
        "eri": mol.intor("int2e", aosym="s1").tolist(),
        "e_nuc": mol.energy_nuc(),
        "n_electrons": mol.nelectron,
        "e_total": energy,
    }


def main():
    out = {"source": f"PySCF {pyscf.__version__}, RHF, STO-3G, Bohr",
           "molecules": {name: integrals(spec) for name, spec in MOLECULES.items()}}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hartree_fock_sto3g.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    for name, mol in out["molecules"].items():
        print(name, mol["e_total"])


if __name__ == "__main__":
    main()
