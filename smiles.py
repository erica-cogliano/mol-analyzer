from helper import *
from rdkit import Chem

SDF_FILE_PATH = "data/compounds.sdf"
COMPOUND_NAME = "Lurtotecan"

def main():
    InitLogger()

    mols = SDMolSupplier(SDF_FILE_PATH)
    for mol in mols:
        name = GetMolName(mol)
        logger.info(f"Checking molecule: {name}")
        if name == COMPOUND_NAME:
            smiles = Chem.MolToSmiles(mol)
            print(f"SMILES for {COMPOUND_NAME}: {smiles}")
            break

if __name__ == "__main__":
    main()