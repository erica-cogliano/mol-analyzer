"""Genera il markdown con le informazioni su tutti i farmaci di TTD."""
from rdkit import Chem

from helper import *
from kovarian_clusters import LoadKOvarianMolecules

def main():
    InitLogger()

    kovarian_mols = LoadKOvarianMolecules()

    DrawMols(kovarian_mols)

    GenDrugsMarkdown(kovarian_mols)


def GenDrugsMarkdown(kovarian_mols):
    """Genera il file markdown con l'elenco dei farmaci e i file markdown per ogni farmaco."""
    drugs_md_path = "docs/drugs.md"
    with open(drugs_md_path, "w") as f:
        f.write("# Farmaci di TTD\n\n")
        f.write("Questa pagina contiene un elenco di tutti i farmaci presenti nel database TTD.\n\n")
        f.write("## Elenco dei farmaci\n\n")
        for mol in tqdm(kovarian_mols, desc="Generando markdown dei farmaci"):
            # Nome della molecola
            name = GetMolName(mol)

            # Nome della molecola sanificato
            safe_name = GetSafeName(name)

            # Scrivi elemento della lista con link.
            f.write(f"- [{name}](drugs/{safe_name}.md)\n")

            GenMolMarkdown(mol)


def GenMolMarkdown(mol: Chem.Mol):
    """Genera un file markdown per una molecola specifica."""
    name = GetMolName(mol)
    safe_name = GetSafeName(name)

    # Nel contempo, genera un file markdown per ogni farmaco nella cartella `docs/drugs/`
    drug_md_path = f"docs/drugs/{safe_name}.md"
    with open(drug_md_path, "w") as drug_file:
        property_names = mol.GetPropNames()

        drug_file.write(f"![Immagine di {name}](images/{safe_name}.png)\n")

        drug_file.write("## Properties\n")
        for prop_name in property_names:
            prop_value = mol.GetProp(prop_name)
            drug_file.write(f"- **{prop_name}**: `{prop_value}`\n")




if __name__ == "__main__":
    main()