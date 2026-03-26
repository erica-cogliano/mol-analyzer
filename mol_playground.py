from helper import *


def PatchSDFWithIndications(sdf_path: str, csv_path: str, output_sdf_path: str):
    """
    Questa funzione aggiunge l'indicazione alle molecole di un file SDF.

    L'indicazione viene presa da un file CSV che contiene due colonne: "drug_name" e "indications".
    Per ogni molecola nel file SDF, viene preso il nome della molecola e viene cercato nel file CSV.
    Se viene trovato, viene aggiunta una nuova proprietà alla molecola con chiave "TTD_INDICATION" e valore l'indicazione trovata nel CSV.
    Il risultato viene salvato in un nuovo file SDF.
    """
    # Carica l'SDF
    supplier = SDMolSupplier(sdf_path)

    # Carica il CSV
    ec20_csv = pd.read_csv(csv_path)

    def GetIndication(csv: pd.DataFrame, mol_name: str) -> str:
        # Dal csv, prendi la riga che nella colonna "drug_name" contiene `mol_name`, e prendi il valore della colonna "indications"
        return str(csv[csv["drug_name"] == mol_name]["indications"].iloc[0])


    modified_mols = []

    for mol in supplier:
        print(GetMolName(mol))
        print(GetMolProperty(mol, "PUBCHEM_IUPAC_NAME"))

        if GetMolProperty(mol, "TTD_INDICATION") is None:
            print("La molecola non ha l'indicazione TTD, aggiungo..")
            indication_to_set = GetIndication(ec20_csv, GetMolName(mol))
            SetMolProperty(mol, "TTD_INDICATION", indication_to_set)

        print(GetMolProperty(mol, "TTD_INDICATION"))

        modified_mols.append(mol)

    # Salva le nuove molecole con indicazione nell'output path
    SaveMolsToSDF(modified_mols, output_sdf_path)


PatchSDFWithIndications(
    sdf_path="data/EC20.sdf",
    csv_path="data/EC20.csv",
    output_sdf_path="data/EC20_with_indications.sdf"
)

PatchSDFWithIndications(
    sdf_path="data/approved_non_k_ovarian_drugs.sdf",
    csv_path="data/ttd_approved_non_k_ovarian_drugs.csv",
    output_sdf_path="data/approved_non_k_ovarian_drugs_with_indications.sdf"
)