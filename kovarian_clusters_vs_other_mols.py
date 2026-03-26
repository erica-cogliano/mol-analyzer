#obiettivo: confrontare gli scaffold dei cluster con le molecole del dataset iniziale (con farmaci solo approvati e senza la classe k ovarico),
#per vedere se ci sono molecole che contengono quegli scaffold e se sono simili a quelli scaffold

import csv
import os
import kovarian_clusters
from filtra_ttd import *
from helper import *
from mcs_search import *


# passi da effettuare:
# 1. ripulire il file ttd_drug_disease_by_drug.csv da tutte le molecole che non sono state approvate o che appartengono alla classe k ovarico

# Se questo file non esiste, creiamolo
if not os.path.exists(APPROVED_NON_K_OVARIAN_CSV_FILE_PATH):
    FilterApprovedNonKOvarianDrugs(ALL_DRUGS_CSV_FILE_PATH, APPROVED_NON_K_OVARIAN_CSV_FILE_PATH)
    assert os.path.exists(APPROVED_NON_K_OVARIAN_CSV_FILE_PATH), "Il file filtrato non è stato creato correttamente."

# 2. Caricare `ttd_approved_non_k_ovarian_drugs.csv` e ottenere la lista di nomi di molecole
#    - A partire da quella lista di nomi di molecole scaricare le MOL da PubChem e salvarle in un file SDF.
#    - se il file SDF esiste già, caricarlo invece di scaricare le MOL.

APPROVED_NON_K_OVARIAN_SDF_FILE_PATH = "data/approved_non_k_ovarian_drugs_with_indications.sdf"

supplier = LoadMolecules(
    APPROVED_NON_K_OVARIAN_SDF_FILE_PATH,
    fallback_path=APPROVED_NON_K_OVARIAN_CSV_FILE_PATH,
    # Essendo piu' di 3 mila farmaci, escludiamo la ricerca
    # nel dominio delle substance per evitare di spendere troppo tempo
    include_substances=False
)

other_mols: list[Mol] = GetListFromSDMolSupplier(supplier)

images_out_dir = "out/other_mols"
if not os.path.exists(images_out_dir):
    DrawMols(other_mols, images_out_dir)

logger.info(f"Caricate {len(other_mols)} molecole da '{APPROVED_NON_K_OVARIAN_SDF_FILE_PATH}'")


# Recuperare gli MCS dei cluster ottenuti in `murckoscaffold.py` e confrontarli con le molecole caricate da `approved_non_k_ovarian_drugs.sdf`
kovarian_interseting_clusters: list[ClusterMCS] = kovarian_clusters.GetInterestingClusters()

for interesting_cluster_mcs in kovarian_interseting_clusters:
    # Vediamo se ogni cluster e' incluso nelle molecole caricate da `approved_non_k_ovarian_drugs.sdf`
    other_mols_with_mcs = FindSubstructInMols(interesting_cluster_mcs.mcs_mol, other_mols)
    logger.info(
        f"Trovate {len(other_mols_with_mcs)} molecole che contengono l'MCS del cluster {interesting_cluster_mcs.cluster_id} nei farmaci approvati non k ovarici"
    )
    for mol in other_mols_with_mcs:
        # Per capire quale patologia e' associata a `mol`
        # 1 prendi il nome di mol
        mol_name = GetMolName(mol)
        mol_indication = GetMolProperty(mol, "TTD_INDICATION")
        # 2 cerca il nome di mol nel file `approved_non_k_ovarian_drugs.csv` e prendi la patologia associata
        #  - carica il file `approved_non_k_ovarian_drugs.csv`
        #  - trova la riga che nella colonna drug_name contiene il nome di questa molecola
        #  - prendi da quella riga il valore della colonna disease_name indications

        logger.info(f"- {mol_name} -> {mol_indication}")

    
    # Analizziamo la similarita' di un MCS di un cluster con le altre molecole
    sims: list[float] = GetSimilarityWithMols(interesting_cluster_mcs.mcs_mol, other_mols)
    
    # Salva la similarita' appena calcolata in un file csv
    cluster_id = interesting_cluster_mcs.cluster_id
    csv_file_path = f"out/tanimoto_similarities_cluster_{cluster_id}.csv"
    with open(csv_file_path, "w") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "drug_name",
                "indication",
                f"cluster_{cluster_id}_similarity"
            ]
        )

        for i, other_mol in enumerate(other_mols):
            mol_name = GetMolName(other_mol)
            mol_indication = GetMolProperty(other_mol, "TTD_INDICATION")

            csv_writer.writerow(
                [
                    mol_name,
                    mol_indication,
                    sims[i]
                ]
            )
