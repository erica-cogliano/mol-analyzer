# Obiettivo: ottenere la mcs delle molecole dei farmaci del k ovarico
# la mcs deve essere fatta sui cluster dei vari farmaci per ottenere la migliore
# struttura possibile con un significato chimico e riconoscibile.
# utilizziamo murcko scaffold per ottenere scaffold piu ampi e significativi


from loguru import logger
from helper import *
import mcs_search

CLUSTER_COUNT = 24
RANDOM_STATE = 10
INTERESTING_CLUSTER_IDS = [3, 5, 10, 16 , 17]


def GetInterestingClusters() -> list[ClusterMCS]:
    """
    Restituisce la lista di ClusterMCS di piccole molecole associate al k ovarico
    che sono stati identificati come interessanti in base a criteri arbitrari.
    """
    kovarian_mols = LoadMolecules(
        file_path="data/compounds.sdf",
        fallback_path="data/ttd_drug_disease_ovarian_by_drug.csv",
    )
    logger.info("Caricate {} molecole".format(len(kovarian_mols)))

    # Filtra le molecole per rimuovere quelle che non sono state caricate correttamente
    kovarian_mols = GetListFromSDMolSupplier(kovarian_mols)
    logger.info("Dopo il filtraggio, rimangono {} molecole".format(len(kovarian_mols)))

    # Ottieni gli MCS dei cluster a partire dalle molecole filtrate
    k_ovarian_cluster_mcss: list[ClusterMCS] = GetClustersMCSFromMols(
        kovarian_mols, cluster_count=CLUSTER_COUNT, random_state=RANDOM_STATE
    )

    # Seleziona i cluster piu' interessanti
    return mcs_search.GetInterestingClusterMCS(k_ovarian_cluster_mcss, INTERESTING_CLUSTER_IDS)


def main():
    InitLogger()

    # Idea:
    # SDF -> Fingerprint -> Matrice di distanze -> Cluster -> MCS per cluster -> Draw

    # trovare l'mcs delle molecole in compounds.sdf
    # Questo file e' una collezione di molecole che sono state testate contro il k ovarico,
    # e che hanno mostrato una certa efficacia. L'obiettivo e' trovare la struttura comune a queste molecole,
    # che potrebbe essere la chiave per capire come funzionano e per progettare nuovi farmaci.
    # Se il file non esiste, viene creato scaricando le molecole dal database PubChem usando il nome dei farmaci
    # nel file "data/ttd_drug_disease_ovarian_by_drug.csv"
    kovarian_mols = LoadMolecules(
        file_path="data/compounds.sdf",
        fallback_path="data/ttd_drug_disease_ovarian_by_drug.csv",
    )
    logger.info("Caricate {} molecole".format(len(kovarian_mols)))
    DrawMols(kovarian_mols)

    PrintBiggestMol(kovarian_mols)

    # Filtra le molecole per rimuovere quelle che non sono state caricate correttamente
    kovarian_mols = GetListFromSDMolSupplier(kovarian_mols)
    logger.info("Dopo il filtraggio, rimangono {} molecole".format(len(kovarian_mols)))

    k_ovarian_cluster_mcss: list[ClusterMCS] = GetClustersMCSFromMols(
        kovarian_mols, cluster_count=CLUSTER_COUNT, random_state=RANDOM_STATE
    )

    DrawClustersMCS(k_ovarian_cluster_mcss)

    interesting_cluster_mcss = mcs_search.GetInterestingClusterMCS(
        k_ovarian_cluster_mcss, INTERESTING_CLUSTER_IDS
    )

    for cluster_mcs in interesting_cluster_mcss:
        # Cerco la "molecola che rappresenta l'MCS del cluster" nelle molecole del dataset
        mols_with_mcs = mcs_search.FindSubstructInMols(
            cluster_mcs.mcs_mol, kovarian_mols
        )
        logger.info(
            f"Trovate {len(mols_with_mcs)} molecole che contengono l'MCS del cluster {cluster_mcs.cluster_id}"
        )
        for mol in mols_with_mcs:
            logger.info(f"- {GetMolName(mol)}")

        mcs_search.ProcessSimilarityWithMols(
            cluster_mcs, kovarian_mols, "kovarian_mols"
        )


# La particolarita' di `if __name__ == "__main__":` e' che permette di eseguire il codice solo
# quando il file viene eseguito direttamente, e non quando viene importato come modulo in un altro file.
# In questo modo, possiamo definire la funzione `FilterApprovedNonKOvarianDrugs` in questo file, e poi
# importarla e usarla in altri file senza eseguire il codice di filtraggio ogni volta che importiamo questo modulo.
if __name__ == "__main__":
    main()
