"""
In questo modulo, eseguiamo due task. A partire dai interessanti MCS dei cluster di molecole:
1. Troviamo quali compound in `compunds.sdf` contengono quel MCS.
2. Misuriamo quanto ogni molecola del dataset e' simile a quel MCS, usando Tanimoto similarity tra la fingerprint del MCS e quella di ogni molecola.
"""

import csv
from loguru import logger
from helper import *


def FindClusterMCSByClusterId(cluster_mcss: list[ClusterMCS], cluster_id: int) -> ClusterMCS:
    """Trova l'MCS del cluster con id `cluster_id` nella lista `cluster_mcss`."""
    for cluster_mcs in cluster_mcss:
        if cluster_mcs.cluster_id == cluster_id:
            return cluster_mcs
    return None


def GetInterestingClusterMCS(cluster_mcss: list[ClusterMCS], interesting_clusters) -> list[ClusterMCS]:
    """Restituisce gli MCS dei cluster di molecole che sono stati identificati come interessanti."""
    interesting_cluster_mcss = []
    for id in interesting_clusters:
        cluster_mcs = FindClusterMCSByClusterId(cluster_mcss, id)
        if cluster_mcs is not None:
            interesting_cluster_mcss.append(cluster_mcs)
        else:
            logger.warning(f"Non ho trovato l'MCS del cluster con id {id}")
    return interesting_cluster_mcss


def FindSubstructInMols(substruct: Mol, mols: list[Mol]) -> list[Mol]:
    """Trova quali compound in `mols` contengono (o includono) la sottostruttura rappresentata da `substruct`.
    Restituisce la lista di molecole che contengono quella sottostruttura."""
    return [mol for mol in mols if mol.HasSubstructMatch(substruct)]


def GetSimilarityWithMols(mol: Mol, other_mols: list[Mol]) -> list[float]:
    """
    Misura quanto ogni molecola della lista `other_mols` e' simile alla molecola rappresentata da `mol`,
    usando Tanimoto similarity tra la fingerprint del MCS e quella di ogni molecola.
    In output avremo una lista di numeri in virgola mobile (floating point numbers o float)
    e.g. [0.2, 0.02]
    """
    mol_fingerprint = GetFingerprint(mol)
    other_fingerprints = GetFingerprintsFromMols(other_mols)

    return DataStructs.BulkTanimotoSimilarity(
        mol_fingerprint, other_fingerprints
    )


def ProcessSimilarityWithMols(cluster_mcs: ClusterMCS, other_mols: list[Mol], out_prefix: str = "other_mols"):
    """
    Questa funzione analizza la similarita' tra l'MCS del cluster in `cluster_mcs` con tutte le altre
    molecole nella lista `other_mols`.
    Il risultato viene salvato in un file CSV nella cartella "out" il cui nome e' `cluster_<id>_<out_prefix>.csv`
    """
    # Analizziamo la similarita' di un MCS di un cluster con le altre molecole
    sims: list[float] = GetSimilarityWithMols(cluster_mcs.mcs_mol, other_mols)
    
    # Salva la similarita' appena calcolata in un file csv
    cluster_id = cluster_mcs.cluster_id
    csv_file_path = f"out/cluster_{cluster_id}_vs_{out_prefix}.csv"
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
    
    logger.info(f"Similarita' scritte nel file {csv_file_path}")
