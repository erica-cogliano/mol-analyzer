# Obiettivo: ottenere la mcs delle molecole dei farmaci del k ovarico
# la mcs deve essere fatta sui cluster dei vari farmaci per ottenere la migliore
# struttura possibile con un significato chimico e riconoscibile.
# utilizziamo murcko scaffold per ottenere scaffold piu ampi e significativi

# trovare l'mcs delle molecole in farmaci-oc.sdf
# Questo file e' una copia di filtered_sdfs.sdf proveniente dal progetto in R
# manualmente modificato per rimuovere molecole vuote.

from loguru import logger
from helper import *

InitLogger()

# Idea:
# SDF -> Fingerprint -> Matrice di distanze -> Cluster -> MCS per cluster -> Draw

# carica il file farmaci-oc in mols
mols = SDMolSupplier("data/farmaci-oc.sdf")
logger.info("Caricate {} molecole".format(len(mols)))

# Filtra le molecole per rimuovere quelle che non sono state caricate correttamente
mols = [m for m in mols if m is not None]
logger.info("Dopo il filtraggio, rimangono {} molecole".format(len(mols)))

# Otteniamo le fingerprint per ogni molecola
fingerprints = GetFingerprintFromSDMol(mols)

# Otteniamo la matrice di distanze
compressed_distance_matrix = GetDistanceMatrixFromFingerprints(fingerprints)

expended_distance_matrix = ExpandDistanceMatrix(
    compressed_distance_matrix, len(fingerprints)
)

clusters = GetKMeansClustersFromDistanceMatrix(expended_distance_matrix, 16)

for i, cluster in enumerate(clusters):
    logger.info("Cluster {}: {}".format(i, cluster))

# L'MCS trova la più grande sottostruttura comune tra due o più molecole.
# Cercare l'MCS su molecole intere potrebbe dare un risultato troppo
# piccolo o "sporcato" dalle catene laterali.
# Usare Bemis-Murcko prima dell'MCS permette di trovare il nucleo comune degli scheletri,
# ignorando le variazioni dei sostituenti.

scaffolds = GetScaffoldsFromSDMol(mols)

# Se invece vogliamo concentrarci solo sulla topologia dello scheletro, ignorando
# anche i tipi di atomi e di legami, possiamo usare MakeScaffoldGeneric
# generic_scaffolds= MakeScaffoldsGeneric(scaffolds)

# otteniamo gli MCS di ogni cluster
cluster_mcss = GetClustersMCS(clusters, mols)
DrawClustersMCS(cluster_mcss)
