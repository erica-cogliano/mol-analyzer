# Obiettivo: ottenere la mcs delle molecole dei farmaci del k ovarico
# la mcs deve essere fatta sui cluster dei vari farmaci per ottenere la migliore
# struttura possibile con un significato chimico e riconoscibile.
# utilizziamo murcko scaffold per ottenere scaffold piu ampi e significativi

# trovare l'mcs delle molecole in farmaci-oc.sdf
# Questo file e' una copia di filtered_sdfs.sdf proveniente dal progetto in R
# manualmente modificato per rimuovere molecole vuote.

from helper import *

# Idea:
# SDF -> Fingerprint -> Matrice di distanze -> Cluster -> MCS per cluster -> Draw

# carica il file farmaci-oc in mols
mols = SDMolSupplier("data/farmaci-oc.sdf")
print("Caricate {} molecole".format(len(mols)))

# Otteniamo le fingerprint per ogni molecola
fingerprints = GetFingerprintFromSDMol(mols)

# Otteniamo la matrice di distanze
compressed_distance_matrix = GetDistanceMatrixFromFingerprints(fingerprints)

expended_distance_matrix = ExpandDistanceMatrix(
    compressed_distance_matrix, len(fingerprints)
)

clusters = GetKMeansClustersFromDistanceMatrix(expended_distance_matrix, 10)

print("Creati {} clusters".format(len(clusters)))
for cluster in clusters:
    print(cluster)

# otteniamo gli MCS di ogni cluster
cluster_mcss = GetClustersMCS(clusters, mols)
DrawClustersMCS(cluster_mcss)
