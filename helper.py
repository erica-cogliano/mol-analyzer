from rdkit.Chem import rdFMCS, rdFingerprintGenerator, Draw, SDMolSupplier, DataStructs
from rdkit.ML.Cluster import Butina


# Definiamo una funzione che ci trasforma le molecole di un SDF in un vettore di fingerprint
# @input: SDMol
# @output: Vettore di fingerprint
def GetFingerprintFromSDMol(mols):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for mol in mols:
        fp = gen.GetFingerprint(mol)
        fps.append(fp)
    return fps


# Definiamo una funzione che generi la matrice di similarita' di tanimoto a partire da fingerprints
# input: fingerprints
# output: matrice di similarita' in formato adatto a Butina.ClusterData
def GetSimilarityMatrixFromFingerprints(fingerprints):
    sim_mat = []
    n_fps = len(fingerprints)

    for i in range(1, n_fps):
        # Per ogni fingerprint, calcolo la sua similarita' con tutte le altre fingerprint
        current_fp = fingerprints[i]
        similarita_con_altre = DataStructs.BulkTanimotoSimilarity(
            current_fp, fingerprints[:i]
        )
        sim_mat.extend(similarita_con_altre)

    return sim_mat


# Definiamo una funzione che generi la matrice delle distanze a partire da fingerprints
# input: vettore di fingerprint
# output: matrice di distanze in formato adatto a Butina.ClusterData
def GetDistanceMatrixFromFingerprints(fingerprints):
    dist_mat = []
    sim_mat = GetSimilarityMatrixFromFingerprints(fingerprints)

    for sim_value in sim_mat:
        # Per ogni valore della matrice di similarita'
        # Trasformo il valore di similarita' in un valore di distanza con la seguente formula
        dist_value = 1.0 - sim_value
        dist_mat.append(dist_value)

    return dist_mat


# Definiamo una funzione che genera i cluster con Butina a partire da una matrice delle distanze
# input:
#  - distance_matrix: matrice delle distanze in formato adatto a Butina.ClusterData
#  - n: Numero di righe e colonne della matrice
# output: clusters
def GetClustersFromDistanceMatrix(distance_matrix, n, dist_tresh):
    return Butina.ClusterData(
        distance_matrix, n, distThresh=dist_tresh, isDistData=True
    )
