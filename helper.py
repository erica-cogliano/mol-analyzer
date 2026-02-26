from rdkit.Chem import (
    rdFMCS,
    rdFingerprintGenerator,
    Draw,
    SDMolSupplier,
    DataStructs,
    MolFromSmarts,
)
from rdkit.ML.Cluster import Butina

from sklearn.manifold import MDS
from sklearn.cluster import KMeans


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


# Questa funzione espande una distance matrix da un formato Butina ad un formato adatto a MDS e KMeans
# input:
#  - compressed_distance_matrix: distance matrix adatto a Butina.ClusterData
#  - n: Numero di righe e colonne della matrice
# output: matrice completa
def ExpandDistanceMatrix(compressed_distance_matrix, n):
    expanded_matrix = []

    for i in range(n):
        new_row = []
        for j in range(n):
            if i == j:
                new_row.append(0.0)
            elif i < j:
                # Butina distance matrix is compressed, so we need to calculate the index in the compressed format
                index = int((j * (j - 1)) / 2 + i)
                new_row.append(compressed_distance_matrix[index])
            else:
                # The distance matrix is symmetric, so we can use the value from the upper triangle
                index = int((i * (i - 1)) / 2 + j)
                new_row.append(compressed_distance_matrix[index])
        expanded_matrix.append(new_row)

    return expanded_matrix


# Definiamo una funzione che genera i cluster con KMeans a partire da una matrice delle distanze
# input:
#  - distance_matrix: matrice delle distanze in formato adatto a MDS e KMeans
#  - cluster_count: numero di cluster da generare
# output: clusters
def GetKMeansClustersFromDistanceMatrix(distance_matrix, cluster_count):
    clusters = []

    mds = MDS(dissimilarity="precomputed")
    fprints_2d = mds.fit_transform(distance_matrix)

    km = KMeans(cluster_count)
    # KMeans fit predict restituisce un vettore di cluster a cui ogni molecola appartiene,
    # ad esempio [0, 0, 1, 1, 2] significa che le prime due molecole appartengono al cluster 0,
    # le successive due al cluster 1 e l'ultima al cluster 2
    clusters = km.fit_predict(fprints_2d)

    # Ora vogliamo trasformare questo vettore di cluster in una lista di cluster,
    # dove ogni cluster e' una tupla di indici di molecole
    cluster_dict = {}
    for idx, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(idx)

    # Convertiamo il dizionario in una lista di tuple
    clusters_list = [tuple(v) for v in cluster_dict.values()]

    return clusters_list


# definiamo una funzione che genera MCSs a partire dai clusters
# input:
# -clusters: un vettore di tuple di indici di molecole
# -mols: contenitore di molecole di cui gli indici fanno riferimento
# output:
# -MCSs
def GetClustersMCS(clusters, mols):
    cluster_results = []
    for i, cluster in enumerate(clusters):
        # un cluster e' una tupla di indici (es.(1,5,10))
        if len(cluster) < 2:
            # salta i clustert con una sola molecola
            continue
        # trasforma cluster (che contiene indici delle molecole) in un vettore di molecole vere e proprie
        mols_in_cluster = [mols[idx] for idx in cluster]
        # trova la MCS per questo specifico cluster
        mcs_res = rdFMCS.FindMCS(mols_in_cluster)
        # converti il risultato MCS in una molecola visualizzabile
        mcs_mol = MolFromSmarts(mcs_res.smartsString)
        cluster_results.append(
            {
                "cluster_id": i,
                "size": len(cluster),
                "mcs_smarts": mcs_res.smartsString,
                "mcs_mol": mcs_mol,
            }
        )
    return cluster_results


# Funzione che disegna gli MCS di ogni cluster
# input: clusters_mcs - risultato della funzione GetClustersMCS
def DrawClustersMCS(clusters_mcs):
    for cluster_mcs in clusters_mcs:
        mcs = cluster_mcs["mcs_mol"]
        id = cluster_mcs["cluster_id"]
        cluster_size = cluster_mcs["size"]
        img = Draw.MolToImage(mcs, legend=f"MCS Cluster {id} (size {cluster_size})")
        img.show()
